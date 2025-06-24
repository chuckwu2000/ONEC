import json
import copy
import argparse
import os
from MyGraph import Graph
from AutoSplit import Splitter
import tempfile
from Init_opcodes import Init_opcodes
from Eliminate_data_layout_op import Eliminater
from Lowering_for_codegen import Lowering_for_codegen
from Softmax_lowering import SoftMax
from Mean_convert import Mean
from Logistic_lowering import Logistic
from Sink_or_Hoist import Safe_Sinker_Hoister
from TileSize_selection import TileSizeSelection
from Layer_wise_schedule import Layer_wise_scheduler
from DF_schedule import DF_scheduler
from Weight_reuse_schedule import Weight_reuse_scheduler
from Pipeline_schedule import set_active_engine
from Pipeline_schedule import pipeline_schedule
from GeneSys_schedule import genesys_schedule
from Simulator import simulator
from Memory_allocation import memory_allocator
from Distributed_SRAM_allocator import Distributed_SRAM_allocator
from CodeGen import CodeGen
from Roofline_model import RooflineModel

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--schema_path", nargs='?', default="utils/schema.fbs")
parser.add_argument("--out_path")
parser.add_argument("--exec_order", nargs='?', default="DF")
parser.add_argument("--split_height", nargs='?', type=int, required=False)
parser.add_argument("--token_size", nargs='?', type=int, default=50)
parser.add_argument("--model_type", nargs='?', type=str, default="bert")
parser.add_argument("--pad_fusion", action='store_true')
parser.add_argument("--remove_data_layout_op", action='store_true')
parser.add_argument("--move_data_layout_op", action='store_true')
parser.add_argument("--softmax_lowering", action='store_true')
parser.add_argument("--mean_convert", action='store_true')
parser.add_argument("--logistic_lowering", action='store_true')
parser.add_argument("--codegen", action='store_true')
parser.add_argument("--code_path")
parser.add_argument("--verbose_performance", action='store_true')

# Test GeneSys's options
parser.add_argument("--genesys", action='store_true')

args = parser.parse_args()
filename = os.path.basename(args.model_path)
model_name = os.path.splitext(filename)[0]
schema_path = args.schema_path

if args.model_type == "bert":
    model_type = 0
elif args.model_type == "CNN":
    model_type = 1

tmp_dir = tempfile.TemporaryDirectory(dir='.')
tmp_dir_path = tmp_dir.name

if os.path.splitext(filename)[1] != '.tflite':
    raise "input model path doesn't match: .tflite extension is required'"

json_model_path = os.path.join(tmp_dir_path, f'{model_name}.json')
pattern_path = os.path.join(tmp_dir_path, f'{model_name}_pattern.json')
os.system(f'flatc --json -o {tmp_dir_path} --raw-binary {schema_path} -- {args.model_path}')
os.system(r'sed -i "s/\([^ ]*\):/\"\1\":/" ' + json_model_path)

with open(json_model_path,'r') as f:
    model = json.load(f)

opcodes = model['operator_codes']
buffers = model['buffers']
subgraphs = model["subgraphs"]
tensors = subgraphs[0]["tensors"]
operators = subgraphs[0]["operators"]

# Don't know why ADD's info is missing, so add it back
for opcode in opcodes:
    if opcode.get('deprecated_builtin_code',0) == 0:
        opcode['deprecated_builtin_code'] = 0
        opcode['builtin_code'] = "ADD"

# Don't know why opcode_index 0 is missing, so add it back
for operator in operators:
    if 'opcode_index' not in operator:
        operator['opcode_index'] = 0

new_opcodes = copy.deepcopy(opcodes)
init_opcoder = Init_opcodes(opcodes, new_opcodes, args.codegen)
init_opcoder.init_opcodes()

ori_graph = Graph(operators, tensors, buffers, new_opcodes, subgraphs[0]['inputs'], subgraphs[0]['outputs'], args.exec_order)

splitter = Splitter(ori_graph, model_type, args.token_size)

# Try to eliminate pad ops (only take effect on CNN model)
if args.pad_fusion and model_type == 1:
    splitter.PaddingFusion()
# When encount bert model, try to eliminate some data layout ops (only take effect on bert model)
if args.remove_data_layout_op and model_type == 0:
    Eliminater(splitter).Eliminate_useless_data_layout_op()
# Perform lowering for codegen
Lowering_for_codegen(splitter).lowering()
# Perform softmax lowering
if args.softmax_lowering and not args.codegen:
    SoftMax(splitter).softmax_lowering()
# Perform mean convert
if args.mean_convert:
    Mean(splitter).convert_mean_to_conv()
# Perform data layout sinking or hoisting
if args.move_data_layout_op:
    Safe_Sinker_Hoister(splitter).data_layout_sink()
# Perform logistic lowering
if args.logistic_lowering:
    Logistic(splitter).logistic_lowering()
ori_graph = splitter.ori_graph

# Pick the best tile size
if args.split_height == None:
    tilesize_selection = TileSizeSelection(ori_graph, model_type)
    # Bert model
    if model_type == 0:
        token_size = args.token_size
    # CNN model
    elif model_type == 1:
        root_op_id = ori_graph.root_op_ids[0]
        root_op = ori_graph.ops[root_op_id]
        root_op_input_tensor_shape = ori_graph.tensors[root_op.info['inputs'][0]]['shape']
        for dim, dim_value in enumerate(root_op_input_tensor_shape):
            if dim_value > 1:
                token_size = dim_value
                break
        splitter.split_height = tilesize_selection.pick_best_tile_size(token_size)
else:
    split_height = args.split_height
splitter.split_height = split_height

################## PERFORM SPLIT ##################
new_graph = splitter.perform_split()

# Memory allocation(not perform cache optimization)
mem_allocator = memory_allocator(new_graph)

# Set each operator's active engine
set_active_engine(new_graph)
###################################################

################## BASELINE ##################
# Our baseline adopts the BF order execution order (same to layer-wise execution order)
layer_wise_graph = copy.deepcopy(new_graph)
layer_wise_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
same_layer_next_opids = splitter.same_layer_next_opids
layer_wise_scheduler = Layer_wise_scheduler(layer_wise_graph, layer_wise_need_allocate_tensors, same_layer_next_opids)
layer_wise_graph = layer_wise_scheduler.layer_wise_schedule()

# Estimate the performance of layer-wise schedule (baseline)
model_sim = simulator(layer_wise_graph, layer_wise_scheduler.tensor_info)
if args.verbose_performance:
    baseline_dma_cycles, baseline_op_cycles, baseline_total_cycles, baseline_total_energy = model_sim.estimate_model(pipeline = False)
    # model_sim.print_performance()
    print(f"Baseline schedule: dma cycles = {baseline_dma_cycles :.1f}, op cycles = {baseline_op_cycles :.1f}, total cycles = {baseline_total_cycles :.1f}")
    print(f"Baseline energy = {baseline_total_energy :.2f} nJ")
    
    roofline_model = RooflineModel(layer_wise_graph, layer_wise_scheduler.tensor_info, baseline_total_cycles)
    roofline_model.roofline_model_build()
    print(f"Turning point: operational intensity = {roofline_model.peak_operational_intensity :.2f}, \
        operations per second = {roofline_model.peak_giga_operations_per_second :.2f}")
    print(f"Model: operational intensity = {roofline_model.operational_intensity :.2f}, \
        operations per second = {roofline_model.giga_operations_per_second :.2f}")
##############################################

if not args.codegen:
    # ################## DF SCHEDULE ##################
    # df_graph = copy.deepcopy(new_graph)
    # df_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
    # df_scheduler = DF_scheduler(df_graph, df_need_allocate_tensors)
    # df_graph = df_scheduler.df_schedule()

    # model_sim = simulator(df_graph, df_scheduler.tensor_info)
    # if args.verbose_performance:
    #     df_dma_cycles, df_op_cycles, df_total_cycles = model_sim.estimate_model(pipeline = False)
    #     # model_sim.print_performance()
    #     print(f"After DF schedule: dma cycles = {df_dma_cycles :.1f}, op cycles = {df_op_cycles :.1f}, total cycles = {df_total_cycles :.1f}")
    #     print(f"speedup = {((baseline_total_cycles/df_total_cycles) - 1) * 100 :.2f}%")
    # ##############################################

    ################## WEIGHTS REUSE ##################
    # Perform the weight reuse schedule on the new_graph
    same_layer_next_opids = splitter.same_layer_next_opids
    weights_reuse_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
    weight_reuse_scheduler = Weight_reuse_scheduler(new_graph, weights_reuse_need_allocate_tensors, same_layer_next_opids)
    weight_reuse_graph = weight_reuse_scheduler.weight_reuse_schedule()

    model_sim = simulator(weight_reuse_graph, weight_reuse_scheduler.tensor_info)
    if args.verbose_performance:
        reuse_dma_cycles, reuse_op_cycles, reuse_total_cycles, reuse_total_energy = model_sim.estimate_model(pipeline = False)
        # model_sim.print_performance()
        print(f"After weight reuse schedule: dma cycles = {reuse_dma_cycles :.1f}, op cycles = {reuse_op_cycles :.1f}, total cycles = {reuse_total_cycles :.1f}")
        print(f"speedup = {((baseline_total_cycles/reuse_total_cycles) - 1) * 100 :.2f}%")
        print(f"Weight reuse energy = {reuse_total_energy :.2f} nJ")
        print(f"Weight reuse energy reduction = {((baseline_total_energy/reuse_total_energy) - 1) * 100 :.2f}%")

        roofline_model = RooflineModel(weight_reuse_graph, weight_reuse_scheduler.tensor_info, reuse_total_cycles)
        roofline_model.roofline_model_build()
        print(f"Reuse tensor Model: operational intensity = {roofline_model.operational_intensity :.2f}, \
              operations per second = {roofline_model.giga_operations_per_second :.2f}")
    ###################################################

    # print(f"Diff data reuse:")
    # for tensor_id in weight_reuse_scheduler.tensor_info:
    #     for me_id, tensor in enumerate(weight_reuse_scheduler.tensor_info[tensor_id].tensors):
    #         if tensor.in_DRAM != normal_scheduler.tensor_info[tensor_id].tensors[me_id].in_DRAM:
    #             print(f"tensor_id = {tensor_id}, pid = {tensor.pid}, cid = {tensor.cid}, in_DRAM = {tensor.in_DRAM}, normal_in_DRAM = diff")

    ################## PIPELINE SCHEDULE ##################
    # Perform software pipeline schedule
    # Test GeneSys's options
    genesys_options = {"no_intermediate_tensor_reuse": False, "can_not_followed_multi_child": False}
    if args.genesys:
        genesys_options = {"no_intermediate_tensor_reuse": True, "can_not_followed_multi_child": True}
        pipeline_new_graph = genesys_schedule(weight_reuse_graph, weights_reuse_need_allocate_tensors, genesys_options)
    else:
        pipeline_new_graph = pipeline_schedule(weight_reuse_graph)

    # Estimate the performance after pipeline schedule
    model_sim = simulator(pipeline_new_graph, weights_reuse_need_allocate_tensors)
    if args.verbose_performance:
        pipeline_dma_cycles, pipeline_op_cycles, pipeline_total_cycles, pipeline_total_energy = model_sim.estimate_model(pipeline = True)
        # model_sim.print_performance()
        if args.genesys:
            print(f"After GeneSys schedule: dma cycles = {pipeline_dma_cycles :.1f}, op cycles = {pipeline_op_cycles :.1f}, total cycles = {pipeline_total_cycles :.1f}")
        else:
            print(f"After pipeline schedule: dma cycles = {pipeline_dma_cycles :.1f}, op cycles = {pipeline_op_cycles :.1f}, total cycles = {pipeline_total_cycles :.1f}")
        print(f"speedup = {((baseline_total_cycles/pipeline_total_cycles) - 1) * 100 :.2f}%")
        print(f"Pipeline energy = {pipeline_total_energy :.2f} nJ")
        print(f"Pipeline energy reduction = {((baseline_total_energy/pipeline_total_energy) - 1) * 100 :.2f}%")

        roofline_model = RooflineModel(pipeline_new_graph, weights_reuse_need_allocate_tensors, pipeline_total_cycles)
        roofline_model.roofline_model_build()
        print(f"Pipeline Model: operational intensity = {roofline_model.operational_intensity :.2f}, \
              operations per second = {roofline_model.giga_operations_per_second :.2f}")
    #######################################################

    # Tensor allocation in DRAM (SRAM allocation is implemented above)
    mem_allocator.dram_allocate(weights_reuse_need_allocate_tensors)
    allocated_tensors = mem_allocator.need_allocate_tensors

# Generate the code for OEM's NPU
if args.codegen:
    distributed_SRAM_allocator = Distributed_SRAM_allocator(new_graph)
    distributed_SRAM_allocator.allocate_tensors()
    code_generator = CodeGen(new_graph, distributed_SRAM_allocator.tensor_info, distributed_SRAM_allocator.cascade_patterns)
    code_generator.code_gen()

# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = ori_graph.export()
new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = new_graph.export()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = layer_wise_graph.export_without_reschedule()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = df_graph.export_without_reschedule()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = weight_reuse_graph.export_without_reschedule()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = pipeline_new_graph.export_without_reschedule()
new_model = copy.deepcopy(model)
new_model['operator_codes'] = new_opcodes
new_model['buffers'] = new_buffers
new_model['subgraphs'][0]['tensors'] = new_tensors
new_model['subgraphs'][0]['inputs'] = new_inputs
new_model['subgraphs'][0]['outputs'] = new_outputs
new_model['subgraphs'][0]['operators'] = new_operators

# Save the rewritten model
with open(json_model_path, 'w') as f:
    json.dump(new_model, f, indent=2)

os.system(f'flatc -o {tmp_dir_path} --binary {schema_path} {json_model_path}')
os.system(f'mv {os.path.join(tmp_dir_path, filename)} {args.out_path}')

# Save the npu instructions
if args.codegen:
    with open(args.code_path, 'w') as f:
        f.write(code_generator.npu_code)

tmp_dir.cleanup()
