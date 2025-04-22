import json
import copy
import argparse
import os
from MyGraph import Graph
from AutoSplit import Splitter
import tempfile
from Eliminate_data_layout_op import Eliminater
from Sink_or_Hoist import Safe_Sinker_Hoister
from TileSize_selection import TileSizeSelection
from Softmax_lowering import SoftMax
from Mean_convert import Mean
from Normal_schedule import Normal_scheduler
from Weight_reuse_schedule import Weight_reuse_scheduler
from Pipeline_schedule import set_active_engine
from Pipeline_schedule import pipeline_schedule
from GeneSys_schedule import genesys_schedule
from Simulator import simulator
from Memory_allocation import memory_allocator

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

# For tensor splitting
has_split = False
has_concat = False
# For lowering the sotfmax op
has_max_pool = False
has_sub = False
has_exp = False
has_conv = False
has_div = False
# For convert the mean op
has_depthwise_conv2d = False

for opcode in opcodes:
    if opcode.get('deprecated_builtin_code',0) == 2:
        has_concat = True
    elif opcode.get('deprecated_builtin_code',0) ==49:
        has_split = True
    elif opcode.get('deprecated_builtin_code',0) == 17:
        has_max_pool = True
    elif opcode.get('deprecated_builtin_code',0) == 41:
        has_sub = True
    elif opcode.get('deprecated_builtin_code',0) == 47:
        has_exp = True
    elif opcode.get('deprecated_builtin_code',0) == 3:
        has_conv = True
    elif opcode.get('deprecated_builtin_code',0) == 42:
        has_div = True

if has_concat == False:
    new_opcodes.append({
        "deprecated_builtin_code": 2,
        "version": 1,
        "builtin_code": "CONCATENATION"
        })
if has_split == False:
    new_opcodes.append({
        "deprecated_builtin_code": 49,
        "version": 1,
        "builtin_code": "SPLIT"
        })
if args.softmax_lowering:
    if has_max_pool == False:
        new_opcodes.append({
            "deprecated_builtin_code": 17,
            "version": 1,
            "builtin_code": "MAX_POOL_2D"
            })
    if has_sub == False:
        new_opcodes.append({
            "deprecated_builtin_code": 41,
            "version": 1,
            "builtin_code": "SUB"
            })
    if has_exp == False:
        new_opcodes.append({
            "deprecated_builtin_code": 47,
            "version": 1,
            "builtin_code": "EXP"
            })
    if has_conv == False:
        new_opcodes.append({
            "deprecated_builtin_code": 3,
            "version": 1,
            "builtin_code": "CONV_2D"
            })
    if has_div == False:
        new_opcodes.append({
            "deprecated_builtin_code": 42,
            "version": 1,
            "builtin_code": "DIV"
            })
if args.mean_convert:
    if has_depthwise_conv2d == False:
        new_opcodes.append({
            "deprecated_builtin_code": 4,
            "version": 1,
            "builtin_code": "DEPTHWISE_CONV_2D"
            })

new_model = copy.deepcopy(model)

new_model['operator_codes'] = new_opcodes

ori_graph = Graph(operators, tensors, buffers, new_opcodes, subgraphs[0]['inputs'], subgraphs[0]['outputs'], args.exec_order)

splitter = Splitter(ori_graph, model_type, args.token_size)

# Try to eliminate pad ops (only take effect on CNN model)
if args.pad_fusion and model_type == 1:
    splitter.PaddingFusion()
# When encount bert model, try to eliminate some data layout ops (only take effect on bert model)
if args.remove_data_layout_op and model_type == 0:
    Eliminater(splitter).Eliminate_useless_data_layout_op()
# Perform softmax lowering
if args.softmax_lowering:
    SoftMax(splitter).softmax_lowering()
# Perform mean convert
if args.mean_convert:
    Mean(splitter).convert_mean_to_depthwise_conv()
# Perform data layout sinking or hoisting
if args.move_data_layout_op and model_type == 0:
    Safe_Sinker_Hoister(splitter).data_layout_sink()
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

################## BASELINE ##################
new_graph = splitter.perform_split()

# Memory allocation(not perform cache optimization)
mem_allocator = memory_allocator(new_graph)

# Set each operator's active engine
set_active_engine(new_graph)

# Estimate the performance before pipeline schedule
model_sim = simulator(new_graph, mem_allocator.need_allocate_tensors)
if args.verbose_performance:
    split_dma_cycles, split_op_cycles, split_total_cycles = model_sim.estimate_model(pipeline = False)
    # model_sim.print_performance()
    print(f"Original: dma cycles = {split_dma_cycles :.1f}, op cycles = {split_op_cycles :.1f}, total cycles = {split_total_cycles :.1f}")
##############################################

################## NORMAL SCHEDULE ##################
# This normal scheduler won't change execution order, just set the tensor's storage area
normal_graph = copy.deepcopy(new_graph)
normal_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
normal_scheduler = Normal_scheduler(normal_graph, normal_need_allocate_tensors)
normal_graph = normal_scheduler.normal_schedule()

model_sim = simulator(normal_graph, normal_scheduler.tensor_info)
if args.verbose_performance:
    normal_dma_cycles, normal_op_cycles, normal_total_cycles = model_sim.estimate_model(pipeline = False)
    print(f"After use cache: dma cycles = {normal_dma_cycles :.1f}, op cycles = {normal_op_cycles :.1f}, total cycles = {normal_total_cycles :.1f}")
    print(f"speedup = {((split_total_cycles/normal_total_cycles) - 1) * 100 :.2f}%")
#####################################################

################## WEIGHTS REUSE ##################
# Perform the weight reuse schedule on the new_graph
same_layer_next_opids = splitter.same_layer_next_opids
weights_reuse_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
weight_reuse_scheduler = Weight_reuse_scheduler(new_graph, weights_reuse_need_allocate_tensors, same_layer_next_opids)
weight_reuse_graph = weight_reuse_scheduler.weight_reuse_schedule()

model_sim = simulator(weight_reuse_graph, weight_reuse_scheduler.tensor_info)
if args.verbose_performance:
    reuse_dma_cycles, reuse_op_cycles, reuse_total_cycles = model_sim.estimate_model(pipeline = False)
    # model_sim.print_performance()
    print(f"After weight reuse schedule: dma cycles = {reuse_dma_cycles :.1f}, op cycles = {reuse_op_cycles :.1f}, total cycles = {reuse_total_cycles :.1f}")
    print(f"speedup = {((split_total_cycles/reuse_total_cycles) - 1) * 100 :.2f}%")
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
    pipeline_dma_cycles, pipeline_op_cycles, pipeline_total_cycles = model_sim.estimate_model(pipeline = True)
    # model_sim.print_performance()
    if args.genesys:
        print(f"After GeneSys schedule: dma cycles = {pipeline_dma_cycles :.1f}, op cycles = {pipeline_op_cycles :.1f}, total cycles = {pipeline_total_cycles :.1f}")
    else:
        print(f"After pipeline schedule: dma cycles = {pipeline_dma_cycles :.1f}, op cycles = {pipeline_op_cycles :.1f}, total cycles = {pipeline_total_cycles :.1f}")
    print(f"speedup = {((split_total_cycles/pipeline_total_cycles) - 1) * 100 :.2f}%")
#######################################################

# Tensor allocation in DRAM (SRAM allocation is implemented above)
mem_allocator.dram_allocate(weights_reuse_need_allocate_tensors)
allocated_tensors = mem_allocator.allocated_tensors

# TODO: CodeGen
# CodeGen

# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = ori_graph.export()
new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = new_graph.export()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = normal_graph.export()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = weight_reuse_graph.export()
# new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = pipeline_new_graph.export()
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
tmp_dir.cleanup()
