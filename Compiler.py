import json
import copy
import argparse
import os
from MyGraph import Graph
from AutoSplit import Splitter
import tempfile
from Normal_schedule import Normal_scheduler
from Weight_reuse_schedule import Weight_reuse_scheduler
from Pipeline_schedule import set_active_engine
from Pipeline_schedule import pipeline_schedule
from Simulator import simulator
from memory_allocation import memory_allocator

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--schema_path", nargs='?', default="utils/schema.fbs")
parser.add_argument("--out_path")
parser.add_argument("--exec_order", nargs='?', default="DF")
parser.add_argument("--split_height", nargs='?', type=int, default=2)
parser.add_argument("--token_size", nargs='?', type=int, default=50)
parser.add_argument("--model_type", nargs='?', type=str, default="bert")
parser.add_argument("--pad_fusion", action='store_true')
parser.add_argument("--move_data_layout_op", action='store_true')
parser.add_argument("--verbose_performance", action='store_true')

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

# Don't know why DistilGPT2 model's opcodes's builtin_code is missing, so add it back
builtin_code_mapping = {6: 'DEQUANTIZE', 9: 'FULLY_CONNECTED', 18: 'MUL', 22: 'RESHAPE', 25: 'SOFTMAX', \
                        28: 'TANH', 36: 'GATHER', 39: 'TRANSPOSE', 40: 'MEAN', 41: 'SUB', \
                        42: 'DIV', 49: 'SPLIT', 65: 'SLICE', 76: 'RSQRT', 78: 'POW', \
                        83: 'PACK', 99: 'SQUARED_DIFFERENCE'}
for opcode in opcodes:
    if opcode.get('builtin_code', None) == None:
        deprecated_builtin_code = opcode.get('deprecated_builtin_code', -1)
        if deprecated_builtin_code in builtin_code_mapping.keys():
            opcode['builtin_code'] = builtin_code_mapping[deprecated_builtin_code]

# Don't know why opcode_index 0 is missing, so add it back
for operator in operators:
    if 'opcode_index' not in operator:
        operator['opcode_index'] = 0

new_opcodes = copy.deepcopy(opcodes)

has_split = False
has_concat = False

for opcode in opcodes:
    if opcode.get('deprecated_builtin_code',0) == 2:
        has_concat = True
    elif opcode.get('deprecated_builtin_code',0) ==49:
        has_split = True

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


new_model = copy.deepcopy(model)

new_model['operator_codes'] = new_opcodes

ori_graph = Graph(operators, tensors, buffers, new_opcodes, subgraphs[0]['inputs'], subgraphs[0]['outputs'], args.exec_order)

splitter = Splitter(ori_graph, args.split_height, model_type, args.token_size)
if args.pad_fusion and model_type == 1:
    splitter.PaddingFusion()
# When emcount bert model, try to eliminate some data layout ops
if args.move_data_layout_op and model_type == 0:
    splitter.Elminate_useless_data_layout_op()

################## BASELINE ##################
new_graph = splitter.perform_split()

# Memory allocation(not perform cache optimization)
mem_allocator = memory_allocator(new_graph)

# Set each operator's active engine
set_active_engine(new_graph)

# Estimate the performance before pipeline schedule
model_sim = simulator(new_graph, mem_allocator.need_allocate_tensors)
if args.verbose_performance:
    split_dma_cycles, split_op_cycles, split_total_cycles, engine_idle_cycles = model_sim.estimate_model(pipeline = False)
    # model_sim.print_performance()
    print(f"Original: dma cycles = {split_dma_cycles :.1f}, op cycles = {split_op_cycles :.1f}, total cycles = {split_total_cycles :.1f}")
    # print(f"mac_idle_cycles: {engine_idle_cycles[0]}, elem_wise_idle_cycles: {engine_idle_cycles[1]}")
##############################################

################## NORMAL SCHEDULE ##################
# This normal scheduler won't change execution order, just set the tensor's storage area
normal_graph = copy.deepcopy(new_graph)
normal_need_allocate_tensors = copy.deepcopy(mem_allocator.need_allocate_tensors)
normal_scheduler = Normal_scheduler(normal_graph, normal_need_allocate_tensors)
normal_graph = normal_scheduler.normal_schedule()

model_sim = simulator(normal_graph, normal_scheduler.tensor_info)
if args.verbose_performance:
    normal_dma_cycles, normal_op_cycles, normal_total_cycles, engine_idle_cycles = model_sim.estimate_model(pipeline = False)
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
    reuse_dma_cycles, reuse_op_cycles, reuse_total_cycles, engine_idle_cycles = model_sim.estimate_model(pipeline = False)
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
pipeline_new_graph = pipeline_schedule(weight_reuse_graph)

# Estimate the performance after pipeline schedule
model_sim = simulator(pipeline_new_graph, weights_reuse_need_allocate_tensors)
if args.verbose_performance:
    pipeline_dma_cycles, pipeline_op_cycles, pipeline_total_cycles, engine_idle_cycles = model_sim.estimate_model(pipeline = True)
    # model_sim.print_performance()
    print(f"cascade ops = {pipeline_new_graph.cascade_matched_ops}")
    print(f"match ops = {pipeline_new_graph.matched_ops}")
    total_fusion_ops = 0
    total_match_ops = 0
    for cascade_ops in pipeline_new_graph.cascade_matched_ops:
        total_fusion_ops += len(cascade_ops)
    for match_ops in pipeline_new_graph.matched_ops:
        total_match_ops += len(match_ops)
    print(f"total fusion ops = {total_fusion_ops}, total match ops = {total_match_ops}, total ops = {len(pipeline_new_graph.operators)}")
    print(f"After pipeline schedule: dma cycles = {pipeline_dma_cycles :.1f}, op cycles = {pipeline_op_cycles :.1f}, total cycles = {pipeline_total_cycles :.1f}")
    print(f"mac_idle_cycles: {engine_idle_cycles[0]}, elem_wise_idle_cycles: {engine_idle_cycles[1]}")
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
