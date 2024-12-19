import json
import copy
import argparse
import os
from MyGraph import Graph
from Block import Block
from AutoSplit import Splitter
import tempfile
from pipeline_schedule import set_active_engine
from pipeline_schedule import pipeline_schedule
from performance import estimate_model
from performance import print_performance

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--schema_path", nargs='?', default="utils/schema.fbs")
parser.add_argument("--out_path")
parser.add_argument("--exec_order", nargs='?', default="DF")
parser.add_argument("--split_height", nargs='?', type=int, default=2)
parser.add_argument("--token_size", nargs='?', type=int, default=50)
parser.add_argument("--model_type", nargs='?', type=str, default="bert")
parser.add_argument("--pad_fusion", action='store_true')
parser.add_argument("--verbose_performance", action='store_true')
parser.add_argument("--block_based", action='store_true')

args = parser.parse_args()
filename = os.path.basename(args.model_path)
model_name = os.path.splitext(filename)[0]
schema_path = args.schema_path

if args.model_type == "bert":
    model_type = 0
elif args.model_type == "yolo":
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

tensor_id_mapping = [ x for x in range(len(tensors))]

ori_graph = Graph(operators, tensors, buffers, new_opcodes, subgraphs[0]['inputs'], subgraphs[0]['outputs'], args.exec_order)

splitter = Splitter(ori_graph, args.split_height, model_type, args.token_size)
if args.pad_fusion and model_type == 1:
    splitter.PaddingFusion()

# Decide each block's range from ori_graph (Block: one entry point and one exit point)
if args.block_based:
    graph_to_blocks = Block(ori_graph, model_type)
    blocks = graph_to_blocks.blocks
else:
    blocks = []

new_graph = splitter.perform_split(blocks)

# Set each operator's active engine for performance estimation
set_active_engine(new_graph)

split_dma_cycles, split_op_cycles, split_total_cycles, engine_idle_cycles = estimate_model(new_graph, pipeline = False)
if args.verbose_performance:
    print(f"Before pipeline schedule: dma cycles = {split_dma_cycles :.1f}, op cycles = {split_op_cycles :.1f}, total cycles = {split_total_cycles :.1f}")
    print(f"mac_idle_cycles: {engine_idle_cycles[0]}, elem_wise_idle_cycles: {engine_idle_cycles[1]}")
pipeline_new_graph = pipeline_schedule(new_graph)
if args.verbose_performance:
    pipeline_dma_cycles, pipeline_op_cycles, pipeline_total_cycles, engine_idle_cycles = estimate_model(pipeline_new_graph, pipeline = True)
    print_performance(pipeline_new_graph)
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
new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = pipeline_new_graph.export()

new_model['buffers'] = new_buffers
new_model['subgraphs'][0]['tensors'] = new_tensors
new_model['subgraphs'][0]['inputs'] = new_inputs
new_model['subgraphs'][0]['outputs'] = new_outputs
new_model['subgraphs'][0]['operators'] = new_operators

# Record the cascade ops and matched ops for CodeGen use
with open(pattern_path, 'w') as f:
    json.dump({'cascade_matched_ops': pipeline_new_graph.cascade_matched_ops, 'matched_ops': pipeline_new_graph.matched_ops}, f)
with open(json_model_path, 'w') as f:
    json.dump(new_model, f, indent=2)

os.system(f'flatc -o {tmp_dir_path} --binary {schema_path} {json_model_path}')
os.system(f'mv {os.path.join(tmp_dir_path, filename)} {args.out_path}')
pattern_json = os.path.basename(args.out_path).replace('.tflite', '_pattern.json')
pattern_out_path = f'{os.path.dirname(args.out_path)}/{pattern_json}'
os.system(f'mv {pattern_path} {pattern_out_path}')

tmp_dir.cleanup()
