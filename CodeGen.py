import json
import argparse
import os
import tempfile
from memory_allocation import memory_allocator
from OPGen import OPGen

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--out_path")
parser.add_argument("--schema_path", nargs='?', default='utils/schema.fbs')
args = parser.parse_args()

filename = os.path.basename(args.model_path)
model_name = os.path.splitext(filename)[0]
schema_path = args.schema_path

tmp_dir = tempfile.TemporaryDirectory(dir='.')
tmp_dir_path = tmp_dir.name

if os.path.splitext(filename)[1] != '.tflite':
    raise "input model path doesn't match: .tflite extension is required"

json_model_path = os.path.join(tmp_dir_path, f'{model_name}.json')
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
        operator['builtin_options_type'] = 'ReshapeOptions'

memory_allocator = memory_allocator(model)
allocated_tensor = memory_allocator.allocated_tensor

npu_code = ""
opgen = OPGen(model, allocated_tensor, npu_code)
for operator in operators:
    opgen.op_codegen(operator)

with open(args.out_path, 'w') as f:
    f.write(opgen.npu_code)

tmp_dir.cleanup()