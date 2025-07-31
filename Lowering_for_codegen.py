# Lowering NPU un-supported ops to supported ones

from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import copy
import numpy as np

class Lowering_for_codegen:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def lowering(self):
        self.tanh_convert()
        self.gelu_lowering()

    # tanh(x) = 2 * logistic(2 * x) - 1
    def tanh_convert(self):
        tanh_ops = self.find_tanh_op()
        for i, op in enumerate(tanh_ops):
            tanh_input_tensor_id = op.info['inputs'][0]
            tanh_output_tensor_id = op.info['outputs'][0]

            # Steps 1: Create logistic op
            # Steps 1.1: Perform the scale folding in the input tensor
            logistic_input_tensor = self.tensors[tanh_input_tensor_id]
            logistic_input_tensor['name'] = 'tanh_lower_logistic_%d' % i
            ori_float_scale = np.int32(logistic_input_tensor['quantization']['scale'][0]).view('float32')
            float_scale = 2 * ori_float_scale
            np_int_scale = np.float32(float_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            logistic_input_tensor['quantization']['scale'] = int_scale
            logistic_input_tensor_id = tanh_input_tensor_id
            # Steps 1.2: Calculate the output tensor's scale and zero point
            # real_tanh_value = (real_logistic_value * 2) - 1, since logistic belongs to (0, 1), tanh belongs to (-1, 1)
            # => S_out * (q - Z_out) = 2 * (S_log_out * (q - Z_log_out)) - 1
            # S_log_out = S_out / 2
            # Z_log_out = Z_out - 1 / S_out
            logistic_output_tensor = self.tensors[tanh_output_tensor_id]
            ori_float_scale = np.int32(logistic_output_tensor['quantization']['scale'][0]).view('float32')
            float_scale = ori_float_scale / 2
            np_int_scale = np.float32(float_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            logistic_output_tensor['quantization']['scale'] = int_scale
            ori_zero_point = logistic_output_tensor['quantization']['zero_point'][0]
            logistic_output_zero_point = round(ori_zero_point - 1 / ori_float_scale)
            logistic_output_tensor['quantization']['zero_point'] = [logistic_output_zero_point]
            logistic_output_tensor_id = tanh_output_tensor_id
            # Steps 1.3: Create the logistic op
            logistic_op = {
                "opcode_index": self.get_opcode_index(14),
                "inputs": [logistic_input_tensor_id],
                "outputs": [logistic_output_tensor_id]
            }
            self.operators.append(logistic_op)
            logistic_opid = len(self.operators) - 1
            self.ops.append(Node(logistic_op, logistic_opid))

            # Update the parent and children of the lower ops
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(logistic_opid)
                    self.ops[logistic_opid].parents.append(parent_opid)
                    break

            child_opid = op.children[0]
            for i, opid in enumerate(self.ops[child_opid].parents):
                if opid == op.opid:
                    del self.ops[child_opid].parents[i]
                    self.ops[child_opid].parents.append(logistic_opid)
                    self.ops[logistic_opid].children.append(child_opid)
                    break
        
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    # gelu(x) = x * logistic(1.702 * x)
    def gelu_lowering(self):
        gelu_ops = self.find_gelu_op()
        for i, op in enumerate(gelu_ops):
            gelu_input_tensor_id = op.info['inputs'][0]
            gelu_output_tensor_id = op.info['outputs'][0]

            # Steps 1: Create reshape op to perform the scale folding (because it needs to keep the x)
            # Steps 1.1: Create shape tensor of the reshape op (same to the gelu's input tensor shape)
            reshape_shape = self.tensors[gelu_input_tensor_id]['shape']
            shape_array = np.array(reshape_shape, dtype = np.int32).astype(np.int32).tobytes()
            shape_buffer = {"data": list(shape_array)}
            self.buffers.append(shape_buffer)
            shape_buffer_id = len(self.buffers) - 1
            shape_tensor = {
                "shape": [len(reshape_shape)],
                "type": "INT32",
                "buffer": shape_buffer_id,
                "name": "gelu_lower_reshape_shape_%d" % i
            }
            self.tensors.append(shape_tensor)
            shape_tensor_id = len(self.tensors) - 1
            # Steps 1.2: Create the output tensor of the reshape op
            reshape_output_tensor = copy.deepcopy(self.tensors[gelu_input_tensor_id])
            reshape_output_tensor['name'] = 'gelu_lower_reshape_%d' % i
            ori_float_scale = np.int32(reshape_output_tensor['quantization']['scale'][0]).view('float32')
            float_scale = 1.702 * ori_float_scale
            np_int_scale = np.float32(float_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype = np.int32)
            int_scale = np_arr.tolist()
            reshape_output_tensor['quantization']['scale'] = int_scale
            self.tensors.append(reshape_output_tensor)
            reshape_output_tensor_id = len(self.tensors) - 1
            # Steps 1.3: Create the reshape op
            reshape_op = {
                "opcode_index": self.get_opcode_index(22),
                "inputs": [gelu_input_tensor_id, shape_tensor_id],
                "outputs": [reshape_output_tensor_id],
                "builtin_options_type": "ReshapeOptions"
            }
            self.operators.append(reshape_op)
            reshape_opid = len(self.operators) - 1
            self.ops.append(Node(reshape_op, reshape_opid))

            # Steps 2: Create logistic op
            # Steps 2.1: Create the output tensor of the logistic op
            logistic_output_tensor = copy.deepcopy(self.tensors[reshape_output_tensor_id])
            logistic_output_tensor['name'] = 'gelu_lower_logistic_%d' % i
            # Calculate the output tensor's scale
            logistic_output_scale = 1 / 127.0
            np_int_scale = np.float32(logistic_output_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype = np.int32)
            int_scale = np_arr.tolist()
            logistic_output_tensor['quantization']['scale'] = int_scale
            logistic_output_tensor['quantization']['zero_point'] = [0]
            self.tensors.append(logistic_output_tensor)
            logistic_output_tensor_id = len(self.tensors) - 1
            # Steps 2.2: Create the logistic op
            logistic_op = {
                "opcode_index": self.get_opcode_index(14),
                "inputs": [reshape_output_tensor_id],
                "outputs": [logistic_output_tensor_id]
            }
            self.operators.append(logistic_op)
            logistic_opid = len(self.operators) - 1
            self.ops.append(Node(logistic_op, logistic_opid))

            # Steps 3: Create the mul op
            mul_op = {
                "opcode_index": self.get_opcode_index(18),
                "inputs": [gelu_input_tensor_id, logistic_output_tensor_id],
                "outputs": [gelu_output_tensor_id],
                "builtin_options_type": "MulOptions"
            }
            self.operators.append(mul_op)
            mul_opid = len(self.operators) - 1
            self.ops.append(Node(mul_op, mul_opid))

            # Update the parent and children of the lower ops
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(reshape_opid)
                    self.ops[reshape_opid].parents.append(parent_opid)
                    self.ops[parent_opid].children.append(mul_opid)
                    self.ops[mul_opid].parents.append(parent_opid)
                    break
            self.ops[reshape_opid].children.append(logistic_opid)
            self.ops[logistic_opid].parents.append(reshape_opid)
            self.ops[logistic_opid].children.append(mul_opid)
            self.ops[mul_opid].parents.append(logistic_opid)

            child_opid = op.children[0]
            for i, opid in enumerate(self.ops[child_opid].parents):
                if opid == op.opid:
                    del self.ops[child_opid].parents[i]
                    self.ops[child_opid].parents.append(mul_opid)
                    self.ops[mul_opid].children.append(child_opid)
                    break
        
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    def find_tanh_op(self):
        tanh_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "TANH":
                tanh_ops.append(op)
        return tanh_ops
    
    def find_gelu_op(self):
        gelu_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "GELU":
                gelu_ops.append(op)
        return gelu_ops

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'