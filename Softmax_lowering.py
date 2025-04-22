from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import copy
import numpy as np

class SoftMax:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def softmax_lowering(self):
        # Find all softmax ops
        softmax_ops = self.find_softmax_op()
        for i, op in enumerate(softmax_ops):
            softmax_input_tensor_id = op.info['inputs'][0]
            softmax_output_tensor_id = op.info['outputs'][0]

            # Step 1: Create max_pool op to compute the max value of the input tensor in last axis
            # No effect on the output tensor's scale and zero point (just pick the max value)
            max_pool_output_tensor = copy.deepcopy(self.tensors[softmax_input_tensor_id])
            max_pool_output_tensor['name'] = 'softmax_lower_max_pool_%d' % i
            reduce_size = max_pool_output_tensor['shape'][-1]
            # Reduce the last axis to 1
            max_pool_output_tensor['shape'][-1] = 1
            self.tensors.append(max_pool_output_tensor)
            max_pool_output_tensor_id = len(self.tensors) - 1
            max_pool_op = {
                "opcode_index": self.get_opcode_index(17),
                "inputs": [softmax_input_tensor_id],
                "outputs": [max_pool_output_tensor_id],
                "builtin_options_type": "Pool2DOptions",
                "builtin_options": {
                    "padding": "VALID",
                    "stride_w": 1,
                    "stride_h": 1,
                    "filter_width": reduce_size,
                    "filter_height": 1
                }
            }
            self.operators.append(max_pool_op)
            max_pool_opid = len(self.operators) - 1
            self.ops.append(Node(max_pool_op, max_pool_opid))

            # Step 2: Subtract the max value from the input tensor
            subtract_output_tensor = copy.deepcopy(self.tensors[softmax_input_tensor_id])
            subtract_output_tensor['name'] = 'softmax_lower_subtract_%d' % i
            # Since Xq' = Xq - Xq_max => X' = S_in * (Xq - Z_in) - S_in * (Xq_max - Z_in) = S_in * (Xq - Xq_max)
            # Output tensor's scale is same to softmax_input_scale, zero point is no needed
            subtract_output_tensor['quantization']['zero_point'] = [0]
            self.tensors.append(subtract_output_tensor)
            subtract_output_tensor_id = len(self.tensors) - 1
            subtract_op = {
                "opcode_index": self.get_opcode_index(41),
                "inputs": [softmax_input_tensor_id, max_pool_output_tensor_id],
                "outputs": [subtract_output_tensor_id],
                "builtin_options_type": "SubOptions"
            }
            self.operators.append(subtract_op)
            subtract_opid = len(self.operators) - 1
            self.ops.append(Node(subtract_op, subtract_opid))

            # Step 3: Compute the exponential of the subtracted tensor
            exp_output_tensor = copy.deepcopy(self.tensors[subtract_output_tensor_id])
            exp_output_tensor['name'] = 'softmax_lower_exp_%d' % i
            # Since exp(Xq - Xq_max)'s range is (0, 1], exp's output scale is (1.0 / 127.0), zero point is no needed
            np_int_scale = np.float32(1.0 / 127.0).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            exp_output_tensor['quantization']['scale'] = int_scale
            exp_output_tensor['quantization']['zero_point'] = [0]
            self.tensors.append(exp_output_tensor)
            exp_output_tensor_id = len(self.tensors) - 1
            exp_op = {
                "opcode_index": self.get_opcode_index(47),
                "inputs": [subtract_output_tensor_id],
                "outputs": [exp_output_tensor_id],
                "builtin_options_type": "ExpOptions"
            }
            self.operators.append(exp_op)
            exp_opid = len(self.operators) - 1
            self.ops.append(Node(exp_op, len(self.operators) - 1))

            # Step 4: Compute the sum of the exponential tensor in the last axis (use conv2d to replace reduce_sum)
            # Step 4.1: Create the weight tensor of the conv2d op
            weight_shape = [1, 1, 1, reduce_size]
            np_ones_array = np.ones(weight_shape, dtype = np.int8)
            ones_array = np_ones_array.astype(np.int8).tobytes()
            weights_buffer = {"data": list(ones_array)}
            self.buffers.append(weights_buffer)
            weights_buffer_id = len(self.buffers) - 1

            np_int_scale = np.float32(1.0).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            one_scale = np_arr.tolist()
            weights_tensor = {
                "shape": weight_shape,
                "type": "INT8",
                "buffer": weights_buffer_id,
                "name": 'mean_lower_depthwise_conv2d_weight_%d' % i,
                "quantization": {'scale': one_scale, 'zero_point': [0]}
            }
            self.tensors.append(weights_tensor)
            weights_tensor_id = len(self.tensors) - 1

            # Step 4.2: Create the output tensor of the conv2d op
            conv_output_tensor = copy.deepcopy(self.tensors[exp_output_tensor_id])
            conv_output_tensor['name'] = 'softmax_lower_conv2d_%d' % i
            conv_output_tensor['shape'][-1] = 1
            # Since summarize the last axis(reduce_size) of exp(Xq - Xq_max) = (0, reduce_size], sum's output scale is (reduce_size / 127.0), zero point is 0
            np_int_scale = np.float32(reduce_size / 127.0).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            conv_output_tensor['quantization']['scale'] = int_scale
            conv_output_tensor['quantization']['zero_point'] = [0]
            self.tensors.append(conv_output_tensor)
            conv_output_tensor_id = len(self.tensors) - 1

            # Step 4.3: Create the conv2d op
            conv_op = {
                "opcode_index": self.get_opcode_index(3),
                "inputs": [exp_output_tensor_id, weights_tensor_id, -1],
                "outputs": [conv_output_tensor_id],
                "builtin_options_type": "DepthwiseConv2DOptions",
                "builtin_options": {
                    "padding": "VALID",
                    "stride_w": 1,
                    "stride_h": 1,
                    "depth_multiplier": 1,
                    "dilation_w_factor": 1,
                    "dilation_h_factor": 1
                }
            }
            self.operators.append(conv_op)
            conv_opid = len(self.operators) - 1
            self.ops.append(Node(conv_op, conv_opid))

            # Step 5: Divide the exponential tensor by the sum tensor
            divide_op = {
                "opcode_index": self.get_opcode_index(42),
                "inputs": [exp_output_tensor_id, conv_output_tensor_id],
                "outputs": [softmax_output_tensor_id],
                "builtin_options_type": "DivOptions"
            }
            self.operators.append(divide_op)
            divide_opid = len(self.operators) - 1
            self.ops.append(Node(divide_op, divide_opid))

            # Update the parent and children of the lower ops
            # Orginal softmax op's input tensor now is the max_pool op & subtract op's input tensor
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(max_pool_opid)
                    self.ops[max_pool_opid].parents.append(parent_opid)
                    self.ops[parent_opid].children.append(subtract_opid)
                    self.ops[subtract_opid].parents.append(parent_opid)
                    break
            self.ops[max_pool_opid].children.append(subtract_opid)
            self.ops[subtract_opid].parents.append(max_pool_opid)
            self.ops[subtract_opid].children.append(exp_opid)
            self.ops[exp_opid].parents.append(subtract_opid)
            self.ops[exp_opid].children.append(conv_opid)
            self.ops[exp_opid].children.append(divide_opid)
            self.ops[conv_opid].parents.append(exp_opid)
            self.ops[conv_opid].children.append(divide_opid)
            self.ops[divide_opid].parents.append(exp_opid)
            self.ops[divide_opid].parents.append(conv_opid)
            # Original softmax op's output tensor now is the divide op's output tensor
            child_opid = op.children[0]
            for i, opid in enumerate(self.ops[child_opid].parents):
                if opid == op.opid:
                    del self.ops[child_opid].parents[i]
                    self.ops[child_opid].parents.append(divide_opid)
                    self.ops[divide_opid].children.append(child_opid)
                    break
            
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    def find_softmax_op(self):
        softmax_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            # For now, can't support that softmax op is the last op
            if opcode_type == "SOFTMAX" and len(op.children) > 0:
                softmax_ops.append(op)
        return softmax_ops

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'

    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length = 4, byteorder = 'little')]
        return out