from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import copy
import numpy as np

class Mean:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def convert_mean_to_depthwise_conv(self):
        # Find all mean ops
        mean_ops = self.find_mean_op()
        for i, op in enumerate(mean_ops):
            mean_input_tensor_id = op.info['inputs'][0]
            mean_output_tensor_id = op.info['outputs'][0]

            # Step 1: Create depthwise conv2d op to compute the mean value of the input tensor in axis
            #         Perform the div(reduce_size) with output tensor's quantization scale
            axis_tensor = self.tensors[op.info['inputs'][1]]
            axis = self.buffers[axis_tensor['buffer']]['data'][0]
            reduce_size = self.tensors[mean_input_tensor_id]['shape'][axis]

            # Step 1.1: Prepare for the depthwise conv2d op's needed info
            input_shape = self.tensors[mean_input_tensor_id]['shape']
            # If the input tensor is 3D, change it to 4D, and if the mean over depth-axis, left shift the channel dim
            if len(input_shape) == 3 and axis == 2:
                height = input_shape[1]
                width = input_shape[2]
                channel = 1
            else:
                raise 'unsupported mean format!!'

            # Step 1.2: Create the weight tensor of the depthwise conv2d op
            weight_shape = [1, height, width, channel]
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

            # Step 1.3: Create the output tensor of the depthwise conv2d op
            depthwise_conv2d_output_tensor = self.tensors[mean_output_tensor_id]
            depthwise_conv2d_output_tensor['name'] = 'mean_lower_depthwise_conv2d_%d' % i
            depthwise_conv2d_output_tensor['shape'][axis] = 1
            # Since Y = S_in * S_w * sigma(Xq - Z_in) = S_in * sigma(Xq - Z_in) [S_w = 1.0]
            # Div the reduce_size: Y = S_in / reduce_size * sigma(Xq - Z_in)
            # Yq = Y / S_out + Z_out = S_in / reduce_size / S_out * sigma(Xq - Z_in) + Z_out
            # Output tensor's scale is S_in / reduce_size / S_out, zero point is Z_out
            int_input_scale = self.tensors[mean_input_tensor_id]['quantization']['scale'][0]
            float_input_scale = np.int32(int_input_scale).view('float32')
            int_output_scale = self.tensors[mean_output_tensor_id]['quantization']['scale'][0]
            float_output_scale = np.int32(int_output_scale).view('float32')
            float_depthwith_conv2d_output_scale = float_input_scale / (reduce_size * float_output_scale)
            np_int_scale = np.float32(float_depthwith_conv2d_output_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            depthwise_conv2d_output_tensor['quantization']['scale'] = int_scale
            depthwise_conv2d_output_tensor_id = mean_output_tensor_id

            # Step 1.4: Create the depthwise conv2d op
            depthwise_conv2d_op = {
                "opcode_index": self.get_opcode_index(4),
                "inputs": [mean_input_tensor_id, weights_tensor_id, -1],
                "outputs": [depthwise_conv2d_output_tensor_id],
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
            self.operators.append(depthwise_conv2d_op)
            depthwise_conv2d_opid = len(self.operators) - 1
            self.ops.append(Node(depthwise_conv2d_op, depthwise_conv2d_opid))

            # Update the parent and children of the converted depthwise conv2d op
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(depthwise_conv2d_opid)
                    self.ops[depthwise_conv2d_opid].parents.append(parent_opid)
                    break
            for child_opid in op.children:
                for i, opid in enumerate(self.ops[child_opid].parents):
                    if opid == op.opid:
                        del self.ops[child_opid].parents[i]
                        self.ops[child_opid].parents.append(depthwise_conv2d_opid)
                        self.ops[depthwise_conv2d_opid].children.append(child_opid)
                        break
        
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    def find_mean_op(self):
        mean_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "MEAN":
                mean_ops.append(op)
        return mean_ops

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'

    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                print(type(num))
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length = 4, byteorder = 'little')]
        return out