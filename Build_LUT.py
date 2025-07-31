# Build LUT

from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import numpy as np
import math

class LUT:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def build_lut(self):
        self.build_exp_lut()
        self.build_reciprocal_lut()
        self.build_rsqrt_lut()
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    # Dequant to the real value, calculate the real result by math function, then quantize it back to the quantized value and store it in the LUT
    def build_exp_lut(self):
        exp_ops = self.find_exp_op()
        for i, op in enumerate(exp_ops):
            exp_input_tensor_id = op.info['inputs'][0]
            exp_output_tensor_id = op.info['outputs'][0]
            int_ifm_scale = self.tensors[exp_input_tensor_id]['quantization']['scale'][0]
            ifm_scale = np.int32(int_ifm_scale).view('float32')
            ifm_zp = self.tensors[exp_input_tensor_id]['quantization']['zero_point'][0]
            int_ofm_scale = self.tensors[exp_output_tensor_id]['quantization']['scale'][0]
            ofm_scale = np.int32(int_ofm_scale).view('float32')
            ofm_zp = self.tensors[exp_output_tensor_id]['quantization']['zero_point'][0]

            results = []
            quantized_min = -128
            quantized_max = 127
            for input in range(-128, 128):
                input_real = (input - ifm_zp) * ifm_scale
                # Avoid overflow happening
                try:
                    output_real = math.exp(input_real)
                except OverflowError:
                    output_real = float('inf')
                if output_real == float('inf'):
                    output_result = 127
                else:
                    output_quantized = int(round(output_real / ofm_scale + ofm_zp))
                    output_result = min(max(output_quantized, quantized_min), quantized_max)
                results.append(output_result)

            # Create the LUT buffer
            lut_shape = [256]
            np_results = np.array(results, dtype = np.int8).tobytes()
            lut_buffer = {"data": list(np_results)}
            self.buffers.append(lut_buffer)
            lut_buffer_id = len(self.buffers) - 1
            # Create the LUT tensor
            lut_tensor = {
                "shape": lut_shape,
                "type": "INT8",
                "buffer": lut_buffer_id,
                "name": "exp_lut_%d" % i,
            }
            self.tensors.append(lut_tensor)
            lut_tensor_id = len(self.tensors) - 1

            # Update the input tensors of the exp op
            op.info['inputs'].append(lut_tensor_id)

    def build_reciprocal_lut(self):
        reciprocal_ops = self.find_reciprocal_op()
        for i, op in enumerate(reciprocal_ops):
            reciprocal_input_tensor_id = op.info['inputs'][0]
            reciprocal_output_tensor_id = op.info['outputs'][0]
            int_ifm_scale = self.tensors[reciprocal_input_tensor_id]['quantization']['scale'][0]
            ifm_scale = np.int32(int_ifm_scale).view('float32')
            ifm_zp = self.tensors[reciprocal_input_tensor_id]['quantization']['zero_point'][0]
            int_ofm_scale = self.tensors[reciprocal_output_tensor_id]['quantization']['scale'][0]
            ofm_scale = np.int32(int_ofm_scale).view('float32')
            ofm_zp = self.tensors[reciprocal_output_tensor_id]['quantization']['zero_point'][0]

            results = []
            quantized_min = -128
            quantized_max = 127
            for input in range(-128, 128):
                input_real = (input - ifm_zp) * ifm_scale
                if input_real == 0:
                    output_real = 127
                else:
                    output_real = 1.0 / input_real
                output_quantized = int(round(output_real / ofm_scale + ofm_zp))
                output_result = min(max(output_quantized, quantized_min), quantized_max)
                results.append(output_result)
                    
            # Create the LUT buffer
            lut_shape = [256]
            np_results = np.array(results, dtype = np.int8).tobytes()
            lut_buffer = {"data": list(np_results)}
            self.buffers.append(lut_buffer)
            lut_buffer_id = len(self.buffers) - 1
            # Create the LUT tensor
            lut_tensor = {
                "shape": lut_shape,
                "type": "INT8",
                "buffer": lut_buffer_id,
                "name": "reciprocal_lut_%d" % i,
            }
            self.tensors.append(lut_tensor)
            lut_tensor_id = len(self.tensors) - 1

            # Update the input tensors of the reciprocal op
            op.info['inputs'].append(lut_tensor_id)

    def build_rsqrt_lut(self):
        rsqrt_ops = self.find_rsqrt_op()
        for i, op in enumerate(rsqrt_ops):
            rsqrt_input_tensor_id = op.info['inputs'][0]
            rsqrt_output_tensor_id = op.info['outputs'][0]
            int_ifm_scale = self.tensors[rsqrt_input_tensor_id]['quantization']['scale'][0]
            ifm_scale = np.int32(int_ifm_scale).view('float32')
            ifm_zp = self.tensors[rsqrt_input_tensor_id]['quantization']['zero_point'][0]
            int_ofm_scale = self.tensors[rsqrt_output_tensor_id]['quantization']['scale'][0]
            ofm_scale = np.int32(int_ofm_scale).view('float32')
            ofm_zp = self.tensors[rsqrt_output_tensor_id]['quantization']['zero_point'][0]

            results = []
            quantized_min = -128
            quantized_max = 127
            for input in range(-128, 128):
                input_real = (input - ifm_zp) * ifm_scale
                if input_real <= 0:
                    output_real = 127
                else:
                    output_real = 1.0 / math.sqrt(input_real)
                output_quantized = int(round(output_real / ofm_scale + ofm_zp))
                output_result = min(max(output_quantized, quantized_min), quantized_max)
                results.append(output_result)

            # Create the LUT buffer
            lut_shape = [256]
            np_results = np.array(results, dtype = np.int8).tobytes()
            lut_buffer = {"data": list(np_results)}
            self.buffers.append(lut_buffer)
            lut_buffer_id = len(self.buffers) - 1
            # Create the LUT tensor
            lut_tensor = {
                "shape": lut_shape,
                "type": "INT8",
                "buffer": lut_buffer_id,
                "name": "rsqrt_lut_%d" % i,
            }
            self.tensors.append(lut_tensor)
            lut_tensor_id = len(self.tensors) - 1

            # Update the input tensors of the rsqrt op
            op.info['inputs'].append(lut_tensor_id)

    def find_exp_op(self):
        exp_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "EXP":
                # There may have some exp ops' inputs had been dequantized, we only handle the int8 input
                exp_input_tensor_id = op.info['inputs'][0]
                if self.tensors[exp_input_tensor_id].get('type', "FLOAT32") == "INT8":
                    exp_ops.append(op)
        return exp_ops
    
    def find_reciprocal_op(self):
        reciprocal_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "RECIPROCAL":
                reciprocal_ops.append(op)
        return reciprocal_ops
    
    def find_rsqrt_op(self):
        rsqrt_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "RSQRT":
                rsqrt_ops.append(op)
        return rsqrt_ops