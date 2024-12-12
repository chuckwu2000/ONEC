import numpy as np
import math
from collections import defaultdict

class OPGen:
    def __init__(self, model, allocated_tensor, npu_code):
        self.model = model
        self.opcodes = model['operator_codes']
        self.tensor = model['subgraphs'][0]['tensors']
        self.buffers = model['buffers']
        self.allocated_tensor = allocated_tensor
        self.npu_code = npu_code

    def QuantizeMultiplier(self, scale):
        if scale == 0.0:
            return 0, 0
        # q: [0.5, 1]
        q, shift = math.frexp(scale)
        q_fixed = np.int32(round(q * (1 << 31)))

        if q_fixed == (1 << 31):
            q_fixed /= 2
            shift += 1

        # If the shift is smaller than -31, all bits are shifted out, so the quantized value is 0
        if shift < -31:
            shift = 0
            q_fixed = 0
        return q_fixed, shift

    def Compute_tensor_size(self, tensor):
        shape = tensor['shape']
        size = 1
        for dim in shape:
            size *= dim
        return size
    
    def CodeGen_DMA_start(self, src, size):
        buffer = f"SET_DMA_SRC_ADDR {src}\n"
        buffer += f"SET_DMA_LENGTH {size}\n"
        buffer += "DMA_START\n"
        return buffer
    
    def CodeGen_DMA_wait(self):
        buffer = "DMA_WAIT\n"
        return buffer

    def CodeGen_Set_input_tensor(self, tensor, base, zero_point):
        buffer = ""
        buffer += f"SET_IFM_BASE {base}\n"
        if len(tensor['shape']) == 4:
            buffer += f"SET_IFM_HEIGHT {tensor['shape'][1]}\n"
            buffer += f"SET_IFM_WIDTH {tensor['shape'][2]}\n"
            buffer += f"SET_IFM_DEPTH {tensor['shape'][3]}\n"
        elif len(tensor['shape']) == 2:
            buffer += f"SET_IFM_HEIGHT {tensor['shape'][0]}\n"
            buffer += f"SET_IFM_WIDTH {tensor['shape'][1]}\n"
            buffer += f"SET_IFM_DEPTH 1\n"
        buffer += f"SET_IFM_ZERO_POINT {zero_point}\n"
        return buffer
    
    def CodeGen_Set_weight_tensor(self, tensor, base, zero_point):
        buffer = ""
        buffer += f"SET_WEIGHT_BASE {base}\n"
        buffer += f"SET_WEIGHT_SIZE {self.Compute_tensor_size(tensor)}\n"
        buffer += f"SET_WEIGHT_ZERO_POINT {zero_point}\n"
        return buffer
    
    def CodeGen_Set_output_tensor(self, tensor, base, zero_point):
        buffer = ""
        buffer += f"SET_OFM_BASE {base}\n"
        if len(tensor['shape']) == 4:
            buffer += f"SET_OFM_HEIGHT {tensor['shape'][1]}\n"
            buffer += f"SET_OFM_WIDTH {tensor['shape'][2]}\n"
            buffer += f"SET_OFM_DEPTH {tensor['shape'][3]}\n"
        elif len(tensor['shape']) == 2:
            buffer += f"SET_OFM_HEIGHT {tensor['shape'][0]}\n"
            buffer += f"SET_OFM_WIDTH {tensor['shape'][1]}\n"
            buffer += f"SET_OFM_DEPTH 1\n"
        buffer += f"SET_OFM_ZERO_POINT {zero_point}\n"
        return buffer

    def fully_connected_codegen(self, operator):
        input_tensor_id = operator['inputs'][0]
        weight_tensor_id = operator['inputs'][1]
        output_tensor_id = operator['outputs'][0]

        input_tensor = self.tensor[input_tensor_id]
        weight_tensor = self.tensor[weight_tensor_id]
        output_tensor = self.tensor[output_tensor_id]

        input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
        weight_quant_scale = np.int32(weight_tensor['quantization']['scale'][0]).view('float32')
        output_qunat_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        scale = input_quant_scale * weight_quant_scale / output_qunat_scale
        multiplier, shift = self.QuantizeMultiplier(scale)

        input_quant_zp = input_tensor['quantization']['zero_point'][0]
        weight_quant_zp = weight_tensor['quantization']['zero_point'][0]
        output_quant_zp = output_tensor['quantization']['zero_point'][0]

        code = ""
        # Start to load weights
        code += self.CodeGen_DMA_start(self.allocated_tensor[weight_tensor_id].start_address, self.Compute_tensor_size(weight_tensor))
        # Set input tensor
        code += self.CodeGen_Set_input_tensor(input_tensor, self.allocated_tensor[input_tensor_id].start_address, input_quant_zp)
        # Set weight tensor
        code += self.CodeGen_Set_weight_tensor(weight_tensor, self.allocated_tensor[weight_tensor_id].start_address, weight_quant_zp)
        # Set output tensor
        code += self.CodeGen_Set_output_tensor(output_tensor, self.allocated_tensor[output_tensor_id].start_address, output_quant_zp)
        # Set scale
        code += f"SET_MULTIPLIER {multiplier}\n"
        code += f"SET_SHIFT {shift}\n"
        code += self.CodeGen_DMA_wait()
        code += "FULLY_CONNECTED\n"
        self.npu_code += code

    def batch_matmul_codegen(self, operator):
        print(f"BATCH_MATMUL: {operator['inputs']}, {operator['outputs']}")

    # TODO: Implement this function
    def conv_codegen(self, operator):
        # Conv's output tensor may depend on other input tensors
        input_tensor_ids = []
        input_tensors = []
        pad_config = None
        
        if len(operator['inputs']) < 4:
            # No need to consume other operators' output tensor (normally because the filter is 1x1)
            input_tensor_ids.append(operator['inputs'][0])
        else:
            pad_config = self.buffers[self.tensor[operator['inputs'][3]]['buffer']]['data']
            # Need to consume other operators' output tensor
            for i in range(4, len(operator['inputs'])):
                input_tensor_ids.append(operator['inputs'][i])
        filter_tensor_id = operator['inputs'][1]
        bias_tensor_id = operator['inputs'][2]
        output_tensor_id = operator['outputs'][0]

        for input_tensor_id in input_tensor_ids:
            input_tensors.append(self.tensor[input_tensor_id])
        filter_tensor = self.tensor[filter_tensor_id]
        bias_tensor = self.tensor[bias_tensor_id]
        output_tensor = self.tensor[output_tensor_id]

        stride_h = operator['builtin_options']['stride_h']
        pad_h = pad_config[0]
        pad_w = pad_config[1]

        tokens = output_tensor['name'].split('_split_')
        output_id = -1
        if len(tokens) == 2:
            if tokens[1].isnumeric():
                output_id = int(tokens[1])
        elif len(tokens) > 2:
                raise(f"Invalid output tensor name: {output_tensor['name']}")
        
        # Initial the input tensor's range required
        split_size = self.tensor[operator['inputs'][0]]['shape'][1]
        input_range_required = defaultdict(list)
        input_id_list = []
        for input_tensor_id in input_tensor_ids:
            input_tensor = self.tensor[input_tensor_id]
            tokens = input_tensor['name'].split('_split_')
            if tokens[1].isnumeric():
                input_id = int(tokens[1])
                # Put the input tensor's range reversed, for the following comarison to find the real boundry
                input_range_required[input_id] = [(input_id + 1) * split_size - 1, input_id * split_size]
                input_id_list.append(input_id)
        
        # Compute the start of the boundry of the input tensor
        for out_h in range(split_size * output_id, split_size * (output_id + 1)):
            in_h_origin = out_h * stride_h - pad_h
            for kernel_h in range(filter_tensor['shape'][1]):
                in_h = in_h_origin + kernel_h
                # OverBound case, no need to consider over the maximum boundry, since it has considered in the model_gen
                source_sid = in_h // split_size
                if in_h < 0 or source_sid not in input_range_required:
                    continue
                else:
                    # print(f"out_h: {out_h}, in_h: {in_h}, source_sid: {source_sid}")
                    input_range_required[source_sid][0] = min(input_range_required[source_sid][0], in_h)
                    input_range_required[source_sid][1] = max(input_range_required[source_sid][1], in_h)

        # TODO: Compose the IFM (future work) 
        for input_tensor_id, source_sid in input_tensor_ids, input_id_list:
            pass
            # input_tensor = self.tensor[input_tensor_id]
            # input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
            # input_quant_zp = input_tensor['quantization']['zero_point'][0]

            # code = self.CodeGen_Set_input_tensor(input_tensor, self.allocated_tensor[input_tensor_id].start_address, input_quant_zp)
            # self.npu_code += code

        # input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
        # weight_quant_scale = np.int32(filter_tensor['quantization']['scale'][0]).view('float32')
        # output_qunat_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        # scale = input_quant_scale * weight_quant_scale / output_qunat_scale
        # multiplier, shift = self.QuantizeMultiplier(scale)

        # input_quant_zp = input_tensor['quantization']['zero_point'][0]
        # filter_quant_zp = filter_tensor['quantization']['zero_point'][0]
        # output_quant_zp = output_tensor['quantization']['zero_point'][0]

        # code = ""
        # # Start to load weights
        # code += self.CodeGen_DMA_start(self.allocated_tensor[filter_tensor_id].start_address, self.Compute_tensor_size(filter_tensor))
        # # Set input tensor
        # code += self.CodeGen_Set_input_tensor(input_tensor, self.allocated_tensor[input_tensor_id].start_address, input_quant_zp)
        # # Set weight tensor
        # code += self.CodeGen_Set_weight_tensor(filter_tensor, self.allocated_tensor[filter_tensor_id].start_address, filter_quant_zp)
        # # Set output tensor
        # code += self.CodeGen_Set_output_tensor(output_tensor, self.allocated_tensor[output_tensor_id].start_address, output_quant_zp)
        # # Set scale
        # code += f"SET_MULTIPLIER {multiplier}\n"
        # code += f"SET_SHIFT {shift}\n"
        # code += self.CodeGen_DMA_wait()
        # code += "CONV\n"
        # self.npu_code += code

    def mul_codegen(self, operator):
        print(f"MUL: {operator['inputs']}, {operator['outputs']}")

    def add_codegen(self, operator):
        print(f"ADD: {operator['inputs']}, {operator['outputs']}")

    def softmax_codegen(self, operator):
        print(f"SOFTMAX: {operator['inputs']}, {operator['outputs']}")

    def concat_codegen(self, operator):
        print(f"CONCAT: {operator['inputs']}, {operator['outputs']}")

    def split_codegen(self, operator):
        print(f"SPLIT: {operator['inputs']}, {operator['outputs']}")

    def op_codegen(self, operator):
        opcode_index = operator['opcode_index']
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'FULLY_CONNECTED':
            self.fully_connected_codegen(operator)
        elif opcode_type == 'CONV_2D':
            self.conv_codegen(operator)
        elif opcode_type == 'BATCH_MATMUL':
            self.batch_matmul_codegen(operator)
        elif opcode_type == 'MUL':
            self.mul_codegen(operator)
        elif opcode_type == 'ADD':
            self.add_codegen(operator)
        elif opcode_type == 'SOFTMAX':
            self.softmax_codegen(operator)
        elif opcode_type == 'CONCATENATION':
            self.concat_codegen(operator)
        elif opcode_type == 'SPLIT':
            self.split_codegen(operator)
        else:
            print(f"[CODE_GEN] Unknown operator: {opcode_type}")