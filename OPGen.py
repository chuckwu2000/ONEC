import numpy as np
import math
class OPGen:
    def __init__(self, model, npu_code):
        self.model = model
        self.opcodes = model['operator_codes']
        self.tensor = model['subgraphs'][0]['tensors']
        self.buffers = model['buffers']
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
        input_tensor = self.tensor[operator['inputs'][0]]
        weight_tensor = self.tensor[operator['inputs'][1]]
        output_tensor = self.tensor[operator['outputs'][0]]

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
        code += self.CodeGen_DMA_start(80000, self.Compute_tensor_size(weight_tensor))
        # Set input tensor
        code += self.CodeGen_Set_input_tensor(input_tensor, 0, input_quant_zp)
        # Set weight tensor
        code += self.CodeGen_Set_weight_tensor(weight_tensor, 80000, weight_quant_zp)
        # Set output tensor
        code += self.CodeGen_Set_output_tensor(output_tensor, 160000, output_quant_zp)
        # Set scale
        code += f"SET_MULTIPLIER {multiplier}\n"
        code += f"SET_SHIFT {shift}\n"
        code += self.CodeGen_DMA_wait()
        code += "FULLY_CONNECTED\n"
        self.npu_code += code

    def batch_matmul_codegen(self, operator):
        print(f"BATCH_MATMUL: {operator['inputs']}, {operator['outputs']}")

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