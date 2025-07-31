# Roofline model

from MyGraph import Graph
from OpClassify import Op_Classify
import struct
import math

op_classify = Op_Classify()
mac_ops = op_classify.mac_ops
need_weights_ops = op_classify.need_weights_ops
unary_elementwise_ops = op_classify.unary_ops
binary_elementwise_ops = op_classify.binary_ops
elementwise_ops = op_classify.elementwise_ops

class RooflineModel:
    def __init__(self, model: Graph, tensor_info, total_cycles):
        self.graph = model
        self.buffers = model.buffers
        self.tensors = model.tensors
        self.ops = model.ops
        self.opcodes = model.opcodes
        # Allocated tensors
        self.tensor_info = tensor_info
        self.total_cycles = total_cycles
        # Clock rate
        self.clock_rate = 3e8  # 300 MHz (same to DRAM memory clock rate)
        # Model's point in the roofline model
        self.total_operations = 0
        self.total_dram_access = 0
        self.operational_intensity = 0
        self.giga_operations_per_second = 0
        # Roofline
        self.peak_operational_intensity = 0
        self.peak_giga_operations_per_second = 0

    # Similar to the simulation, but records different information
    def roofline_model_build(self):
        for op in self.graph.ordered_ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            # MAC main ops
            if opcode_type in mac_ops:
                operations, dram_accesses = self.mac_op_operations_and_dram_accesses(op, opcode_type)
                self.total_operations += operations
                self.total_dram_access += dram_accesses
            # Elementwise ops
            elif opcode_type in elementwise_ops:
                operations, dram_accesses = self.elementwise_op_operations_and_dram_accesses(op, opcode_type)
                self.total_operations += operations
                self.total_dram_access += dram_accesses
        # Model point calculation
        self.operational_intensity = math.log(self.total_operations / self.total_dram_access, 10)
        self.giga_operations_per_second = math.log(self.total_operations / 1e9 / (self.total_cycles / self.clock_rate), 10)
        # Roofline model's memory bound and compute bound's cross points
        # MAC(128x128 PEs) + EXP + RECIPROCAL + ADD + SUB + MUL (Vector len 128 elements)
        peak_operations = 128 * 128 * 2 + 128 * (6 + 13 + 1 + 1 + 1)
        peak_operations_per_second = peak_operations * self.clock_rate
        self.peak_giga_operations_per_second = math.log(peak_operations_per_second / 1e9, 10)
        # Peak operational intensity is the cross point of memory bound and compute bound (in x-axis)
        self.peak_operational_intensity = math.log(peak_operations_per_second / (19.2 * 1e9), 10)
        # Roofline's start point (0, 1.28) -> log10(1) = 0, log10(19.2) = 1.28

    def mac_op_operations_and_dram_accesses(self, op, op_type):
        tensors = self.tensors
        info = op.info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        ofm = self.tensors[outputs[0]]
        ofm_shape = ofm.get("shape")
        if ofm.get("type") == "INT8":
            ifm_elem_size = 8
            ofm_elem_size = 8
        else:
            ifm_elem_size = 32
            ofm_elem_size = 32
        
        weights_storage_size = 0
        B = 1
        if op_type == "CONV_2D":
            B = ofm_shape[0]
            OH = ofm_shape[1]
            OW = ofm_shape[2]
            filter = tensors[inputs[1]]
            filter_shape = filter.get("shape")
            FH = filter_shape[1]
            FW = filter_shape[2]
            IC = filter_shape[3]
            OC = filter_shape[0]
            stride = info["builtin_options"]["stride_h"]
        elif op_type == "DEPTHWISE_CONV_2D":
            # Special case in mean's convert
            if(len(ofm_shape) == 3):
                ofm_shape.append(1)
            B = ofm_shape[0]
            OH = ofm_shape[1]
            OW = ofm_shape[2]
            filter = tensors[inputs[1]]
            filter_shape = filter.get("shape")
            FH = filter_shape[1]
            FW = filter_shape[2]
            IC = 1
            OC = 1
            stride = info["builtin_options"]["stride_h"]
        elif op_type == "FULLY_CONNECTED":
            OH = ofm_shape[-2]
            OW = 1
            FH = 1
            FW = 1
            weight = tensors[inputs[1]]
            weight_shape = weight.get("shape")
            IC = weight_shape[1]
            OC = weight_shape[0]
            stride = 1
        elif op_type == "MAX_POOL_2D":
            if (len(ofm_shape) == 3):
                B = ofm_shape[0]
                OH = ofm_shape[1]
                OW = ofm_shape[2]
            else:
                B = ofm_shape[0]
                OH = ofm_shape[1] * ofm_shape[2]
                OW = ofm_shape[3]
            FH = info["builtin_options"]["filter_height"]
            FW = info["builtin_options"]["filter_width"]
            IC = 1
            OC = 1
            stride = info["builtin_options"]["stride_h"]
        IH = (OH - 1) * stride + FH
        IW = (OW - 1) * stride + FW

        ############## Calculate the operations ##############
        output_elements = B * OH * OW * OC
        one_element_compute_needed = FH * FW * IC
        if op_type == "MAX_POOL_2D":
            # Perform one max operation
            operations = output_elements * one_element_compute_needed
        else:
            # MAC operations contain mul and add
            operations = output_elements * one_element_compute_needed * 2

        ############## Calculate the DRAM accesses(bytes) ##############
        dram_accesses = 0
        # Compute ifm storage size
        ifm_storge_size = B * IH * IW * IC * (ifm_elem_size / 8)
        # Compute ofm storage size
        ofm_storge_size = B * OH * OW * OC * (ofm_elem_size / 8)
        # Compute weight storage size
        if op_type in need_weights_ops:
            weights_storage_size = OC * FH * FW * IC * (ifm_elem_size / 8)

        # Check input tensor
        if op_type == "FULLY_CONNECTED" and len(inputs) > 3:
            dram_access += ifm_storge_size
        elif op_type == "CONV_2D" and len(inputs) > 5:
            dram_accesses += ifm_storge_size
        elif op_type == "DEPTHWISE_CONV_2D" and len(inputs) > 5:
            dram_accesses += ifm_storge_size
        else:
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                if tensor_metadata.cid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += ifm_storge_size
                    break
        # Check weight tensor
        if op_type in need_weights_ops:
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                if tensor_metadata.cid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += weights_storage_size
                    break
        # Check output tensor
        for tensor_metadata in self.tensor_info[outputs[0]].tensors:
            if tensor_metadata.pid == op.opid:
                if tensor_metadata.in_DRAM:
                    dram_accesses += ofm_storge_size
                break

        return operations, dram_accesses

    def elementwise_op_operations_and_dram_accesses(self, op, op_type):
        tensors = self.tensors
        info = op.info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        if op_type in unary_elementwise_ops:
            ofm = tensors[outputs[0]]
            ofm_shape = ofm.get("shape")

            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############## Calculate the operations ##############
            if op_type == "EXP":
                operation_count = 1
            elif op_type == "RECIPROCAL":
                operation_count = 1
            elif op_type == "RSQRT":
                # We use the LUT to get the rsqrt result, so there is no operation perform
                operation_count = 1
            elif op_type == "POW":
                exp_tensor = tensors[inputs[1]]
                exp_buffer_data = bytes(self.buffers[exp_tensor['buffer']]['data'])
                # Parse the exp_buffer_data to get the exponent, '<': little-endian, 'f': float
                Exponent = int(struct.unpack('<f', exp_buffer_data)[0])
                operation_count = Exponent - 1
            # Logistic(x) = 1 / (1 + exp(-x)): sub + exp + reciprocal = 20 (add -> scale folding)
            elif op_type == "TANH":
                # Tanh(x) = 2 * logistic(2 * x) - 1: logistic (2 * mul + add -> scale folding)
                operation_count = 20
            elif op_type == "GELU":
                # Gelu(x) = x * logistic(1.702 * x): mul + logistic (mul -> scale folding)
                operation_count = 21
            if op_type == "QUANTIZE":
                # mul + add
                operation_count = 2
            elif op_type == "DEQUANTIZE":
                operation_count = 2
            operations = ofm_elems * operation_count

            ############## Calculate the DRAM accesses(bytes) ##############
            dram_accesses = 0
            # Check input tensor
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                if tensor_metadata.cid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += self.tensor_info[inputs[0]].size
                    break
            # Check output tensor
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += self.tensor_info[outputs[0]].size
                    break
            
            # If no operations, set dram_accesses to 0
            if operations == 0:
                dram_accesses = 0
            
            return operations, dram_accesses
        elif op_type in binary_elementwise_ops:
            ofm = tensors[outputs[0]]
            ofm_shape = ofm.get("shape")

            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############## Calculate the operations ##############
            if op_type == "ADD":
                operation_count = 1
            elif op_type == "SUB":
                operation_count = 1
            elif op_type == "MUL":
                operation_count = 1
            elif op_type == "SQUARED_DIFFERENCE":
                # SquaredDifference(x, y) = (x - y) * (x - y): sub + mul
                operation_count = 2
            operations = ofm_elems * operation_count

            ############## Calculate the DRAM accesses(bytes) ##############
            dram_accesses = 0
            # Check input tensor1
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                if tensor_metadata.cid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += self.tensor_info[inputs[0]].size
                    break
            # Check input tensor2
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                if tensor_metadata.cid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += self.tensor_info[inputs[1]].size
                    break
            # Check output tensor
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == op.opid:
                    if tensor_metadata.in_DRAM:
                        dram_accesses += self.tensor_info[outputs[0]].size
                    break
            
            return operations, dram_accesses