# This simulator is aligned with our NPU design
from MyGraph import Graph
from Architecture_feature import ArchitectureFeatures
from OpClassify import Op_Classify
import math
import struct
import copy

op_classify = Op_Classify()
data_layout_ops = op_classify.data_layout_ops
mac_ops = op_classify.mac_ops
need_weights_ops = op_classify.need_weights_ops
need_bias_ops = op_classify.need_bias_ops
unary_elementwise_ops = op_classify.unary_ops
binary_elementwise_ops = op_classify.binary_ops
elementwise_ops = op_classify.elementwise_ops

class simulator:
    def __init__(self, model: Graph, tensors_info):
        self.model = model
        self.buffers = model.buffers
        self.tensors = model.tensors
        self.ops = model.ops
        self.opcodes = model.opcodes
        self.tensor_info = tensors_info
        
    # Based on the elementwise engine's architecture, one tile will perform VECTOR_LEN elements' operations
    def estimate_elementwise_op_cycles(self, opid: int, op_type: str) -> int:
        tensors = self.tensors
        info = self.ops[opid].info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        if op_type in unary_elementwise_ops:
            ifm = tensors[inputs[0]]
            ofm = tensors[outputs[0]]
            ofm_shape = ofm.get("shape")

            if ifm.get("type") == "INT8":
                ifm_elem_size = 8
            else:
                ifm_elem_size = 32
            if ofm.get("type") == "INT8":
                ofm_elem_size = 8
            else:
                ofm_elem_size = 32

            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############### DRAM access cycles ############### 
            # Our elementwise engine adopts a SIMD vector design
            # IFM's data transfer
            ifm_storage_size = self.tensor_info[inputs[0]].size * (ifm_elem_size / 8)
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        ifm_storage_size = 0
                    break
            # OFM's data transfer
            ofm_storage_size = self.tensor_info[outputs[0]].size * (ofm_elem_size / 8)
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == opid:
                    if not tensor_metadata.in_DRAM:
                        ofm_storage_size = 0
                    break
            total_dram_accesses = ifm_storage_size + ofm_storage_size
            # DRAM transfer bytes per cycle
            dram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
            dram_transfer_cycles = math.ceil(total_dram_accesses / float(dram_bandwidth))

            ############### Compute cycles & SRAM access cycles ###############
            ifm_storage_size = self.tensor_info[inputs[0]].size * (ifm_elem_size / 8)
            ofm_storage_size = self.tensor_info[outputs[0]].size * (ofm_elem_size / 8)
            total_sram_accesses = ifm_storage_size + ofm_storage_size
            # SRAM transfer per cycle
            sram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Sram_clock_scale
            sram_transfer_cycles = math.ceil(total_sram_accesses / float(sram_bandwidth))

            # Element-wise operation will speedup by vectorization
            if op_type == "EXP":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["EXP"]
            elif op_type == "RECIPROCAL":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["RECIPROCAL"]
            elif op_type == "RSQRT":
                # There have quick rsqrt method, but it looks like can't be used in our design
                # Above reference: https://zh.wikipedia.org/zh-tw/%E5%B9%B3%E6%96%B9%E6%A0%B9%E5%80%92%E6%95%B0%E9%80%9F%E7%AE%97%E6%B3%95
                # So, we use the LUT to get the rsqrt result
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LUT"]
            elif op_type == "POW":
                # Pow(x, y) = x^y
                # Extract the exponent from the second input tensor, tflite store the data in little-endian format
                exp_tensor = tensors[inputs[1]]
                exp_buffer_data = bytes(self.buffers[exp_tensor['buffer']]['data'])
                # Parse the exp_buffer_data to get the exponent, '<': little-endian, 'f': float
                Exponent = int(struct.unpack('<f', exp_buffer_data)[0])
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"] * (Exponent - 1)
            elif op_type == "TANH":
                # Tanh(x): 2 * Logistic(2 * x) - 1
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"] + ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"])
            elif op_type == "GELU":
                # Gelu(x) = x * logistic(1.702 * x)
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"] + ArchitectureFeatures.output_cycles_per_elem["MUL"])
            
            if op_type == "QUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            elif op_type == "DEQUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            
            op_cycles = math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem
        elif op_type in binary_elementwise_ops:
            ifm1 = tensors[inputs[0]]
            ifm2 = tensors[inputs[1]]
            ofm = tensors[outputs[0]]
            ofm_shape = ofm.get("shape")

            if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
                ifm1_elem_size = 8
                ifm2_elem_size = 8
                ofm_elem_size = 8
            else:
                ifm1_elem_size = 32
                ifm2_elem_size = 32
                ofm_elem_size = 32

            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############### DRAM access cycles ############### 
            # Our elementwise engine adopts a SIMD vector design
            # IFM1's data transfer
            ifm1_storage_size = self.tensor_info[inputs[0]].size * (ifm1_elem_size / 8)
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        ifm1_storage_size = 0
                    break
            # IFM2's data transfer
            ifm2_storage_size = self.tensor_info[inputs[1]].size * (ifm2_elem_size / 8)
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        ifm2_storage_size = 0
                    break
            # OFM's data transfer
            ofm_storage_size = self.tensor_info[outputs[0]].size * (ofm_elem_size / 8)
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == opid:
                    if not tensor_metadata.in_DRAM:
                        ofm_storage_size = 0
                    break
            total_dram_accesses = ifm1_storage_size + ifm2_storage_size + ofm_storage_size
            # DRAM transfer bytes per cycle
            dram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
            dram_transfer_cycles = math.ceil(total_dram_accesses / float(dram_bandwidth))
            
            ############### Compute cycles & SRAM access cycles ###############
            ifm1_storage_size = self.tensor_info[inputs[0]].size * (ifm1_elem_size / 8)
            ifm2_storage_size = self.tensor_info[inputs[1]].size * (ifm2_elem_size / 8)
            ofm_storage_size = self.tensor_info[outputs[0]].size * (ofm_elem_size / 8)
            total_sram_accesses = ifm1_storage_size + ifm2_storage_size + ofm_storage_size
            # SRAM transfer per cycle
            sram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Sram_clock_scale
            sram_transfer_cycles = math.ceil(total_sram_accesses / float(sram_bandwidth))

            # Element-wise operation will speedup by vectorization
            if op_type == "ADD":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
            elif op_type == "SUB":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
            elif op_type == "MUL":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
            elif op_type == "SQUARED_DIFFERENCE":
                # SquaredDifference(x, y) = (x - y)(x - y)
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"] + ArchitectureFeatures.output_cycles_per_elem["MUL"]
            
            op_cycles = math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem

        dma_transfer_cycles = dram_transfer_cycles + sram_transfer_cycles
        total_cycles = op_cycles + dma_transfer_cycles
        return dma_transfer_cycles, op_cycles, total_cycles
    
    def estimate_mac_op_cycles(self, opid: int, op_type: str) -> int:
        tensors = self.tensors
        info = self.ops[opid].info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        ofm = tensors[outputs[0]]
        ofm_shape = copy.deepcopy(ofm.get("shape"))
        if ofm.get("type") == "INT8":
            ifm_elem_size = 8
            ofm_elem_size = 8
        else:
            ifm_elem_size = 32
            ofm_elem_size = 32
        
        weights_storage_size = 0
        bias_storage_size = 0
        # Now only support batch = 1
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
        # This estimation is not accurate
        elif op_type == "BATCH_MATMUL":
            OH = ofm_shape[1]
            OW = ofm_shape[2]
            FH = 1
            FW = 1
            IC = ofm_shape[3]
            OC = ofm_shape[3]
            stride = 1
        IH = (OH - 1) * stride + FH
        IW = (OW - 1) * stride + FW

        ############### DRAM access cycles ###############
        # Check whether the tensors need to move from DRAM to SRAM
        # Compute ifm storage size
        ifm_storge_size = B * IH * IW * IC * (ifm_elem_size / 8)
        # Compute ofm storage size
        ofm_storge_size = B * OH * OW * OC * (ofm_elem_size / 8)
        # Compute weight storage size
        if op_type == "CONV_2D" or op_type == "DEPTHWISE_CONV_2D":
            weights_storage_size = OC * FH * FW * IC * (ifm_elem_size / 8)
            # bias_storage_size = OH * OW * OC * (32 / 8)
        if op_type == "FULLY_CONNECTED":
            weights_storage_size = OC * FH * FW * IC * (ifm_elem_size / 8)
        # Check input tensor (if input need to perform concat, the tensor must fetch from DRAM)
        if op_type == "FULLY_CONNECTED" and len(inputs) > 3:
            pass
        elif op_type == "CONV_2D" and len(inputs) > 5:
            pass
        elif op_type == "DEPTHWISE_CONV_2D" and len(inputs) > 5:
            pass
        else:
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        ifm_storge_size = 0
                    break
        # Check weight tensor
        if op_type in need_weights_ops:
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        weights_storage_size = 0
                    break
        # Check bias tensor
        if op_type in need_bias_ops:
            for tensor_metadata in self.tensor_info[inputs[2]].tensors:
                if tensor_metadata.cid == opid:
                    if not tensor_metadata.in_DRAM:
                        bias_storage_size = 0
                    break
        # Check output tensor
        for tensor_metadata in self.tensor_info[outputs[0]].tensors:
            if tensor_metadata.pid == opid:
                if not tensor_metadata.in_DRAM:
                    ofm_storge_size = 0
                break
        # Compute the dram access cycles
        total_dram_accesses = ifm_storge_size + weights_storage_size + bias_storage_size + ofm_storge_size
        dram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
        dram_transfer_cycles = math.ceil(total_dram_accesses / float(dram_bandwidth))

        ############### Compute cycles & SRAM access cycles ###############
        # SRAM transfer bytes per cycle
        sram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Sram_clock_scale

        # Our MAC engine compute a 2D windows(in FC: 1 x n(dims), in CONV: filter) until all its OC computed (reuse windows' data)
        # Compute the number of OC that MAC PEs can compute in one cycle
        total_PEs = ArchitectureFeatures.MAC_height * ArchitectureFeatures.MAC_width
        one_element_compute_needed = FH * FW * IC
        if one_element_compute_needed <= total_PEs:
            oc_finish_per_cycle = math.floor(total_PEs / one_element_compute_needed)
        else:
            raise ValueError("Not support one element compute needed > total MAC PEs")
        # Every time deal with new ocs, the weight and init_inputs need to be reloaded
        launch_new_oc_times = math.ceil(OC / float(oc_finish_per_cycle))
        
        total_windows = B * OH * OW

        weights_needed_size = FH * FW * IC * (ifm_elem_size / 8)
        load_weights_cycles  = math.ceil(weights_needed_size / float(sram_bandwidth)) * launch_new_oc_times
        # Needed data in input tensor is not sequential at H dimension
        load_init_inputs_cycles = FH * math.ceil(FW * IC * (ifm_elem_size / 8) / float(sram_bandwidth)) * launch_new_oc_times
        store_outputs_cycles = math.ceil(total_windows * OC / float(sram_bandwidth))

        op_cycles = total_windows * launch_new_oc_times * ArchitectureFeatures.output_cycles_per_elem["MAC"]

        dma_transfer_cycles = dram_transfer_cycles + load_weights_cycles + load_init_inputs_cycles + store_outputs_cycles
        total_cycles = op_cycles + dma_transfer_cycles
        return dma_transfer_cycles, op_cycles, total_cycles

    # Estimate the number of cycles for a given operation
    def estimate_op_cycles(self, opid: int) -> int:
        op = self.ops[opid]
        opcode_index = op.info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type in elementwise_ops:
            dma_cycles, op_cycles, total_cycles = self.estimate_elementwise_op_cycles(opid, opcode_type)
            op.estimated_DMA_cycles = dma_cycles
            op.estimated_op_cycles = op_cycles
            op.estimated_total_cycles = total_cycles
        elif opcode_type in mac_ops:
            dma_cycles, op_cycles, total_cycles = self.estimate_mac_op_cycles(opid, opcode_type)
            op.estimated_DMA_cycles = dma_cycles
            op.estimated_op_cycles = op_cycles
            op.estimated_total_cycles = total_cycles
        elif opcode_type in data_layout_ops:
            # NPU won't do these ops, so for now set the cycles to 0
            dma_cycles, op_cycles, total_cycles = 0, 0, 0
            op.estimated_DMA_cycles = dma_cycles
            op.estimated_op_cycles = op_cycles
            op.estimated_total_cycles = total_cycles
        else:
            dma_cycles = 0
            op_cycles = 0
            total_cycles = 0
            print(f"Not yet supported {opcode_type}'s cycle estimation, its opcode_index is {opcode_index}")
        return dma_cycles, op_cycles, total_cycles

    def estimate_model(self, pipeline: bool) -> int:
        total_dma_cycles = 0
        total_op_cycles = 0
        total_cycles = 0

        mac_idle_cycles = 0
        elem_wise_idle_cycles = 0

        for op in self.model.ordered_ops:
            opid = op.opid
            # w/wo pipeline schedule, the estimated total cycles will be different, since we consider the memory footprint between DRAM and SRAM
            dma_cycles, op_cycles, op_total_cycles = self.estimate_op_cycles(opid)
            # For later pipeline schedule, op_total_cycles will be used to determine the overlap range
            if not pipeline:
                self.ops[opid].non_overlap_cycles = op_total_cycles
            total_dma_cycles += dma_cycles
            total_op_cycles += op_cycles
            total_cycles += op_total_cycles

            if self.ops[opid].is_mac_main_op == True:
                elem_wise_idle_cycles += op_cycles
            elif self.ops[opid].is_elem_wise_main_op == True:
                mac_idle_cycles += op_cycles
            
        if pipeline:
            cascade_matched_ops = self.model.cascade_matched_ops
            cascade_total_op = 0
            cascade_save_cycles = 0
            # If the op has multiple consumers, we need to store the output tensor back to SRAM (for other consumers)
            have_multi_consumer = True if len(self.ops[opid].children) > 1 else False
            # For now, assume the two op parallel execution will take the longer one
            for cascade_matched_op in cascade_matched_ops:
                cascade_total_op += len(cascade_matched_op)
                op1 = self.ops[cascade_matched_op[0]]
                op2_estimated_op_cycles = 0
                op2_estimated_total_cycles = 0
                for op_id in range(1, len(cascade_matched_op)):
                    op2 = self.ops[cascade_matched_op[op_id]]
                    op2_estimated_op_cycles += op2.estimated_op_cycles
                    op2_estimated_total_cycles += op2.estimated_total_cycles
                if op1.estimated_total_cycles > op2_estimated_total_cycles:
                    total_op_cycles -= op2_estimated_op_cycles
                    total_cycles -= op2_estimated_op_cycles
                    cascade_save_cycles += op2_estimated_op_cycles
                else:
                    total_op_cycles -= op1.estimated_op_cycles
                    total_cycles -= op1.estimated_op_cycles
                    cascade_save_cycles += op1.estimated_op_cycles
            matched_ops = self.model.matched_ops
            match_total_op = 0
            match_save_cycles = 0
            # For now, assume the two op parallel execution will take the longer one
            for matched_op in matched_ops:
                match_total_op += len(matched_op)
                op1 = self.ops[matched_op[0]]
                op2_estimated_op_cycles = 0
                op2_estimated_total_cycles = 0
                for op_id in range(1, len(matched_op)):
                    op2 = self.ops[matched_op[op_id]]
                    op2_estimated_op_cycles += op2.estimated_op_cycles
                    op2_estimated_total_cycles += op2.estimated_total_cycles
                if op1.estimated_total_cycles > op2_estimated_total_cycles:
                    total_op_cycles -= op2_estimated_op_cycles
                    total_cycles -= op2_estimated_op_cycles
                    match_save_cycles += op2_estimated_op_cycles
                else:
                    total_op_cycles -= op1.estimated_op_cycles
                    total_cycles -= op1.estimated_op_cycles
                    match_save_cycles += op1.estimated_op_cycles
            # print(f"cascade_total_op: {cascade_total_op}, cascade_save_cycles: {cascade_save_cycles}")
            # print(f"match_total_op: {match_total_op}, match_save_cycles: {match_save_cycles}")
        return total_dma_cycles, total_op_cycles, total_cycles

    def print_performance(self):
        for order, op in enumerate(self.model.ordered_ops):
            opcode_index = op.info.get("opcode_index")
            opid = op.opid
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            dma_cycles = op.estimated_DMA_cycles
            op_cycles = op.estimated_op_cycles
            op_total_cycles = op.estimated_total_cycles
            print("*" * 50)
            print(f"order: {order}, opid: {opid}, opcode_type: {opcode_type}, DMA cycles: {dma_cycles}, \
                    OP cycles: {op_cycles}, Total cycles: {op_total_cycles}")
            print(f"op info: {op.info}")