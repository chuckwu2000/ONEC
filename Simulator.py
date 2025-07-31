# This simulator is aligned with our NPU design

from MyGraph import Graph
from Architecture_feature import ArchitectureFeatures
from OpClassify import Op_Classify
import math
import struct
import subprocess
import os
import re

op_classify = Op_Classify()
data_layout_ops = op_classify.data_layout_ops
mac_ops = op_classify.mac_ops
need_weights_ops = op_classify.need_weights_ops
need_bias_ops = op_classify.need_bias_ops
unary_elementwise_ops = op_classify.unary_ops
binary_elementwise_ops = op_classify.binary_ops
elementwise_ops = op_classify.elementwise_ops
use_lut_ops = op_classify.use_lut_ops

class simulator:
    def __init__(self, model: Graph, tensors_info):
        self.model = model
        self.buffers = model.buffers
        self.tensors = model.tensors
        self.ops = model.ops
        self.opcodes = model.opcodes
        self.tensor_info = tensors_info
        self.total_energy = 0
        self.dram_energy = 0
        self.sram_energy = 0
        self.core_energy = 0
        
    # Based on the elementwise engine's architecture, one tile will perform VECTOR_LEN elements' operations
    def estimate_elementwise_op_cycles(self, opid: int, op_type: str) -> int:
        tensors = self.tensors
        info = self.ops[opid].info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        initial_dram_reads = 0
        final_dram_writes = 0
        dram_transfer_size = 0
        if op_type in unary_elementwise_ops:
            ifm = tensors[inputs[0]]
            ofm = tensors[outputs[0]]
            ifm_shape = ifm.get("shape")
            ofm_shape = ofm.get("shape")

            if ifm.get("type") == "INT8":
                ifm_elem_size = 8
            else:
                ifm_elem_size = 32
            if ofm.get("type") == "INT8":
                ofm_elem_size = 8
            else:
                ofm_elem_size = 32

            # ifm's elements
            ifm_elems = 1
            for dim in ifm_shape:
                ifm_elems *= dim
            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############### DRAM access cycles ############### 
            # Our elementwise engine adopts a SIMD vector design
            # IFM's data transfer
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += min(ifm_elems, ArchitectureFeatures.VECTOR_LEN) * (ifm_elem_size / 8)
                        dram_transfer_size += self.tensor_info[inputs[0]].size
                    break
            if op_type in use_lut_ops and ifm_elem_size == 8:
                for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                    # Find the corresponding tensor metadata
                    if tensor_metadata.cid == opid:
                        if tensor_metadata.in_DRAM:
                            initial_dram_reads += 256 * (ifm_elem_size / 8)  # LUT size is 256
                            dram_transfer_size += 256
            # OFM's data transfer
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == opid:
                    if tensor_metadata.in_DRAM:
                        final_dram_writes += min(ofm_elems, ArchitectureFeatures.VECTOR_LEN) * (ofm_elem_size / 8)
                        dram_transfer_size += self.tensor_info[outputs[0]].size
                    break

            ############### Compute cycles ###############
            # Element-wise operation will speedup by vectorization
            if op_type == "EXP":
                # To reduce the elementwise engine size, we use the LUT to get the exp result
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LUT"]
                operation_count = 1
            elif op_type == "RECIPROCAL":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LUT"]
                operation_count = 1
            elif op_type == "RSQRT":
                # There have quick rsqrt method, but it looks like can't be used in our design
                # Above reference: https://zh.wikipedia.org/zh-tw/%E5%B9%B3%E6%96%B9%E6%A0%B9%E5%80%92%E6%95%B0%E9%80%9F%E7%AE%97%E6%B3%95
                # So, we use the LUT to get the rsqrt result
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LUT"]
                operation_count = 1
            elif op_type == "POW":
                # Pow(x, y) = x^y
                # Extract the exponent from the second input tensor, tflite store the data in little-endian format
                exp_tensor = tensors[inputs[1]]
                exp_buffer_data = bytes(self.buffers[exp_tensor['buffer']]['data'])
                # Parse the exp_buffer_data to get the exponent, '<': little-endian, 'f': float
                Exponent = int(struct.unpack('<f', exp_buffer_data)[0])
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"] * (Exponent - 1)
                operation_count = Exponent - 1
            elif op_type == "TANH":
                # Tanh(x): 2 * Logistic(2 * x) - 1
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"])
                operation_count = 20
            elif op_type == "GELU":
                # Gelu(x) = x * logistic(1.702 * x)
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"] + ArchitectureFeatures.output_cycles_per_elem["MUL"])
                operation_count = 21
            
            if op_type == "QUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
                operation_count = 2
            elif op_type == "DEQUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
                operation_count = 2
            op_cycles = math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem

            ############### Compute memory access energy ###############
            # DRAM access energy (will get after ramulator2_simulate phase)
            # SRAM access energy
            self.total_energy += ifm_elems * ifm_elem_size * ArchitectureFeatures.sram_cost
            self.total_energy += ofm_elems * ofm_elem_size * ArchitectureFeatures.sram_cost
            self.sram_energy += ifm_elems * ifm_elem_size * ArchitectureFeatures.sram_cost
            self.sram_energy += ofm_elems * ofm_elem_size * ArchitectureFeatures.sram_cost
            # Compute energy for elementwise operations
            self.total_energy += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * ArchitectureFeatures.vector_cost * operation_count
            self.core_energy += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * ArchitectureFeatures.vector_cost * operation_count
        elif op_type in binary_elementwise_ops:
            ifm1 = tensors[inputs[0]]
            ifm2 = tensors[inputs[1]]
            ofm = tensors[outputs[0]]
            ifm1_shape = ifm1.get("shape")
            ifm2_shape = ifm2.get("shape")
            ofm_shape = ofm.get("shape")

            if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
                ifm1_elem_size = 8
                ifm2_elem_size = 8
                ofm_elem_size = 8
            else:
                ifm1_elem_size = 32
                ifm2_elem_size = 32
                ofm_elem_size = 32

            # ifm1's elements
            ifm1_elems = 1
            for dim in ifm1_shape:
                ifm1_elems *= dim
            # ifm2's elements
            ifm2_elems = 1
            for dim in ifm2_shape:
                ifm2_elems *= dim
            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            ############### DRAM access cycles ############### 
            # Our elementwise engine adopts a SIMD vector design
            # IFM1's data transfer
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += min(ifm1_elems, ArchitectureFeatures.VECTOR_LEN) * (ifm1_elem_size / 8)
                        dram_transfer_size += self.tensor_info[inputs[0]].size
                    break
            # IFM2's data transfer
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += min(ifm2_elems, ArchitectureFeatures.VECTOR_LEN) * (ifm2_elem_size / 8)
                        dram_transfer_size += self.tensor_info[inputs[1]].size
                    break
            # OFM's data transfer
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == opid:
                    if tensor_metadata.in_DRAM:
                        final_dram_writes += min(ofm_elems, ArchitectureFeatures.VECTOR_LEN) * (ofm_elem_size / 8)
                        dram_transfer_size += self.tensor_info[outputs[0]].size
                    break
            
            ############### Compute cycles ###############
            # Element-wise operation will speedup by vectorization
            if op_type == "ADD":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
                operation_count = 1
            elif op_type == "SUB":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
                operation_count = 1
            elif op_type == "MUL":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
                operation_count = 1
            elif op_type == "SQUARED_DIFFERENCE":
                # SquaredDifference(x, y) = (x - y)(x - y)
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"] + ArchitectureFeatures.output_cycles_per_elem["MUL"]
                operation_count = 2
            op_cycles = math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem

            ############### Compute memory access energy ###############
            # DRAM access energy (will get after ramulator2_simulate phase)
            # SRAM access energy
            self.total_energy += ifm1_elems * ifm1_elem_size * ArchitectureFeatures.sram_cost
            self.total_energy += ifm2_elems * ifm2_elem_size * ArchitectureFeatures.sram_cost
            self.total_energy += ofm_elems * ofm_elem_size * ArchitectureFeatures.sram_cost
            self.sram_energy += ifm1_elems * ifm1_elem_size * ArchitectureFeatures.sram_cost
            self.sram_energy += ifm2_elems * ifm2_elem_size * ArchitectureFeatures.sram_cost
            self.sram_energy += ofm_elems * ofm_elem_size * ArchitectureFeatures.sram_cost
            # Compute energy for elementwise operations
            self.total_energy += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * ArchitectureFeatures.vector_cost * operation_count
            self.core_energy += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * ArchitectureFeatures.vector_cost * operation_count

        dram_transfer_size -= (initial_dram_reads + final_dram_writes)
        # First tile's read from DRAM & last tile's write to DRAM can't be overlapped by double buffer
        latency, latency_energy = self.ramulator2_simulate(initial_dram_reads + final_dram_writes)
        dram_transfer_cycles, dram_transfer_energy = self.ramulator2_simulate(dram_transfer_size)
        dma_transfer_cycles = max(0, dram_transfer_cycles - op_cycles) + latency
        total_cycles = op_cycles + dma_transfer_cycles
        # DRAM access energy
        self.total_energy += (latency_energy + dram_transfer_energy)
        self.dram_energy += (latency_energy + dram_transfer_energy)
        return dma_transfer_cycles, op_cycles, total_cycles
    
    def estimate_mac_op_cycles(self, opid: int, op_type: str) -> int:
        tensors = self.tensors
        info = self.ops[opid].info
        inputs = info.get("inputs")
        outputs = info.get("outputs")

        ofm = tensors[outputs[0]]
        ofm_shape = ofm.get("shape")
        if ofm.get("type") == "INT8":
            ifm_elem_size = 8
            ofm_elem_size = 8
        else:
            ifm_elem_size = 32
            ofm_elem_size = 32
        
        weights_storage_size = 0
        # Now only support batch = 1
        B = 1
        if op_type == "CONV_2D":
            # Special case in mean's convert
            if(len(ofm_shape) == 3):
                ofm_shape.insert(0, 1)
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
            B = ofm_shape[0]
            OH = ofm_shape[1]
            OW = ofm_shape[2]
            filter = tensors[inputs[1]]
            filter_shape = filter.get("shape")
            FH = filter_shape[1]
            FW = filter_shape[2]
            IC = filter_shape[3]
            OC = ofm_shape[3]
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

        # Our MAC engine reuse input data in a 2D window (in FC: 1 x n(dims), in CONV: filter) until all its OC computed
        total_PEs = ArchitectureFeatures.MAC_height * ArchitectureFeatures.MAC_width
        one_element_compute_needed = FH * FW * IC
        if one_element_compute_needed <= total_PEs:
            oc_finish_per_cycle = math.floor(total_PEs / one_element_compute_needed)
        else:
            oc_finish_per_cycle = total_PEs / one_element_compute_needed
            # raise ValueError("Not support one element compute needed > total MAC PEs")

        ############### DRAM access cycles ###############
        initial_dram_reads = 0
        final_dram_writes = 0
        dram_transfer_size = 0
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
        initial_input_dram_read = one_element_compute_needed * (ifm_elem_size / 8)
        if op_type == "FULLY_CONNECTED" and len(inputs) > 3:
            initial_dram_reads += initial_input_dram_read
            dram_transfer_size += ifm_storge_size
        elif op_type == "CONV_2D" and len(inputs) > 5:
            initial_dram_reads += initial_input_dram_read
            dram_transfer_size += ifm_storge_size
        elif op_type == "DEPTHWISE_CONV_2D" and len(inputs) > 5:
            initial_dram_reads += initial_input_dram_read
            dram_transfer_size += ifm_storge_size
        else:
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += initial_input_dram_read
                        dram_transfer_size += ifm_storge_size
                    break
        # Check weight tensor
        # Initial weight read need all weights
        if op_type in need_weights_ops:
            for tensor_metadata in self.tensor_info[inputs[1]].tensors:
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += weights_storage_size
                        dram_transfer_size += weights_storage_size
                    break
        # Check output tensor
        final_output_dram_write = math.ceil(oc_finish_per_cycle) * (ofm_elem_size / 8)
        for tensor_metadata in self.tensor_info[outputs[0]].tensors:
            if tensor_metadata.pid == opid:
                if tensor_metadata.in_DRAM:
                    final_dram_writes += final_output_dram_write
                    dram_transfer_size += ofm_storge_size
                break

        ############### Compute memory access energy ###############
        # DRAM access energy (will get after ramulator2_simulate phase)
        # SRAM access energy
        self.total_energy += ifm_storge_size * ifm_elem_size * ArchitectureFeatures.sram_cost
        self.total_energy += weights_storage_size * ifm_elem_size * ArchitectureFeatures.sram_cost
        self.total_energy += ofm_storge_size * ofm_elem_size * ArchitectureFeatures.sram_cost
        self.sram_energy += ifm_storge_size * ifm_elem_size * ArchitectureFeatures.sram_cost
        self.sram_energy += weights_storage_size * ifm_elem_size * ArchitectureFeatures.sram_cost
        self.sram_energy += ofm_storge_size * ofm_elem_size * ArchitectureFeatures.sram_cost

        ############### Compute cycles ###############
        # Every time deal with new ocs, the weight and init_inputs need to be reloaded
        compute_full_oc_times = math.ceil(OC / float(oc_finish_per_cycle))
        
        total_windows = B * OH * OW
        # Requantization is performed alongside MAC operations in the pipeline (compute an output likely takes 1 cycle)
        op_cycles = total_windows * compute_full_oc_times * ArchitectureFeatures.output_cycles_per_elem["MAC"]
        # Compute energy for MAC operations
        self.total_energy += op_cycles * ArchitectureFeatures.mac_cost
        self.core_energy += op_cycles * ArchitectureFeatures.mac_cost

        ############### Memory access cycles ###############
        # ----- We assume that cost of access SRAM is zero (by prefetching) -----
        # weights_needed_size = initial_weight_dram_read
        # load_weights_cycles  = math.ceil(weights_needed_size / float(sram_bandwidth)) * compute_full_oc_times * total_windows
        # # Needed data in input tensor is not sequential at H dimension
        # load_inputs_cycles = FH * math.ceil(FW * IC * (ifm_elem_size / 8) / float(sram_bandwidth)) * total_windows
        # # SRAM transfer bytes per cycle
        # sram_bandwidth = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Sram_clock_scale
        # store_outputs_cycles = math.ceil(total_windows * OC / float(sram_bandwidth))
        # op_cycles += load_weights_cycles + load_inputs_cycles + store_outputs_cycles

        dram_transfer_size -= (initial_dram_reads + final_dram_writes)
        # First tile's read from DRAM & last tile's write to DRAM can't be overlapped
        latency, latency_energy  = self.ramulator2_simulate(initial_dram_reads + final_dram_writes)
        dram_transfer_cycles, dram_transfer_energy = self.ramulator2_simulate(dram_transfer_size)
        dma_transfer_cycles = max(0, dram_transfer_cycles - op_cycles) + latency
        total_cycles = op_cycles + dma_transfer_cycles
        # DRAM access energy
        self.total_energy += (latency_energy + dram_transfer_energy)
        self.dram_energy += (latency_energy + dram_transfer_energy)
        return dma_transfer_cycles, op_cycles, total_cycles

    # Estimate the number of cycles for a given operation and energy consumption (only contain memory access now)
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

        ##### For Experiment #####
        # fused_op_cycles = 0
        # elementwise_op_cycles = 0
        # mac_op_cycles = 0

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

            ##### For Experiment #####
            # op = self.ops[opid]
            # opcode_index = op.info.get("opcode_index")
            # opcode_type = self.opcodes[opcode_index].get("builtin_code")
            # if opcode_type in mac_ops:
            #     mac_op_cycles += op_cycles
            # elif opcode_type in elementwise_ops:
            #     elementwise_op_cycles += op_cycles

        if pipeline:
            cascade_matched_ops = self.model.cascade_matched_ops
            cascade_total_op = 0
            cascade_save_cycles = 0
            # For now, assume the two op parallel execution will take the longer one
            for cascade_matched_op in cascade_matched_ops:
                cascade_total_op += len(cascade_matched_op)
                op1 = self.ops[cascade_matched_op[0]]
                op2_estimated_op_cycles = 0
                for op_id in range(1, len(cascade_matched_op)):
                    op2 = self.ops[cascade_matched_op[op_id]]
                    op2_estimated_op_cycles += op2.estimated_op_cycles
                if op1.estimated_op_cycles > op2_estimated_op_cycles:
                    total_op_cycles -= op2_estimated_op_cycles
                    total_cycles -= op2_estimated_op_cycles
                    cascade_save_cycles += op2_estimated_op_cycles
                    ##### For Experiment #####
                    # fused_op_cycles += op1.estimated_op_cycles
                    # mac_op_cycles -= op1.estimated_op_cycles
                    # elementwise_op_cycles -= op2_estimated_op_cycles
                else:
                    total_op_cycles -= op1.estimated_op_cycles
                    total_cycles -= op1.estimated_op_cycles
                    cascade_save_cycles += op1.estimated_op_cycles
                    ##### For Experiment #####
                    # fused_op_cycles += op2_estimated_op_cycles
                    # mac_op_cycles -= op1.estimated_op_cycles
                    # elementwise_op_cycles -= op2_estimated_op_cycles
            matched_ops = self.model.matched_ops
            match_total_op = 0
            match_save_cycles = 0
            # For now, assume the two op parallel execution will take the longer one
            for matched_op in matched_ops:
                match_total_op += len(matched_op)
                op1 = self.ops[matched_op[0]]
                op2_estimated_op_cycles = 0
                for op_id in range(1, len(matched_op)):
                    op2 = self.ops[matched_op[op_id]]
                    op2_estimated_op_cycles += op2.estimated_op_cycles
                if op1.estimated_op_cycles > op2_estimated_op_cycles:
                    total_op_cycles -= op2_estimated_op_cycles
                    total_cycles -= op2_estimated_op_cycles
                    match_save_cycles += op2_estimated_op_cycles
                    
                    ### For Experiment #####
                    # fused_op_cycles += op1.estimated_op_cycles
                    # opcode_index = op1.info.get("opcode_index")
                    # opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    # if opcode_type in mac_ops:
                    #     mac_op_cycles -= op1.estimated_op_cycles
                    #     elementwise_op_cycles -= op2_estimated_op_cycles
                    # elif opcode_type in elementwise_ops:
                    #     mac_op_cycles -= op2_estimated_op_cycles
                    #     elementwise_op_cycles -= op1.estimated_op_cycles
                else:
                    total_op_cycles -= op1.estimated_op_cycles
                    total_cycles -= op1.estimated_op_cycles
                    match_save_cycles += op1.estimated_op_cycles

                    ##### For Experiment #####
                    # fused_op_cycles += op2_estimated_op_cycles
                    # opcode_index = op1.info.get("opcode_index")
                    # opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    # if opcode_type in mac_ops:
                    #     mac_op_cycles -= op1.estimated_op_cycles
                    #     elementwise_op_cycles -= op2_estimated_op_cycles
                    # elif opcode_type in elementwise_ops:
                    #     mac_op_cycles -= op2_estimated_op_cycles
                    #     elementwise_op_cycles -= op1.estimated_op_cycles
            for elem_ops in self.model.seq_elem_ops:
                exec_time = [self.ops[opid].estimated_op_cycles for opid in elem_ops]
                max_exec_time = max(exec_time)
                # Fused op will take the longest time in the sequence elementwise operators 
                total_op_cycles -= sum(exec_time) - max_exec_time
                total_cycles -= sum(exec_time) - max_exec_time
        ##### For Experiment #####
        # print(f"mac ratio: {mac_op_cycles / total_op_cycles * 100:.2f}%, elementwise ratio: {elementwise_op_cycles / total_op_cycles * 100:.2f}%, fused ratio: {fused_op_cycles / total_op_cycles * 100:.2f}%")
        # print(f"!!!! elementwise_idle ratio: {1 - (elementwise_op_cycles + fused_op_cycles) / total_cycles:.2f}, mac_idle ratio: {1 - (mac_op_cycles + fused_op_cycles) / total_cycles:.2f} !!!!")
        # print(f"fused_op_ratio: {fused_op_cycles / total_cycles * 100:.2f}%, \
        #         mac_op_ratio: {mac_op_cycles / total_cycles * 100:.2f}%, \
        #         elementwise_op_ratio: {elementwise_op_cycles / total_cycles * 100:.2f}%, \
        #         dma_transfer_ratio: {total_dma_cycles / total_cycles * 100:.2f}%")
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

    def ramulator2_simulate(self, transfer_size: int):
        env = os.environ.copy()
        # Link the libramulator.so
        env["LD_LIBRARY_PATH"] = "extern/ramulator2"
        exe_path = "extern/ramulator2/resources/OEM_wrappers/ramulator2"
        cmd = [
            exe_path,
            ArchitectureFeatures.config_path,
            str(0),  # src_addr
            str(transfer_size),  # dram_transfer_size (bytes)
            str(0)   # read_write (0 for read, 1 for write), but origin ramulator2 not support write behavior
        ]
        result = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, env = env)
        if result.returncode != 0:
            print(f"Error running command: {' '.join(cmd)}")
        parts = result.stdout.strip().split()
        # num_reads = int(parts[0])
        # num_writes = int(parts[1])
        total_dram_access_ns = float(parts[2])
        total_dram_access_cycles = math.ceil(total_dram_access_ns / ArchitectureFeatures.core_period)

        # To get total_energy
        text = result.stdout
        match = re.search(r'total_energy:\s*([0-9.]+)', text)
        if match:
            total_energy = float(match.group(1))
        else:
            total_energy = 0.0
        return total_dram_access_cycles, total_energy