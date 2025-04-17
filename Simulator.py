# This simulator is referenced from Planaria's implementation
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
need_requant_ops = op_classify.need_requant_ops
need_dequant_ops = op_classify.need_dequant_ops

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
        need_requant = False

        initial_dram_reads = 0
        final_dram_writes = 0
        dram_transfer_size = 0
        op_cycles = 0
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
            if ifm_elem_size == 8 and ofm_elem_size == 8:
                need_requant = True

            # ifm's elements
            ifm_elems = 1
            for dim in ifm_shape:
                ifm_elems *= dim
            # ofm's elements
            ofm_elems = 1
            for dim in ofm_shape:
                ofm_elems *= dim

            # Data transfer cycles
            # Our elementwise engine adopts a SIMD vector design
            # IFM's data transfer
            for tensor_metadata in self.tensor_info[inputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.cid == opid:
                    if tensor_metadata.in_DRAM:
                        initial_dram_reads += min(ifm_elems, ArchitectureFeatures.VECTOR_LEN) * (ifm_elem_size / 8)
                        dram_transfer_size += self.tensor_info[inputs[0]].size
                        break
            # OFM's data transfer
            for tensor_metadata in self.tensor_info[outputs[0]].tensors:
                # Find the corresponding tensor metadata
                if tensor_metadata.pid == opid:
                    if tensor_metadata.in_DRAM:
                        final_dram_writes += min(ofm_elems, ArchitectureFeatures.VECTOR_LEN) * (ofm_elem_size / 8)
                        dram_transfer_size += self.tensor_info[outputs[0]].size
                        break

            # Computations cycles
            # Element-wise operation will speedup by vectorization
            if op_type == "LOGISTIC":
                # Logistic(x) = 1 / (1 + exp(-x))
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"]
            elif op_type == "EXP":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["EXP"]
            elif op_type == "SOFTMAX":
                # Softmax(x) = exp(x) / sum(exp(x))
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["EXP"] + ArchitectureFeatures.output_cycles_per_elem["RECIPROCAL"] + \
                                ArchitectureFeatures.output_cycles_per_elem["MUL"])
                # sum(exp(x)) can't be vectorized
                reduce_sum_need = ifm_shape[-1]
                reduce_cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
                op_cycles += reduce_cycle_per_elem * reduce_sum_need
            elif op_type == "RSQRT":
                # Rsqrt(x) = 1 / sqrt(x) almost equal to 3*mul + 1*sub
                # Above reference: https://blog.csdn.net/qq_26499321/article/details/73724763
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["RSQRT"]
            elif op_type == "POW":
                # Pow(x, y) = x^y
                # Extract the exponent from the second input tensor, tflite store the data in little-endian format
                exp_tensor = tensors[inputs[1]]
                exp_buffer_data = bytes(self.buffers[exp_tensor['buffer']]['data'])
                # Parse the exp_buffer_data to get the exponent, '<': little-endian, 'f': float
                Exponent = int(struct.unpack('<f', exp_buffer_data)[0])
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"] * (Exponent - 1)
            elif op_type == "TANH":
                # Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["EXP"] * 2 + ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"] * 2 + \
                                ArchitectureFeatures.output_cycles_per_elem["RECIPROCAL"] + ArchitectureFeatures.output_cycles_per_elem["MUL"])
            elif op_type == "GELU":
                # Gelu(x) = x * logistic(1.702 * x)
                cycle_per_elem = (ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"] + ArchitectureFeatures.output_cycles_per_elem["MUL"] * 2)
            elif op_type == "QUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            elif op_type == "DEQUANTIZE":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
             
            # Exponantial need to do dequant first
            if need_requant and op_type in need_dequant_ops:
                cycle_per_elem += ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            if need_requant and op_type in need_requant_ops:
                cycle_per_elem += ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            op_cycles += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem
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
                need_requant = True
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
                        initial_dram_reads += min(ifm2_elems ,ArchitectureFeatures.VECTOR_LEN) * (ifm2_elem_size / 8)
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
            
            # Computations cycles
            # Element-wise operation will speedup by vectorization
            if op_type == "ADD":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
            elif op_type == "SUB":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
            elif op_type == "MUL":
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
            elif op_type == "DIV":
                # Div(x, y) = x * 1 / y
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["RECIPROCAL"] + ArchitectureFeatures.output_cycles_per_elem["MUL"]
            elif op_type == "SQUARED_DIFFERENCE":
                # SquaredDifference(x, y) = (x - y)(x - y)
                cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"] + ArchitectureFeatures.output_cycles_per_elem["MUL"]
            
            # Exponantial need to do dequant first
            if need_requant and op_type in need_dequant_ops:
                cycle_per_elem += ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            if need_requant and op_type in need_requant_ops:
                cycle_per_elem += ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            op_cycles += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem

        dram_transfer_size -= (initial_dram_reads + final_dram_writes)
        dram_bandwidth = ArchitectureFeatures.axi_bit_width / 8
        # First tile's read from DRAM & last tile's write to DRAM can't be overlapped
        latency = math.ceil(initial_dram_reads / float(dram_bandwidth)) + math.ceil(final_dram_writes / float(dram_bandwidth))
        dram_transfer_cycles = math.ceil(dram_transfer_size / float(dram_bandwidth))
        dma_transfer_cycles = max(0, dram_transfer_cycles - op_cycles) + latency
        total_cycles = op_cycles + dma_transfer_cycles
        return dma_transfer_cycles, op_cycles, total_cycles
    
    # Refer to Planaria's implementation
    def estimate_mac_op_cycles(self, opid: int, op_type: str) -> int:
        tensors = self.tensors
        info = self.ops[opid].info
        inputs = info.get("inputs")
        outputs = info.get("outputs")
        need_requant = False

        ofm = tensors[outputs[0]]
        ofm_shape = copy.deepcopy(ofm.get("shape"))
        if ofm.get("type") == "INT8":
            ifm_elem_size = 8
            ofm_elem_size = 8
            need_requant = True
        else:
            ifm_elem_size = 32
            ofm_elem_size = 32
        
        # OFM's size (bytes) & elements
        ofm_elems = 1
        for dim in ofm_shape:
            ofm_elems *= dim
        
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

        # Based on systolic array's architecture, compute how many tiles are needed
        tiling = self.tiling_compute(IC, OH, OW, OC, FH, FW, B)
        num_b, b = tiling['B/b']
        num_ic, ic = tiling['IC/ic']
        num_oh, oh = tiling['OH/oh']
        num_ow, ow = tiling['OW/ow']
        num_oc, oc = tiling['OC/oc']
        num_tiles = num_b * num_ic * num_oh * num_ow * num_oc

        ih = (oh - 1) * stride + FH
        iw = (ow - 1) * stride + FW
        # Compute ifm storage size
        ifm_storge_size = b * ih * iw * ic * (ifm_elem_size / 8)
        # Compute ofm storage size
        ofm_storge_size = b * oh * ow * oc * (ofm_elem_size / 8)
        # Compute weight storage size
        if op_type == "CONV_2D" or op_type == "DEPTHWISE_CONV_2D":
            weights_storage_size = oc * FH * FW * ic * (ofm_elem_size / 8)
            bias_storage_size = oh * ow * oc * (32 / 8)
        if op_type == "FULLY_CONNECTED":
            weights_storage_size = oc * FH * FW * ic * (ofm_elem_size / 8)

        # Check whether the tensors need to move from DRAM to SRAM
        # Check input tensor (transformer's V's FC needs additional handle)
        for tensor_metadata in self.tensor_info[inputs[0]].tensors:
            if tensor_metadata.cid == opid:
                if not tensor_metadata.in_DRAM:
                    ifm_storge_size = 0
                break
        if op_type == "FULLY_CONNECTED" and len(inputs) > 3:
            # input[0] is equal to input[3]
            for idx in range(4, len(inputs)):
                tensor = self.tensor_info[inputs[idx]]
                for tensor_metadata in tensor.tensors:
                    if tensor_metadata.cid == opid:
                        if not tensor_metadata.in_DRAM:
                            ifm_storge_size -= tensor.size
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
        
        # SRAM's writes and reads
        writes = {}
        reads = {}
        writes['input_buffer'] = ifm_storge_size
        # Perform bias addition in output buffer
        writes['output_buffer'] = ofm_storge_size + bias_storage_size
        writes['weight_buffer'] = weights_storage_size
        reads['output_buffer'] = ofm_storge_size

        if op_type == "DEPTHWISE_CONV_2D":
            for namespace in writes:
                writes[namespace] *= IC
            for namespace in reads:
                reads[namespace] *= IC

        # Compute the dram access cycles
        dram_transfer_cycles, latency = self.estimate_mac_engine_dram_access(tiling, writes, reads)

        # Computations cycles
        op_cycles = math.ceil(oc / ArchitectureFeatures.MAC_width) * oh * ow * math.ceil((ic * FH * FW) / ArchitectureFeatures.MAC_height)
        op_cycles *= ArchitectureFeatures.output_cycles_per_elem["MAC"]
        op_cycles *= num_tiles
        # Depthwise convolution will multiply the output channel
        if op_type == "DEPTHWISE_CONV_2D":
            op_cycles *= ofm_shape[3]
        if need_requant:
            cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
            op_cycles += math.ceil(ofm_elems / ArchitectureFeatures.VECTOR_LEN) * cycle_per_elem

        dma_transfer_cycles = max(0, dram_transfer_cycles - op_cycles) + latency
        total_cycles = op_cycles + dma_transfer_cycles
        return dma_transfer_cycles, op_cycles, total_cycles
    
    # Based on the systolic array's architecture, compute how many tiles are needed
    # Modify by own self
    def tiling_compute(self, IC, OH, OW, OC, FH, FW, B):
        tiling_dict = {'B/b': (1, B), 'IC/ic': (1, IC), 'OH/oh': (1, OH), 'OW/ow': (1, OW), 'OC/oc': (1, OC)}
        # ic_num = math.ceil(IC / ArchitectureFeatures.MAC_height)
        per_ic = math.ceil(ArchitectureFeatures.MAC_height / (FH * FW))
        ic_num = math.ceil(IC / per_ic)
        oc_num = math.ceil(OC / ArchitectureFeatures.MAC_width)
        # Get max ic per tile
        tiling_dict['IC/ic'] = (ic_num, per_ic)
        # Get max oc per tile
        tiling_dict['OC/oc'] = (oc_num, ArchitectureFeatures.MAC_width)

        # Since we had perform tensor tiling, there is no risk to exceed the buffer size
        # We choose 1 to be each tile's ow, oh
        tiling_dict["OH/oh"] = (OH, 1)
        tiling_dict["OW/ow"] = (OW, 1)
        
        # In here, we assume that the tile's input/output won't exceed the buffer size
        return tiling_dict

    # Planaria's DRAM access    
    def estimate_mac_engine_dram_access(self, tiling, writes, reads) -> int:
        # If tile loop depends on the namespace index, make the read size larger
        tile_deps = {}
        tile_deps['B/b']   = {'input_buffer': True, 'weight_buffer': False, 'output_buffer': True}
        tile_deps['OW/ow'] = {'input_buffer': True, 'weight_buffer': False, 'output_buffer': True}
        tile_deps['OH/oh'] = {'input_buffer': True, 'weight_buffer': False, 'output_buffer': True}
        tile_deps['IC/ic'] = {'input_buffer': True, 'weight_buffer': True, 'output_buffer': False}
        tile_deps['OC/oc'] = {'input_buffer': False, 'weight_buffer': True, 'output_buffer': True}
        # Best order
        best_order = ('OW/ow', 'OH/oh', 'IC/ic', 'OC/oc', 'B/b')
        write_promote = {'weight_buffer': True, 'input_buffer': True, 'output_buffer': True}
        read_promote = {'output_buffer': True}

        # max: needed for the op(include all the tiles)
        # initial: needed for the first tile
        max_write_size = {}
        max_read_size = {}
        for namespace in writes:
            max_write_size[namespace] = writes[namespace]
        for namespace in reads:
            max_read_size[namespace] = reads[namespace]

        for loop in best_order:
            num_tiles, tile_size = tiling[loop]
            # Promote all writes
            for namespace in writes:
                # Promote is true
                if write_promote[namespace]:
                    if tile_deps[loop][namespace]:
                        writes[namespace] *= num_tiles
            # Promote all reads
            for namespace in reads:
                # Promote is true
                if read_promote[namespace]:
                    if tile_deps[loop][namespace]:
                        reads[namespace] *= num_tiles
        # Prefer to load full weight buffer
        max_write_size['weight_buffer'] = writes['weight_buffer']

        initial_dram_reads = 0
        final_dram_writes = 0
        for namespace in max_write_size:
            initial_dram_reads += max_write_size[namespace]
        for namespace in max_read_size:
            final_dram_writes += max_read_size[namespace]
        total_dram_reads = 0
        total_dram_writes = 0
        for namespace in writes:
            total_dram_reads += writes[namespace]
        for namespace in reads:
            total_dram_writes += reads[namespace]
        total_dram_accesses = total_dram_writes + total_dram_reads
        middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes

        # Compute the latency (fisrt tile's read & last tile's write can't be overlapped)
        dram_bandwidth = ArchitectureFeatures.axi_bit_width / 8
        latency = math.ceil(initial_dram_reads / float(dram_bandwidth)) + math.ceil(final_dram_writes / float(dram_bandwidth))
        memory_cycles_required = int(math.ceil(float(middle_dram_accesses) / dram_bandwidth))
        return memory_cycles_required, latency

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