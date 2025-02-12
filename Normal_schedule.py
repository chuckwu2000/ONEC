import math
from Architecture_feature import ArchitectureFeatures

# The elementwise main op
elem_wise_ops = ["ADD", "SUB", "MUL", "LOGISTIC", "RSQRT", "SQUARED_DIFFERENCE", "SOFTMAX", "GELU", "LEAKY_RELU", "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", "TANH", "POW"]
# The mac main op
mac_ops = ["MEAN", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "MAX_POOL_2D", "BATCH_MATMUL"]
# The operation that will be fall back to CPU (not contain RESHAPE)
data_layout_ops = ["CONCATENATION", "SPLIT", "SPLIT_V", "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR", "PACK"]
fall_back_cpu_ops = data_layout_ops
# The input of the operation
unary_ops = ["LOGISTIC", "RSQRT", "SOFTMAX", "GELU", "LEAKY_RELU", "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", "TANH", "POW", "MEAN", "MAX_POOL_2D"]
binary_ops = ["ADD", "SUB", "MUL", "SQUARED_DIFFERENCE", "BATCH_MATMUL"]
trinary_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]

class Normal_scheduler:
    def __init__(self, graph, need_allocate_tensors, fragment_ratio = 0.15):
        self.graph = graph
        self.need_allocate_tensors = need_allocate_tensors
        # When the DF order stop at the operation, those wait consume tensors need to be store back to DRAM
        self.tensors_wait_consume = {}
        self.fragment_ratio = fragment_ratio
        # Since our memory allocation adapt greedy allocation, we need to reserve space for the fragmentation (default: 15%)
        reserved_ratio = 1 - self.fragment_ratio
        self.visited = [False for _ in range(len(self.graph.ops))]
        # NPU's SRAM limits
        self.sram_buffer_size = math.floor(ArchitectureFeatures.SRAM_MAX_SIZE * reserved_ratio)

    # Based on the DF order, but consider the limited SRAM size
    def normal_schedule(self):
        # Traverse the DF order
        ordered_ops = self.graph.ordered_ops
        for op in ordered_ops:
            if self.visited[op.opid]:
                continue
            self.visited[op.opid] = True
            opid = op.opid
            op_info = op.info
            opcode_index = op_info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

            # Operation that will execute in CPU
            if opcode_type in fall_back_cpu_ops:
                # Input tensor
                for tensor_id in op_info['inputs']:
                    # Skip the tensor that related to the split_axis or perm
                    buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                    if len(buffer) != 0:
                        continue
                    # Update the tensor metadata
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if tensor.cid == opid:
                            tensor.in_DRAM = True
                    # Update the tensors_wait_consume
                    if tensor_id in self.tensors_wait_consume:
                        self.tensors_wait_consume[tensor_id] -= 1    
                # Output tensor
                for tensor_id in op_info['outputs']:
                    # Update the tensor metadata
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        # Store the output tensor in the DRAM
                        if tensor.pid == opid:
                            tensor.in_DRAM = True
                # Clean the useless tensors
                self.clean_useless_tensors()
            # Operation that will execute in NPU
            else:
                # Operation will execute either in MAC engine or Element-wise engine
                if opcode_type in elem_wise_ops:
                    # Input tensor
                    unary, binary = 1, 2
                    input_nums = [unary, binary]
                    if opcode_type in unary_ops:
                        input_idx = 0
                    elif opcode_type in binary_ops:
                        input_idx = 1
                    for tensor_id in op_info['inputs'][0:input_nums[input_idx]]:
                        # Reuse the tensor in NPU's SRAM
                        if tensor_id in self.tensors_wait_consume:
                            ############################################################
                            # EX: When execute the Mul operation, the conv's output tensor had allocated in the SRAM
                            #    Conv
                            #    /  |
                            #  Add  |
                            #   |  /
                            #   Mul
                            ############################################################
                            # EX: When execute the Mul operation, the add's output tensor had allocated in the SRAM
                            #   Add
                            #    |
                            #   Mul
                            ############################################################
                            # EX: When execute the Add operation, the conv's output tensor had allocated in the SRAM
                            #   Conv
                            #    |
                            #   Add
                            ############################################################

                            # Update the tensors_wait_consume
                            self.tensors_wait_consume[tensor_id] -= 1
                        # Load the tensor from DRAM to SRAM
                        else:
                            ############################################################ 
                            # EX: When execute the Add operation, the input tensor had allocated in the DRAM
                            #   Transpose
                            #       |
                            #      Add
                            ############################################################
                            # Constant tensor
                            ############################################################

                            self.sram_buffer_size -= self.need_allocate_tensors[tensor_id].size
                            # Update the tensor metadata
                            for tensor in self.need_allocate_tensors[tensor_id].tensors:
                                if tensor.cid == opid:
                                    tensor.in_DRAM = True
                            # Update the tensors_wait_consume
                            wait_consume = len(self.graph.op_lookup_input[tensor_id])
                            if wait_consume > 0:
                                self.tensors_wait_consume.update({tensor_id: wait_consume - 1})
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        self.sram_buffer_size -= self.need_allocate_tensors[tensor_id].size
                        # Update the tensor metadata
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.pid == opid:
                                tensor.in_DRAM = False
                        # Update the tensors_wait_consume
                        wait_consume = len(self.graph.op_lookup_input[tensor_id])
                        if wait_consume > 0:
                            self.tensors_wait_consume.update({tensor_id: wait_consume})
                elif opcode_type in mac_ops:
                    # Input tensor
                    unary, binary, trinary = 1, 2, 3
                    input_nums = [unary, binary, trinary]
                    if opcode_type in unary_ops:
                        input_idx = 0
                    elif opcode_type in binary_ops:
                        input_idx = 1
                    elif opcode_type in trinary_ops:
                        input_idx = 2
                    for tensor_id in op_info['inputs'][0:input_nums[input_idx]]:
                        # Reuse the tensor in NPU's SRAM
                        if tensor_id in self.tensors_wait_consume:
                            ############################################################
                            # EX: When execute the Conv2 operation, the conv1's output tensor had allocated in the ME_OUTPUT_BUFFER
                            #   Conv1
                            #     |
                            #   Conv2
                            ############################################################
                            # EX: When execute the Conv operation, the add's output tensor had allocated in the EE_OUTPUT_BUFFER
                            #   Add
                            #    |
                            #   Conv
                            ############################################################

                            # Update the tensors_wait_consume
                            self.tensors_wait_consume[tensor_id] -= 1
                        # Load the tensor from DRAM to SRAM
                        else:
                            ############################################################
                            # EX: When execute the Conv operation, the input tensor had allocated in the DRAM
                            #   Transpose
                            #       |
                            #      Conv
                            ############################################################
                            # Constant tensor (ex: weight)
                            ############################################################

                            # Sometimes the empty tensor will be set to -1 default (ex: bias)
                            if tensor_id == -1:
                                continue
                            # Load tensor from DRAM to SRAM
                            self.sram_buffer_size -= self.need_allocate_tensors[tensor_id].size
                            # Update the tensors_wait_consume
                            wait_consume = len(self.graph.op_lookup_input[tensor_id])
                            if wait_consume > 0:
                                self.tensors_wait_consume.update({tensor_id: wait_consume - 1})
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        self.sram_buffer_size -= self.need_allocate_tensors[tensor_id].size
                        # Update the tensor metadata
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.pid == opid:
                                tensor.in_DRAM = False
                        # Update the tensors_wait_consume
                        wait_consume = len(self.graph.op_lookup_input[tensor_id])
                        if wait_consume > 0:
                            self.tensors_wait_consume.update({tensor_id: wait_consume})
            # Check whether concat this operation will overuse the SRAM
            can_keep_in_block = self.check_buffer_overuse()
            # If overuse the SRAM, order turn to weight reuse schedule
            if not can_keep_in_block:
                # Set the wait_consume's tensor's tensor metadata's storage to DRAM
                for tensor_id in self.tensors_wait_consume:
                    # Update the tensor metadata
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        # Check whether the tensor's self.visited[cid] is False, which means it didn't be consumed by the operation
                        if self.visited[tensor.cid] == False:
                            tensor.in_DRAM = True
                    # Update the tensors_wait_consume (set all the tensor's wait_consume to 0)
                    self.tensors_wait_consume.update({tensor_id: 0})
            else:
                pass
            # Clean the useless tensors
            self.clean_useless_tensors()
        return self.graph

    # Clean the useless tensors in the tensors_wait_consume
    def clean_useless_tensors(self):
        can_remove = []
        for tensor_id in self.tensors_wait_consume:
            if self.tensors_wait_consume[tensor_id] <= 0:
                # No need to use this tensor anymore, can release data of tensor in SRAM
                self.release_cache(self.need_allocate_tensors[tensor_id])
                can_remove.append(tensor_id)
        for tensor_id in can_remove:
            self.tensors_wait_consume.pop(tensor_id)

    # For release the cache
    def release_cache(self, tensor):
        tensor_size = tensor.size
        self.sram_buffer_size += tensor_size
            
    # Check whether the buffer overuse
    def check_buffer_overuse(self):
        if self.sram_buffer_size < 0:
            return False
        return True