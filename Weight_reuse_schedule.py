import math
import copy
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

class Weight_reuse_scheduler:
    def __init__(self, graph, need_allocate_tensors, same_layer_next_opids, fragment_ratio = 0.15):
        self.graph = graph
        self.need_allocate_tensors = need_allocate_tensors
        # same_layer_next_opids[x] = (x's head_opid, x's next_opid, #ops in layer)
        self.same_layer_next_opids = same_layer_next_opids
        # When the DF order stop at the operation, those wait consume tensors need to be store back to DRAM
        self.tensors_wait_consume = {}
        self.fragment_ratio = fragment_ratio
        self.visited = [False for _ in range(len(self.graph.ops))]
        self.scheduled = [False for _ in range(len(self.graph.ops))]
        self.new_order = 0
        self.weights_reuse_mapping = {}
        # Since our memory allocation adapt greedy allocation, we need to reserve space for the fragmentation (default: 15%)
        reserved_ratio = 1 - self.fragment_ratio
        # NPU's SRAM limits
        self.sram_buffer_size = math.floor(ArchitectureFeatures.SRAM_MAX_SIZE * reserved_ratio)

    # Based on the DF order, but consider the limited SRAM size
    def weight_reuse_schedule(self):
        # Record the start of the block
        block_start_order = 0
        # Traverse the DF order to reschedule
        ordered_ops = self.graph.ordered_ops
        for order, op in enumerate(ordered_ops):
            if self.visited[op.opid] and self.scheduled[op.opid]:
                # Some operations be set to visited because the tensor usage had been counted, in here we need to update the tensor's wait_consume
                self.update_tensor_wait_consume(op)
                self.clean_useless_tensors()
                continue
            self.visited[op.opid] = True
            opid = op.opid
            op_info = op.info
            opcode_index = op_info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

            # The number of the operation in the same layer
            num_of_op_in_layer = self.same_layer_next_opids.get(op.opid, [-1, -1, 1])[2]

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
                            # EX: When execute the Mul operation, the conv's output tensor had allocated in the EE_INPUT_BUFFER
                            #    Conv
                            #    /  |
                            #  Add  |
                            #   |  /
                            #   Mul
                            ############################################################
                            # EX: When execute the Mul operation, the add's output tensor had allocated in the EE_OUTPUT_BUFFER
                            #   Add
                            #    |
                            #   Mul
                            ############################################################
                            # EX: When execute the Add operation, the conv's output tensor had allocated in the ME_OUTPUT_BUFFER
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
 
                            if tensor_id == -1:
                                continue
                            self.sram_buffer_size -= (self.need_allocate_tensors[tensor_id].size * num_of_op_in_layer)
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
                        self.sram_buffer_size -= (self.need_allocate_tensors[tensor_id].size * num_of_op_in_layer)
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
                            # EX: When execute the Conv2 operation, the conv1's output tensor had allocated in the SRAM
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
                            
                            # Sometimes the tensor maybe empty, ex: no bias
                            if tensor_id == -1:
                                continue
                            self.sram_buffer_size -= (self.need_allocate_tensors[tensor_id].size * num_of_op_in_layer)
                            # Update the tensors_wait_consume
                            wait_consume = len(self.graph.op_lookup_input[tensor_id])
                            if wait_consume > 0:
                                self.tensors_wait_consume.update({tensor_id: wait_consume - 1})
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        self.sram_buffer_size -= (self.need_allocate_tensors[tensor_id].size * num_of_op_in_layer)
                        # Update the tensor metadata
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.pid == opid:
                                tensor.in_DRAM = False
                        # Update the tensors_wait_consume
                        wait_consume = len(self.graph.op_lookup_input[tensor_id])
                        if wait_consume > 0:
                            self.tensors_wait_consume.update({tensor_id: wait_consume})
            # Update the visited, since we had already count the tensor usage in the same layer
            self.update_visited(op)
            # Check whether concat this operation will overuse the SRAM
            can_keep_in_block = self.check_buffer_overuse()
            # If overuse the SRAM, order turn to weight reuse schedule
            if not can_keep_in_block:
                # TODO: may need to check the stop opid whether is layer's head, it may have diff aspects on the memory overuse
                self.weights_reuse_mapping[opid] = self.new_order
                self.new_order += 1
                self.scheduled[opid] = True
                block_end_order = order
                self.perform_weight_reuse(block_start_order, block_end_order)
                block_start_order = order + 1
                # Set the wait_consume's tensor's tensor metadata's storage to DRAM
                for tensor_id in self.tensors_wait_consume:
                    # Update the tensor metadata
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        # Check whether the tensor's self.visited[cid] is False, which means it didn't be consumed by the operation
                        if self.visited[tensor.cid] == False:
                            tensor.in_DRAM = True
                    # Update the tensors_wait_consume (set all the tensor's wait_consume to 0)
                    self.tensors_wait_consume.update({tensor_id: 0})
                self.clean_sram()
            else:
                self.weights_reuse_mapping[opid] = self.new_order
                self.new_order += 1
                self.scheduled[opid] = True
            # Clean the useless tensors
            self.clean_useless_tensors()
        # Update the order of the operations
        for op in self.graph.ops:
            op.schedule_order = self.weights_reuse_mapping[op.opid]
        # Update the schedule order in the ordered_ops
        self.graph.ordered_ops = sorted(self.graph.ops, key=lambda x: x.schedule_order)
        # Update the operators in the split_graph
        self.graph.operators = []
        for op in self.graph.ordered_ops:
            self.graph.operators.append(op.info)
        return self.graph
    
    # Update the visited op's tensor's wait_consume
    def update_tensor_wait_consume(self, op):
        op_info = op.info
        consume_set = set()
        produce_set = set()
        for tensor_id in op_info['inputs']:
            consume_set.add(tensor_id)
        for tensor_id in op_info['outputs']:
            produce_set.add(tensor_id)
        if -1 in consume_set:
            consume_set.remove(-1)
        for tensor_id in consume_set:
            if tensor_id in self.tensors_wait_consume:
                self.tensors_wait_consume[tensor_id] -= 1
        for tensor_id in produce_set:
            wait_consume = len(self.graph.op_lookup_input[tensor_id])
            if wait_consume > 0:
                self.tensors_wait_consume.update({tensor_id: wait_consume})

    # Update sibling operation's visited
    def update_visited(self, op):
        now_opid = op.opid
        head_opid = self.same_layer_next_opids.get(now_opid, [-1, -1, 1])[0]
        if head_opid == -1:
            return
        tmp_opid = head_opid
        while tmp_opid != -1:
            self.visited[tmp_opid] = True
            tmp_opid = self.same_layer_next_opids.get(tmp_opid, [-1, -1, 1])[1]

    # Clean the useless tensors in the tensors_wait_consume
    def clean_useless_tensors(self):
        can_remove = []
        for tensor_id in self.tensors_wait_consume:
            if self.tensors_wait_consume[tensor_id] <= 0:
                # No need to use this tensor anymore, can release data of tensor in SRAM, but looks like can't clean well
                self.release_cache(self.need_allocate_tensors[tensor_id])
                can_remove.append(tensor_id)
        for tensor_id in can_remove:
            self.tensors_wait_consume.pop(tensor_id)

    # For release the cache
    def release_cache(self, tensor):
        tensor_size = tensor.size
        self.sram_buffer_size += tensor_size

    # Clean the whole SRAM
    def clean_sram(self):
        self.sram_buffer_size = math.floor(ArchitectureFeatures.SRAM_MAX_SIZE * (1 - self.fragment_ratio))
            
    # Check whether the buffer overuse
    def check_buffer_overuse(self):
        if self.sram_buffer_size < 0:
            return False
        return True

    # Perform the weight reuse schedule
    def perform_weight_reuse(self, block_start_order, block_end_order):
        # Store all the operations that on the one of the path in the block
        ops_on_next_path = []
        for op in self.graph.ordered_ops[block_start_order: block_end_order + 1]:
            next_opid = self.same_layer_next_opids.get(op.opid, [-1, -1, 1])[1]
            if next_opid != -1:
                ops_on_next_path.append(next_opid)
        while len(ops_on_next_path) > 0:
            new_list = []
            for now_opid in ops_on_next_path:
                now_op = self.graph.ops[now_opid]
                next_opid = self.same_layer_next_opids.get(now_op.opid, [-1, -1, 1])[1]
                if next_opid != -1:
                    new_list.append(next_opid)
                if self.scheduled[now_opid]:
                    continue
                # Update the now_op's tensor metadata same as the head_op's tensor metadata
                head_opid = self.same_layer_next_opids.get(now_op.opid, [-1, -1, 1])[0]
                if head_opid != -1:
                    self.update_tensor_metadata(head_opid, now_opid)
                self.weights_reuse_mapping[now_opid] = self.new_order
                self.new_order += 1
                self.scheduled[now_opid] = True
            ops_on_next_path = new_list

    # Update the tensor metadata of the sibling operation
    def update_tensor_metadata(self, head_opid, now_opid):
        # Update the input tensor metadata
        for head_tensor_id, now_tensor_id in zip(self.graph.ops[head_opid].info['inputs'], self.graph.ops[now_opid].info['inputs']):
            # For store the head_tensor's metadata
            head_in_DRAM = True
            # Fetch the metadata of the head_tensor (focus on the in_DRAM and internal_memory_storage)
            for tensor in self.need_allocate_tensors[head_tensor_id].tensors:
                if tensor.cid == head_opid:
                    head_in_DRAM = tensor.in_DRAM
                    break
            for tensor in self.need_allocate_tensors[now_tensor_id].tensors:
                if tensor.cid == now_opid:
                    tensor.in_DRAM = head_in_DRAM
                    break
        # Update the output tensor metadata
        for head_tensor_id, now_tensor_id in zip(self.graph.ops[head_opid].info['outputs'], self.graph.ops[now_opid].info['outputs']):
            head_in_DRAM = True
            # Fetch the metadata of the head_tensor (focus on the in_DRAM and internal_memory_storage)
            for tensor in self.need_allocate_tensors[head_tensor_id].tensors:
                if tensor.pid == head_opid:
                    head_in_DRAM = tensor.in_DRAM
                    break
            for tensor in self.need_allocate_tensors[now_tensor_id].tensors:
                if tensor.pid == now_opid:
                    tensor.in_DRAM = head_in_DRAM

    class virtual_memory_allocate:
        def __init__(self, graph, need_allocate_tensors, same_layer_next_opids):
            self.graph = graph
            self.allocated_opids = set()
            self.allocated_tensors = set()
            self.tensor_info = need_allocate_tensors
            self.same_layer_next_opids = same_layer_next_opids
            
        # Virtual memory allocation for calculate real SRAM peak usage (give the opids that in weight reuse block)
        def virtual_memory_allocate(self, opids):
            ops = self.graph.ops
            # Step 1: Record the same layer's opids (only handle the opids that not yet be allocated)
            need_allocate_opids = set()
            candidate_opids = opids - self.allocated_opids
            while len(candidate_opids) != 0:
                tmp_opids = []
                for opid in candidate_opids:
                    need_allocate_opids.add(opid)
                    next_opid = self.same_layer_next_opids.get(opid, [-1, -1, 1])[1]
                    if next_opid != -1:
                        tmp_opids.append(next_opid)
                candidate_opids = tmp_opids - self.allocated_opids
            self.allocated_opids.update(need_allocate_opids)
            
            # Step 2: Update tensor's live range
            need_allocate_tensors = set()
            for opid in need_allocate_opids:
                for tensor_id in ops[opid].info['inputs'] + ops[opid].info['outputs']:
                    # Update the tensor's first time used
                    if ops[opid].schedule_order < self.tensor_info[tensor_id].live_range.get('first_time_used', len(ops)):
                        self.tensor_info[tensor_id].live_range['first_time_used'] = ops[opid].schedule_order
                    # Update the tensor's last time used
                    if ops[opid].schedule_order > self.tensor_info[tensor_id].live_range.get('last_time_used', -1):
                        self.tensor_info[tensor_id].live_range['last_time_used'] = ops[opid].schedule_order
                    need_allocate_tensors.add(tensor_id)
            self.allocated_tensors.update(need_allocate_tensors)
            
            # Step 3: Initialize memory allocation
            total_used_size = 0
            size_non_inc_allocated_tensors = []
            
            # Step 4: Greedy by size
            # Step 4.1: Sort tensors by size
            sorted_tensors = sorted(self.allocated_tensors, key=lambda x: self.tensor_info[x].size, reverse=True)
            # Step 4.2: Allocate memory (greedy by size)
            for tensor in sorted_tensors:
                prev_start = 0
                best_start = None
                smallest_gap = ArchitectureFeatures.SRAM_MAX_SIZE

                for allocated_tensor in size_non_inc_allocated_tensors:
                    max_first_op = max(self.tensor_info[tensor].live_range['first_time_used'], self.tensor_info[allocated_tensor].live_range['first_time_used'])
                    min_last_op = min(self.tensor_info[tensor].live_range['last_time_used'], self.tensor_info[allocated_tensor].live_range['last_time_used'])
                    # Live range has overlap (try to find the gap between this tensor's start address & previous start)
                    if max_first_op <= min_last_op:
                        gap = allocated_tensor.start_address - prev_start
                        if gap >= self.tensor_info[tensor].size and gap < smallest_gap:
                            smallest_gap = gap
                            best_start = prev_start
                        prev_start = max(prev_start, allocated_tensor.end_address)
                if best_start is None:
                    best_start = prev_start
                tensor.start_address = best_start
                tensor.end_address = best_start + tensor.size
                total_used_size = max(total_used_size, tensor.end_address)
                # Check total used size whether over the SRAM's size
                if total_used_size > ArchitectureFeatures.SRAM_MAX_SIZE:
                    return False
                size_non_inc_allocated_tensors.append(tensor)
            # The tensors of this block can be greedy allocated without overuse the SRAM
            return True