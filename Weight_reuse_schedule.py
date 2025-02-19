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
    def __init__(self, graph, need_allocate_tensors, same_layer_next_opids):
        self.graph = graph
        self.need_allocate_tensors = need_allocate_tensors
        # same_layer_next_opids[x] = (x's head_opid, x's next_opid, #ops in layer)
        self.same_layer_next_opids = same_layer_next_opids
        self.visited = [False for _ in range(len(self.graph.ops))]
        self.need_virtual_memory_allocate_opids = set()
        self.new_order = 0
        self.weights_reuse_mapping = {}
        self.virtual_memory_allocator = self.virtual_memory_allocate(graph, same_layer_next_opids)

    # Based on the DF order, but consider the limited SRAM size
    def weight_reuse_schedule(self):
        # Record the start of the block
        block_start_order = 0
        # Traverse the DF order to reschedule
        ordered_ops = self.graph.ordered_ops
        for order, op in enumerate(ordered_ops):
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
                # Output tensor
                for tensor_id in op_info['outputs']:
                    # Update the tensor metadata
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        # Store the output tensor in the DRAM
                        if tensor.pid == opid:
                            tensor.in_DRAM = True
            # Operation that will execute in NPU
            else:
                # Operation will execute either in MAC engine or Element-wise engine
                # Reuse the tensor in NPU's SRAM
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
                if opcode_type in elem_wise_ops:
                    self.need_virtual_memory_allocate_opids.add(opid)
                    # Input tensor
                    unary, binary = 1, 2
                    input_nums = [unary, binary]
                    if opcode_type in unary_ops:
                        input_idx = 0
                    elif opcode_type in binary_ops:
                        input_idx = 1
                    for tensor_id in op_info['inputs'][0:input_nums[input_idx]]:
                        if tensor_id == -1:
                            continue
                        # When encount the constant tensor in input, it means the tensor had allocated in the DRAM
                        # Update the tensor metadata
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            for tensor in self.need_allocate_tensors[tensor_id].tensors:
                                if tensor.cid == opid:
                                    tensor.in_DRAM = True
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        # Update the tensor metadata
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.pid == opid:
                                tensor.in_DRAM = False
                # Reuse the tensor in NPU's SRAM
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
                elif opcode_type in mac_ops:
                    self.need_virtual_memory_allocate_opids.add(opid)
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
                        # Sometimes the tensor maybe empty, ex: no bias
                        if tensor_id == -1:
                            continue
                        # When encount the constant tensor in input, it means the tensor had allocated in the DRAM
                        # Update the tensor metadata
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            for tensor in self.need_allocate_tensors[tensor_id].tensors:
                                if tensor.cid == opid:
                                    tensor.in_DRAM = True
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        # Update the tensor metadata
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.pid == opid:
                                tensor.in_DRAM = False
            # Check whether concat this operation will overuse the SRAM
            overuse_sram, self.need_allocate_tensors = self.virtual_memory_allocator.virtual_memory_allocate(self.need_virtual_memory_allocate_opids, self.need_allocate_tensors)
            if overuse_sram:
                # Then plan the weight reuse order mapping
                block_end_order = order - 1
                self.perform_weight_reuse(block_start_order, block_end_order)
                block_start_order = order
                # Set the out of block tensors to DRAM
                self.virtual_memory_allocator.update_outer_block_tensor_metadata(self.visited, self.need_allocate_tensors)

                # Reinitialize the virtual memory allocator
                self.virtual_memory_allocator.re_init()
                # Start from new block's first op
                self.need_virtual_memory_allocate_opids = set()
                self.need_virtual_memory_allocate_opids.add(opid)
                _, self.need_allocate_tensors = self.virtual_memory_allocator.virtual_memory_allocate(self.need_virtual_memory_allocate_opids, self.need_allocate_tensors)
                self.weights_reuse_mapping[opid] = self.new_order
                self.new_order += 1
            else:
                self.weights_reuse_mapping[opid] = self.new_order
                self.new_order += 1
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
                if self.visited[now_opid]:
                    continue
                # Update the now_op's tensor metadata same as the head_op's tensor metadata
                head_opid = self.same_layer_next_opids.get(now_op.opid, [-1, -1, 1])[0]
                if head_opid != -1:
                    self.update_sibling_tensor_metadata(head_opid, now_opid)
                self.weights_reuse_mapping[now_opid] = self.new_order
                self.new_order += 1
                self.visited[now_opid] = True
            ops_on_next_path = new_list

    # Update the tensor metadata of the sibling operation
    def update_sibling_tensor_metadata(self, head_opid, now_opid):
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
                    break     

    class virtual_memory_allocate:
        def __init__(self, graph, same_layer_next_opids):
            self.graph = graph
            self.same_layer_next_opids = same_layer_next_opids
            self.allocated_opids = set()
            self.allocated_tensors = set()

        def re_init(self):
            self.allocated_opids = set()
            self.allocated_tensors = set()
            
        # Virtual memory allocation for calculate real SRAM peak usage (give the opids that in weight reuse block)
        def virtual_memory_allocate(self, opids, original_tensor_info):
            ops = self.graph.ops
            tensor_info = copy.deepcopy(original_tensor_info)
            # Step 1: Record the same layer's opids (only handle the opids that not yet be allocated)
            need_allocate_opids = set()
            candidate_opids = opids - self.allocated_opids
            while len(candidate_opids) != 0:
                tmp_opids = set()
                for opid in candidate_opids:
                    need_allocate_opids.add(opid)
                    next_opid = self.same_layer_next_opids.get(opid, [-1, -1, 1])[1]
                    if next_opid != -1:
                        tmp_opids.add(next_opid)
                candidate_opids = tmp_opids - self.allocated_opids
            self.allocated_opids.update(need_allocate_opids)
            
            # Step 2: Update tensor's live range
            need_allocate_tensors = set()
            for opid in need_allocate_opids:
                for tensor_id in ops[opid].info['inputs'] + ops[opid].info['outputs']:
                    # Update the tensor's first time used
                    if ops[opid].schedule_order < tensor_info[tensor_id].live_range.get('first_time_used', len(ops)):
                        tensor_info[tensor_id].live_range['first_time_used'] = ops[opid].schedule_order
                    # Update the tensor's last time used
                    if ops[opid].schedule_order > tensor_info[tensor_id].live_range.get('last_time_used', -1):
                        tensor_info[tensor_id].live_range['last_time_used'] = ops[opid].schedule_order
                    need_allocate_tensors.add(tensor_id)
            self.allocated_tensors.update(need_allocate_tensors)
            
            # Step 3: Initialize memory allocation
            total_used_size = 0
            size_non_inc_allocated_tensors = []
            
            # Step 4: Greedy by size
            # Step 4.1: Sort tensors by size
            sorted_tensors = sorted(self.allocated_tensors, key=lambda x: tensor_info[x].size, reverse=True)
            # Step 4.2: Allocate memory (greedy by size)
            for tensor_id in sorted_tensors:
                prev_start = 0
                best_start = None
                smallest_gap = ArchitectureFeatures.SRAM_MAX_SIZE

                for allocated_tensor in size_non_inc_allocated_tensors:
                    max_first_op = max(tensor_info[tensor_id].live_range['first_time_used'], tensor_info[allocated_tensor].live_range['first_time_used'])
                    min_last_op = min(tensor_info[tensor_id].live_range['last_time_used'], tensor_info[allocated_tensor].live_range['last_time_used'])
                    # Live range has overlap (try to find the gap between this tensor's start address & previous start)
                    if max_first_op <= min_last_op:
                        gap = tensor_info[allocated_tensor].start_addr - prev_start
                        if gap >= tensor_info[tensor_id].size and gap < smallest_gap:
                            smallest_gap = gap
                            best_start = prev_start
                        prev_start = max(prev_start, tensor_info[allocated_tensor].end_addr)
                if best_start is None:
                    best_start = prev_start
                tensor_info[tensor_id].start_addr = best_start
                tensor_info[tensor_id].end_addr = best_start + tensor_info[tensor_id].size
                total_used_size = max(total_used_size, tensor_info[tensor_id].end_addr)
                # Update the tensor metadata's start address & end address
                self.update_inner_block_tensor_metadata(self.allocated_opids, tensor_info)
                # Check total used size whether over the SRAM's size
                # If over, return the original tensor info (which means exclude the current candidate opids)
                if total_used_size > ArchitectureFeatures.SRAM_MAX_SIZE:
                    return True, original_tensor_info
                size_non_inc_allocated_tensors.append(tensor_id)
            # The tensors of this block can be greedy allocated without overuse the SRAM
            return False, tensor_info
        
        def update_inner_block_tensor_metadata(self, allocated_opids, tensor_info):
            for tensor_id in self.allocated_tensors:
                for tensor_metadata in tensor_info[tensor_id].tensors:
                    if tensor_metadata.cid in allocated_opids:
                        tensor_metadata.start_address = tensor_info[tensor_id].start_addr
                        tensor_metadata.end_address = tensor_info[tensor_id].end_addr
        
        # Update those tensor out the block, those tensor will be stored in the DRAM
        def update_outer_block_tensor_metadata(self, visited, tensor_info):
            for tensor_id in self.allocated_tensors:
                for tensor_metadata in tensor_info[tensor_id].tensors:
                    if not visited[tensor_metadata.cid]:
                        tensor_metadata.in_DRAM = True    