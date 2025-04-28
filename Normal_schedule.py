from Architecture_feature import ArchitectureFeatures
from OpClassify import Op_Classify

op_classify = Op_Classify()
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops
fall_back_cpu_ops = op_classify.fall_back_cpu_ops
unary_ops = op_classify.unary_ops
binary_ops = op_classify.binary_ops
trinary_ops = op_classify.trinary_ops

class Normal_scheduler:
    def __init__(self, graph, need_allocate_tensors):
        self.graph = graph
        self.tensor_info = need_allocate_tensors
        self.visited = [False for _ in range(len(self.graph.ops))]
        self.tensor_in_SRAM = set()
        self.need_virtual_allocate_opids = set()
        self.virtual_tensor_allocator = self.virtual_tensor_allocate(graph)

    def normal_schedule(self):
        ordered_ops = self.graph.ordered_ops
        for op in ordered_ops:
            opid = op.opid
            if self.visited[opid]:
                continue
            # Step 1: Update the visited opid
            self.visited[opid] = True
            
            # Step 2: Perform virtual tensor allocation to calculate the real SRAM peak usage
            opcode_index = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            # Only update the opids in the block, if opcode_type is element-wise or mac
            if opcode_type in elem_wise_ops or opcode_type in mac_ops:
                self.need_virtual_allocate_opids.add(opid)
                overuse_sram = self.virtual_tensor_allocator.virtual_tensor_allocate(self.need_virtual_allocate_opids, self.tensor_info)
                if overuse_sram:
                    if len(self.need_virtual_allocate_opids) == 1:
                        raise Exception(f"Can't put single op's tensor in SRAM, op info: {op.info}")
                    self.tensor_in_SRAM = set()
                    self.need_virtual_allocate_opids = set()
                    self.need_virtual_allocate_opids.add(opid)
                    _ = self.virtual_tensor_allocator.virtual_tensor_allocate(self.need_virtual_allocate_opids, self.tensor_info)

            # Step 3: Update the tensor metadata (tensor's storage area)
            # Operation that will execute in CPU
            if opcode_type in fall_back_cpu_ops:
                # Update each opid's tensor metadata
                op_info = self.graph.ops[opid].info
                # Input tensor
                for tensor_id in op_info['inputs']:
                    # Skip the tensor that related to the split_axis or perm
                    buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                    if len(buffer) != 0:
                        continue
                    # Update the tensor metadata
                    for tensor in self.tensor_info[tensor_id].tensors:
                        if tensor.cid == opid:
                            tensor.in_DRAM = True
                # Output tensor
                for tensor_id in op_info['outputs']:
                    # Update the tensor metadata
                    for tensor in self.tensor_info[tensor_id].tensors:
                        # Store the output tensor in the DRAM
                        if tensor.pid == opid:
                            tensor.in_DRAM = True
            # Operation that will execute in NPU's element-wise unit
            elif opcode_type in elem_wise_ops:
                op_info = self.graph.ops[opid].info
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
                    # Set the tensor metadata, if the tensor had allocated in the SRAM
                    if tensor_id in self.tensor_in_SRAM:
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.cid == opid:
                                tensor_metadata.in_DRAM = False
                    # The tensor may cross the block (in previous block in_DRAM := False), so need to set the tensor's in_DRAM
                    else:
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.cid == opid:
                                tensor_metadata.in_DRAM = True
                        self.tensor_in_SRAM.add(tensor_id)
                # Output tensor
                for tensor_id in op_info['outputs']:
                    # Update the tensor metadata
                    for tensor_metadata in self.tensor_info[tensor_id].tensors:
                        if tensor_metadata.pid == opid:
                            tensor_metadata.in_DRAM = False
                    self.tensor_in_SRAM.add(tensor_id)
            # Operation that will execute in NPU's MAC unit
            elif opcode_type in mac_ops:
                op_info = self.graph.ops[opid].info
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
                    # Set the tensor metadata, if the tensor had allocated in the SRAM
                    if tensor_id in self.tensor_in_SRAM:
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.cid == opid:
                                tensor_metadata.in_DRAM = False
                    # The tensor may cross the block (in previous block in_DRAM := False), so need to set the tensor's in_DRAM
                    else:
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.cid == opid:
                                tensor_metadata.in_DRAM = True
                        self.tensor_in_SRAM.add(tensor_id)
                # Output tensor
                for tensor_id in op_info['outputs']:
                    # Update the tensor metadata
                    for tensor_metadata in self.tensor_info[tensor_id].tensors:
                        if tensor_metadata.pid == opid:
                            tensor_metadata.in_DRAM = False
                    self.tensor_in_SRAM.add(tensor_id)
        # Update the allocated tensors
        for tensor_id in self.tensor_info:
            for tensor_metadata in self.tensor_info[tensor_id].tensors:
                if tensor_metadata.in_DRAM == False:
                    self.allocated_tensors[tensor_id] = self.tensor_info[tensor_id]
        return self.graph
    
    class virtual_tensor_allocate:
        def __init__(self, graph):
            self.graph = graph

        # Virtual memory allocation for calculate real SRAM peak usage (give the opids that in the block)
        def virtual_tensor_allocate(self, opids, tensor_info):
            ops = self.graph.ops
            # Step 1: Update tensor's live range
            need_allocate_tensor_ids = set()
            for opid in opids:
                for tensor_id in ops[opid].info['inputs'] + ops[opid].info['outputs']:
                    if tensor_id == -1:
                        continue
                    # Update the tensor's first time used
                    if ops[opid].schedule_order < tensor_info[tensor_id].live_range.get('first_time_used', len(ops)):
                        tensor_info[tensor_id].live_range['first_time_used'] = ops[opid].schedule_order
                    # Update the tensor's last time used
                    if ops[opid].schedule_order > tensor_info[tensor_id].live_range.get('last_time_used', -1):
                        tensor_info[tensor_id].live_range['last_time_used'] = ops[opid].schedule_order
                    need_allocate_tensor_ids.add(tensor_id)

            # Step 2: Initialize memory allocation
            total_used_size = 0
            ordered_allocated_tensors = []

            # Step 3: Greedy by size
            # Step 3.1: Sort tensors by size
            sorted_tensor_ids = sorted(need_allocate_tensor_ids, key=lambda x: tensor_info[x].size, reverse=True)
            # Step 3.2: Allocate memory
            for tensor_id in sorted_tensor_ids:
                prev_start = 0
                best_start = None
                smallest_gap = ArchitectureFeatures.SRAM_MAX_SIZE

                for allocated_tensor in ordered_allocated_tensors:
                    max_first_op = max(tensor_info[tensor_id].live_range['first_time_used'], tensor_info[allocated_tensor].live_range['first_time_used'])
                    min_last_op = min(tensor_info[tensor_id].live_range['last_time_used'], tensor_info[allocated_tensor].live_range['last_time_used'])
                    # Live range has overlap (try to find the gap between this tensor's start address & previous start)
                    if max_first_op <= min_last_op:
                        gap = tensor_info[allocated_tensor].sram_start_addr - prev_start
                        if gap >= tensor_info[tensor_id].size and gap < smallest_gap:
                            smallest_gap = gap
                            best_start = prev_start
                        prev_start = max(prev_start, tensor_info[allocated_tensor].sram_end_addr)
                # If no large enough gap can reuse or this tensor's live range overlap with all the allocated tensors,
                # prev_start will start from the highest end address
                if best_start is None:
                    best_start = prev_start
                tensor_info[tensor_id].sram_start_addr = best_start
                tensor_info[tensor_id].sram_end_addr = best_start + tensor_info[tensor_id].size
                total_used_size = max(total_used_size, tensor_info[tensor_id].sram_end_addr)
                # Check total used size whether over the SRAM's size
                if total_used_size > ArchitectureFeatures.SRAM_MAX_SIZE:
                    return True
                ordered_allocated_tensors.append(tensor_id)
                # Sort the allocated tensors by start address(ascending) and if the start address is the same, sort by size(descending)
                ordered_allocated_tensors = sorted(ordered_allocated_tensors, key=lambda t: (tensor_info[t].sram_start_addr, -tensor_info[t].size))
            # The tensors of this block can be greedy allocated without overuse the SRAM
            # Update the tensor metadata (set the SRAM address of the tensors in the block)
            self.update_tensor_metadata(tensor_info, opids)
            return False

        def update_tensor_metadata(self, tensor_info, opids):
            for tensor_id in tensor_info:
                for tensor_metadata in tensor_info[tensor_id].tensors:
                    if tensor_metadata.cid in opids or tensor_metadata.pid in opids:
                        tensor_metadata.sram_start_address = tensor_info[tensor_id].sram_start_addr
                        tensor_metadata.sram_end_address = tensor_info[tensor_id].sram_end_addr