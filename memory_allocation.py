from collections import defaultdict
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures

# The operation that will be fall back to CPU (not contain RESHAPE)
data_layout_ops = ["CONCATENATION", "SPLIT", "SPLIT_V", "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR", "PACK"]
reduce_ops = ["REDUCE_MAX"]
fall_back_cpu_ops = data_layout_ops + reduce_ops

# The input of the operation
binary_ops = ["ADD", "SUB", "MUL", "SQUARED_DIFFERENCE", "BATCH_MATMUL"]
trinary_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]
weight_reuse_ops = trinary_ops

class tensor_memory:
    def __init__(self, tensor_id):
        self.tensor_id = tensor_id
        self.size = -1
        self.memory_storage = Mem_area.DRAM
        self.live_range = defaultdict(dict)
        self.tensors = []

class tensor_metadata:
    def __init__(self, pid, cid):
        # pid: parent opid, cid: child opid
        self.pid = pid
        self.cid = cid
        self.memory_storage = Mem_area.DRAM
        self.start_address = -1
        self.end_address = -1

class memory_allocator:
    def __init__(self, graph):
        self.graph = graph
        self.need_allocate_tensors = defaultdict(tensor_memory)
        self.need_allocate_tensor_ids = []
        self.allocated_tensors = defaultdict(tensor_memory)
        self.init_all_tensors()
        self.ops_relation = [[set(), set(), set(), set(), set()] for _ in range(len(self.graph.ops))]
        self.visited = [False for _ in range(len(self.graph.ops))]
        self.weights_reuse_order = 0
        self.weights_reuse_mapping = defaultdict(int)

    def init_all_tensors(self):
        ops = self.graph.ordered_ops
        for op in ops:
            for tensor_id in op.info['inputs']:
                # Add the new tensor_memory to the need_allocate_tensors
                if tensor_id not in self.need_allocate_tensor_ids:
                    self.need_allocate_tensor_ids.append(tensor_id)
                    self.need_allocate_tensors[tensor_id] = tensor_memory(tensor_id)
                # Add the new tensor_metadata to the tensor_memory's tensors
                buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                if len(buffer) != 0:
                    # Constant tensor
                    self.need_allocate_tensors[tensor_id].tensors.append(tensor_metadata(op.opid, op.opid))
                else:
                    # Non-constant tensor
                    for parent_id in op.parents:
                        # Check if the parent op is the tensor's parent
                        find_tensor_parent = False
                        parent_op = self.graph.ops[parent_id]
                        for tid in parent_op.info['outputs']:
                            if tid == tensor_id:
                                find_tensor_parent = True
                                break
                        if not find_tensor_parent:
                            continue
                        # Check if the tensor_metadata is already allocated
                        tensor_metadata_allocated = False
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.cid == op.opid and tensor.pid == parent_id:
                                tensor_metadata_allocated = True
                                break
                        if not tensor_metadata_allocated:
                            self.need_allocate_tensors[tensor_id].tensors.append(tensor_metadata(parent_id, op.opid))
            for tensor_id in op.info['outputs']:
                # Add the new tensor_memory to the need_allocate_tensors
                if tensor_id not in self.need_allocate_tensor_ids:
                    self.need_allocate_tensor_ids.append(tensor_id)
                    self.need_allocate_tensors[tensor_id] = tensor_memory(tensor_id)
                # Add the new tensor_metadata to the tensor_memory's tensors
                for child_id in op.children:
                    # It seems no need to check duplicate tensor_metadata in output (maybe is the traversal order we pick)
                    self.need_allocate_tensors[tensor_id].tensors.append(tensor_metadata(op.opid, child_id))

        # Compute tensor size
        total_size = 0
        for tensor_id in self.need_allocate_tensor_ids:
            self.need_allocate_tensors[tensor_id].size = self.compute_tensor_size(tensor_id)
            total_size += self.need_allocate_tensors[tensor_id].size

        # Check if the size of tensors exceeds the DRAM's max size
        if total_size > ArchitectureFeatures.DRAM_MAX_SIZE:
            raise ValueError("The total size of tensors exceeds the DRAM's max size")
        
    def compute_tensor_size(self, tensor_id) -> int:
        tensor = self.graph.tensors[tensor_id]
        shape = tensor['shape']
        if tensor.get("type") == "INT8":
            elem_size = 8
        else:
            elem_size = 32
        element = 1
        for dim in shape:
            element *= dim
        size = element * (elem_size // 8)
        return size

    def weight_reuse_schedule_and_set_cache_storage(self, same_layer_next_opids):
        # IN: store the tensor_id of the tensor which store in SRAM before the operation
        # OUT: store the tensor_id of the tensor which store in SRAM after the operation
        # GEN: store the tensor_id of the tensor which will be placed in SRAM after the operation
        # KILL: store the tensor_id of the tensor which will be removed from SRAM after the operation
        # Mem-need = union(IN) + GEN

        # IN(x) = union(OUT(y)) where y is the predecessor of x
        # OUT(x) = (IN(x) - KILL(x)) + GEN(x)
        # GEN(x) = tensor_id of the output tensor or conv/fc's constant tensor(ex: weights, bias)
        # KILL(x) = after the operation, tensor in tensor_wait_consume's value is 0
        tensor_wait_consume = {}
        # Enumerate with in, out, gen, kill
        _in, _out, _gen, _kill, _sram_need = 0, 1, 2, 3, 4
        # Number of input
        _trinary = 3

        weights_in_sram = set()
        block_start_order = 0
        ordered_ops = self.graph.ordered_ops
        for order, op in enumerate(ordered_ops):
            if self.visited[op.opid]:
                continue
            self.visited[op.opid] = True
            op_info = op.info
            id = op.opid
            opcode_index = op_info.get("opcode_index")
            opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
            # Check if the operation will be fall back to CPU
            if opcode_type in fall_back_cpu_ops:
                # Update tensor storage
                for tensor_id in op_info['inputs']:
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if tensor.cid == id:
                            tensor.memory_storage = Mem_area.DRAM
                for tensor_id in op_info['outputs']:
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if tensor.pid == id:
                            tensor.memory_storage = Mem_area.DRAM
                # IN
                for parent_id in op.parents:
                    self.ops_relation[id][_in] = self.ops_relation[id][_in].union(self.ops_relation[parent_id][_out])
                # KILL
                for input_tensor_id in op_info['inputs']:
                    if input_tensor_id in tensor_wait_consume:
                        tensor_wait_consume[input_tensor_id] -= 1
                        if tensor_wait_consume[input_tensor_id] == 0:
                            self.ops_relation[id][_kill].add(input_tensor_id)
                            tensor_wait_consume.pop(input_tensor_id)
                # GEN
                self.ops_relation[id][_gen] = set(op_info['outputs'])
                for output_tensor_id in op_info['outputs']:
                    wait_consume = len(self.graph.op_lookup_input[output_tensor_id])
                    if wait_consume > 0:
                        tensor_wait_consume.update({output_tensor_id: wait_consume})
                # OUT
                self.ops_relation[id][_out] = (self.ops_relation[id][_in] - self.ops_relation[id][_kill]) | self.ops_relation[id][_gen]
                # Clean the tensor that have been consumed, since there may have some tensor have been consumed by the operation in other branch
                need_remove = []
                for tensor_id in self.ops_relation[id][_out]:
                    if tensor_wait_consume.get(tensor_id, 0) == 0:
                        need_remove.append(tensor_id)
                for tensor_id in need_remove:
                    self.ops_relation[id][_out].remove(tensor_id)
                # Mem-need
                for tensor_id in self.ops_relation[id][_in]:
                    if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.SRAM:
                        self.ops_relation[id][_sram_need].add(tensor_id)
            # Operation that will be compute on NPU
            else:
                # Update tensor storage
                for tensor_id in op_info['outputs']:
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if tensor.pid == id:
                            tensor.memory_storage = Mem_area.SRAM
                    self.need_allocate_tensors[tensor_id].memory_storage = Mem_area.SRAM
                # Handle the memory storage of the constant tensor
                if opcode_type in weight_reuse_ops:
                    for tensor_id in op_info['inputs'][1: _trinary]:
                        # Check whether is the first time to fetch the constant tensor from DRAM
                        if tensor_id not in weights_in_sram:
                            for tensor in self.need_allocate_tensors[tensor_id].tensors:
                                if tensor.cid == id:
                                    tensor.memory_storage = Mem_area.DRAM
                                    break
                            continue
                        for tensor in self.need_allocate_tensors[tensor_id].tensors:
                            if tensor.cid == id:
                                tensor.memory_storage = Mem_area.SRAM
                        self.need_allocate_tensors[tensor_id].memory_storage = Mem_area.SRAM
                # IN
                for parent_id in op.parents:
                    self.ops_relation[id][_in] = self.ops_relation[id][_in].union(self.ops_relation[parent_id][_out])
                # KILL
                # Tensor splitting may introduces some custom operations, which some boundary tensor will append to the input tensor
                start_id = 0
                if opcode_type in trinary_ops:
                    if len(op_info['inputs']) > 3:
                        start_id = 1
                for input_tensor_id in op_info['inputs'][start_id:]:
                    if input_tensor_id in tensor_wait_consume:
                        tensor_wait_consume[input_tensor_id] -= 1
                        if tensor_wait_consume[input_tensor_id] == 0:
                            self.ops_relation[id][_kill].add(input_tensor_id)
                            tensor_wait_consume.pop(input_tensor_id)
                # GEN
                self.ops_relation[id][_gen] = set(op_info['outputs'])
                for output_tensor_id in op_info['outputs']:
                    wait_consume = len(self.graph.op_lookup_input.get(output_tensor_id, []))
                    if wait_consume > 0:
                        tensor_wait_consume.update({output_tensor_id: wait_consume})
                if opcode_type in binary_ops:
                    for tensor_id in op_info['inputs']:
                        # Need to record the constant tensor
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            self.ops_relation[id][_gen].add(tensor_id)
                elif opcode_type in weight_reuse_ops:
                    for tensor_id in op_info['inputs'][1: _trinary]:
                        # Need to record the constant tensor
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            self.ops_relation[id][_gen].add(tensor_id)
                            # The weights or bias that we can reuse on other split path, set the tensor_wait_consume to the max value
                            tensor_wait_consume.update({tensor_id: 2147483647})
                            weights_in_sram.add(tensor_id)
                # Ex: no bias, then the tensor_id of bias will be -1
                if -1 in self.ops_relation[id][_gen]:
                    self.ops_relation[id][_gen].remove(-1)
                # OUT
                self.ops_relation[id][_out] = (self.ops_relation[id][_in] - self.ops_relation[id][_kill]) | self.ops_relation[id][_gen]
                # Clean the tensor that have been consumed, since there may have some tensor have been consumed by the operation in other branch
                need_remove = []
                for tensor_id in self.ops_relation[id][_out]:
                    if tensor_wait_consume.get(tensor_id, 0) == 0:
                        need_remove.append(tensor_id)
                for tensor_id in need_remove:
                    self.ops_relation[id][_out].remove(tensor_id)
                # SRAM need
                for tensor_id in self.ops_relation[id][_in]:
                    if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.SRAM:
                        self.ops_relation[id][_sram_need].add(tensor_id)
                self.ops_relation[id][_sram_need] = self.ops_relation[id][_sram_need] | self.ops_relation[id][_gen]
                # The case that input tensor is store in DRAM, which will be move to SRAM to wait for the operation
                self.ops_relation[id][_sram_need].add(op_info['inputs'][0])
                if opcode_type in binary_ops:
                    self.ops_relation[id][_sram_need].add(op_info['inputs'][1])
                elif opcode_type in trinary_ops:
                    self.ops_relation[id][_sram_need].add(op_info['inputs'][1])
                    self.ops_relation[id][_sram_need].add(op_info['inputs'][2])
                if -1 in self.ops_relation[id][_sram_need]:
                    self.ops_relation[id][_sram_need].remove(-1)
            # Check whether the block depth should stop in this op
            (over_use, stop_id) = self.check_depth(id)
            if over_use:
                # Need to reschedule, but in here we just record the mapping of the opid and the weight reuse order
                block_end_order = self.graph.ops[stop_id].schedule_order
                self.weight_reuse_schedule(block_start_order, block_end_order, same_layer_next_opids)
                block_start_order = order
                self.weights_reuse_mapping[id] = self.weights_reuse_order
                self.weights_reuse_order += 1
                # Set the wait_consume's tensor's tensor metadata's storage to DRAM
                for tensor_id in tensor_wait_consume:
                    # Check whether the tensor's visited[cid] is False, which means it didn't be consumed by the operation
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if self.visited[tensor.cid] == False:
                            tensor.memory_storage = Mem_area.DRAM
                # Set the stop_id's output tensor(also is id's input tensor)'s tensor metadata's storage to DRAM
                # First, collect same_layer_next_opids from the id's head
                same_layer_id_opids = []
                head_id = same_layer_next_opids.get(id, (-1, -1))[0]
                tmp_id = head_id
                while tmp_id != -1:
                    same_layer_id_opids.append(tmp_id)
                    tmp_id = same_layer_next_opids.get(tmp_id, (-1, -1))[1]
                for tensor_id in self.graph.ops[stop_id].info['outputs']:
                    for tensor in self.need_allocate_tensors[tensor_id].tensors:
                        if tensor.cid in same_layer_id_opids:
                            tensor.memory_storage = Mem_area.DRAM
                # Clean the weights in SRAM & update the out relation
                for tensor_id in weights_in_sram:
                    tensor_wait_consume.pop(tensor_id)
                    for id in same_layer_id_opids:
                        if tensor_id in self.ops_relation[id][_out]:
                            self.ops_relation[id][_out].remove(tensor_id)
                weights_in_sram.clear()
            else:
                self.weights_reuse_mapping[id] = self.weights_reuse_order
                self.weights_reuse_order += 1
        return self.weights_reuse_mapping

    def check_depth(self, id) -> tuple:
        # Decide the depth of the weight reuse
        _sram_need = 4
        sram_max_size_per_split_path = ArchitectureFeatures.SRAM_MAX_SIZE
        ops = self.graph.ops
        ops_relation = self.ops_relation
        # Check if the size of tensors exceeds the SRAM's max size
        sram_usage = self.compute_sram_usage(ops_relation[id][_sram_need])
        if sram_usage > sram_max_size_per_split_path:
            print(f"overuse in op_info: {ops[id].info}")
            print(f"_sram_need: {ops_relation[id][4]}")
            # Stop in ops[id]'s last parent, then the exec-order will become BF
            schedule_order = -1
            stop_id = -1
            for parent_id in ops[id].parents:
                # Choose the last order of the parent
                if ops[parent_id].schedule_order > schedule_order:
                    schedule_order = ops[parent_id].schedule_order
                    stop_id = parent_id
            return (True, stop_id)
        return (False, -1)
    
    def weight_reuse_schedule(self, block_start_order, block_end_order, same_layer_next_opids):
        ops_on_next_path = []
        for op in self.graph.ordered_ops[block_start_order: block_end_order + 1]:
            next_opid = same_layer_next_opids.get(op.opid, (-1, -1))[1]
            if next_opid != -1:
                ops_on_next_path.append(next_opid)
        while len(ops_on_next_path) > 0:
            new_list = []
            for now_opid in ops_on_next_path:
                now_op = self.graph.ops[now_opid]
                next_opid = same_layer_next_opids.get(now_op.opid, (-1, -1))[1]
                if next_opid != -1:
                    new_list.append(next_opid)
                if self.visited[now_opid]:
                    continue
                self.visited[now_opid] = True
                self.weights_reuse_mapping[now_opid] = self.weights_reuse_order
                self.weights_reuse_order += 1
            ops_on_next_path = new_list
    
    def compute_sram_usage(self, mem_need):
        need_allocate_tensors = self.need_allocate_tensors
        mem_need_size = 0
        for tensor_id in mem_need:
            mem_need_size += need_allocate_tensors[tensor_id].size
        return mem_need_size

    def memory_allocate(self, use_sram = False):
        # Compute tensor's live range
        self.compute_tensor_live_range()
            
        # Perform DRAM's mem allocation
        self.dram_allocate()
        # Perform SRAM's mem allocation
        if use_sram:
            # TODO: SRAM's mem allocation
            self.sram_allocate()
        return self.allocated_tensors

    def compute_tensor_live_range(self):
        operators = self.graph.operators

        # Record each input/output tensor's first and last time used
        for opid, op_info in enumerate(operators):
            for tensor_id in op_info['inputs'] + op_info['outputs']:
                # Update this tensor's first time used
                if opid < self.need_allocate_tensors[tensor_id].live_range.get('first_time_used', len(operators)):
                    self.need_allocate_tensors[tensor_id].live_range['first_time_used'] = opid
                # Update this tensor's last time used
                if opid > self.need_allocate_tensors[tensor_id].live_range.get('last_time_used', -1):
                    self.need_allocate_tensors[tensor_id].live_range['last_time_used'] = opid

    def dram_allocate(self):
        self.greedy_by_size()

    def greedy_by_size(self) -> list:
        # Step 1: Sort tensors by size
        tensor_storage_in_DRAM = []
        for tensor_id in self.need_allocate_tensor_ids:
            if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.DRAM:
                tensor_storage_in_DRAM.append(self.need_allocate_tensors[tensor_id])
        sorted_tensor = sorted(tensor_storage_in_DRAM, key=lambda x: x.size, reverse=True)

        # Step 2: Initialize memory allocation
        total_consumed_size = 0
        size_non_inc_allocated_tensors = []
        
        # Step 3: Allocate memory (greedy by size)
        for tensor in sorted_tensor:
            prev_start = 0
            best_start = None
            smallest_gap = ArchitectureFeatures.DRAM_MAX_SIZE

            for allocated_tensor in size_non_inc_allocated_tensors:
                max_first_op = max(tensor.live_range['first_time_used'], allocated_tensor.live_range['first_time_used'])
                min_last_op = min(tensor.live_range['last_time_used'], allocated_tensor.live_range['last_time_used'])
                if max_first_op <= min_last_op:
                    gap = allocated_tensor.start_address - prev_start
                    if gap >= tensor.size and gap < smallest_gap:
                        smallest_gap = gap
                        best_start = prev_start
                    prev_start = max(prev_start, allocated_tensor.end_address)
            if best_start is None:
                best_start = prev_start
            tensor.start_address = best_start
            tensor.end_address = best_start + tensor.size
            total_consumed_size = max(total_consumed_size, tensor.end_address)
            
            size_non_inc_allocated_tensors.append(tensor)
            self.allocated_tensors[tensor.tensor_id] = tensor
    
    def sram_allocate(self):
        # Step 1:
        for tensor_id in self.need_allocate_tensor_ids:
            if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.SRAM:
                tensor = self.need_allocate_tensors[tensor_id]
                self.allocated_tensors[tensor.tensor_id] = tensor