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

class tensor_memory:
    def __init__(self, tensor_id):
        self.tensor_id = tensor_id
        self.size = -1
        self.memory_storage = Mem_area.DRAM
        self.live_range = defaultdict(dict)
        self.start_address = -1
        self.end_address = -1

class memory_allocator:
    def __init__(self, graph):
        self.graph = graph
        self.need_allocate_tensors = defaultdict(tensor_memory)
        self.need_allocate_tensor_ids = []
        self.allocated_tensors = defaultdict(tensor_memory)
        self.init_all_tensors()

    def init_all_tensors(self):
        operators = self.graph.operators
        for op_info in operators:
            for tensor_id in op_info['inputs'] + op_info['outputs']:
                if tensor_id not in self.need_allocate_tensor_ids:
                    self.need_allocate_tensor_ids.append(tensor_id)
                    self.need_allocate_tensors[tensor_id] = tensor_memory(tensor_id)

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

    def set_cache_storage(self):
        # IN: store the tensor_id of the tensor which store in SRAM before the operation
        # OUT: store the tensor_id of the tensor which store in SRAM after the operation
        # GEN: store the tensor_id of the tensor which will be placed in SRAM after the operation
        # KILL: store the tensor_id of the tensor which will be removed from SRAM after the operation
        # Mem-need = union(IN) + GEN

        # IN(x) = union(OUT(y)) where y is the predecessor of x
        # OUT(x) = (IN(x) - KILL(x)) + GEN(x)
        # GEN(x) = tensor_id of the output tensor or conv/fc's constant tensor(ex: weights, bias)
        # KILL(x) = after the operation, tensor in tensor_wait_consume's value is 0
        sram_max_size_per_split_path = ArchitectureFeatures.SRAM_MAX_SIZE
        tensor_wait_consume = {}
        # Enumerate with in, out, gen, kill
        _in, _out, _gen, _kill, _sram_need = 0, 1, 2, 3, 4
        ops_relation = [[set(), set(), set(), set(), set()] for _ in range(len(self.graph.ops))]
        # Number of input
        _binary, _trinary = 2, 3

        ops = self.graph.ops
        for id, op in enumerate(ops):
            op_info = op.info
            opcode_index = op_info.get("opcode_index")
            opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
            # Check if the operation will be fall back to CPU
            if opcode_type in fall_back_cpu_ops:
                # Update tensor storage
                for tensor_id in op_info['inputs'] + op_info['outputs']:
                    self.need_allocate_tensors[tensor_id].memory_storage = Mem_area.DRAM
                # IN
                for parent_id in op.parents:
                    ops_relation[id][_in] = ops_relation[id][_in].union(ops_relation[parent_id][_out])
                # KILL
                for input_tensor_id in op_info['inputs']:
                    if input_tensor_id in tensor_wait_consume:
                        tensor_wait_consume[input_tensor_id] -= 1
                        if tensor_wait_consume[input_tensor_id] == 0:
                            ops_relation[id][_kill].add(input_tensor_id)
                            tensor_wait_consume.pop(input_tensor_id)
                # GEN
                ops_relation[id][_gen] = set(op_info['outputs'])
                for output_tensor_id in op_info['outputs']:
                    wait_consume = len(self.graph.op_lookup_input[output_tensor_id])
                    if wait_consume > 0:
                        tensor_wait_consume.update({output_tensor_id: wait_consume})
                # OUT
                ops_relation[id][_out] = (ops_relation[id][_in] - ops_relation[id][_kill]) | ops_relation[id][_gen]
                # Clean the tensor that have been consumed, since there may have some tensor have been consumed by the operation in other branch
                need_remove = []
                for tensor_id in ops_relation[id][_out]:
                    if tensor_wait_consume.get(tensor_id, 0) == 0:
                        need_remove.append(tensor_id)
                for tensor_id in need_remove:
                    ops_relation[id][_out].remove(tensor_id)
                # Mem-need
                for tensor_id in ops_relation[id][_in]:
                    if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.SRAM:
                        ops_relation[id][_sram_need].add(tensor_id)
            # Operation that will be compute on NPU
            else:
                # Update tensor storage
                for tensor_id in op_info['outputs']:
                    self.need_allocate_tensors[tensor_id].memory_storage = Mem_area.SRAM
                # IN
                for parent_id in op.parents:
                    ops_relation[id][_in] = ops_relation[id][_in].union(ops_relation[parent_id][_out])
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
                            ops_relation[id][_kill].add(input_tensor_id)
                            tensor_wait_consume.pop(input_tensor_id)
                # GEN
                ops_relation[id][_gen] = set(op_info['outputs'])
                for output_tensor_id in op_info['outputs']:
                    wait_consume = len(self.graph.op_lookup_input.get(output_tensor_id, []))
                    if wait_consume > 0:
                        tensor_wait_consume.update({output_tensor_id: wait_consume})
                if opcode_type in binary_ops:
                    for tensor_id in op_info['inputs'][1: _binary]:
                        # Need to record the constant tensor
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            ops_relation[id][_gen].add(tensor_id)
                elif opcode_type in trinary_ops:
                    for tensor_id in op_info['inputs'][1: _trinary]:
                        # Need to record the constant tensor
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            ops_relation[id][_gen].add(tensor_id)
                # Ex: no bias, then the tensor_id of bias will be -1
                if -1 in ops_relation[id][_gen]:
                    ops_relation[id][_gen].remove(-1)
                # OUT
                ops_relation[id][_out] = (ops_relation[id][_in] - ops_relation[id][_kill]) | ops_relation[id][_gen]
                # Clean the tensor that have been consumed, since there may have some tensor have been consumed by the operation in other branch
                need_remove = []
                for tensor_id in ops_relation[id][_out]:
                    if tensor_wait_consume.get(tensor_id, 0) == 0:
                        need_remove.append(tensor_id)
                for tensor_id in need_remove:
                    ops_relation[id][_out].remove(tensor_id)
                # SRAM need
                for tensor_id in ops_relation[id][_in]:
                    if self.need_allocate_tensors[tensor_id].memory_storage == Mem_area.SRAM:
                        ops_relation[id][_sram_need].add(tensor_id)
                ops_relation[id][_sram_need] = ops_relation[id][_sram_need] | ops_relation[id][_gen]
                # The case that input tensor is store in DRAM, which will be move to SRAM to wait for the operation
                ops_relation[id][_sram_need].add(op_info['inputs'][0])
                if opcode_type in binary_ops:
                    ops_relation[id][_sram_need].add(op_info['inputs'][1])
                elif opcode_type in trinary_ops:
                    ops_relation[id][_sram_need].add(op_info['inputs'][1])
                    ops_relation[id][_sram_need].add(op_info['inputs'][2])
                if -1 in ops_relation[id][_sram_need]:
                    ops_relation[id][_sram_need].remove(-1)

        # for cascade_ops in self.graph.cascade_matched_ops:
        #     sram_need = set()
        #     for op in cascade_ops:
        #         sram_need = sram_need | ops_relation[op][_sram_need]
        #     sram_usage = self.compute_sram_usage(sram_need)
        #     if sram_usage > sram_max_size_per_split_path:
        #         print("="*50)
        #         print(f"size exceed at cascade_ops: {cascade_ops}")
        #         print(f"sram_need: {sram_need}")
        #         print(f"sram_usage: {sram_usage}")
                    
        for id, op in enumerate(ops):
            # Check if the size of tensors exceeds the SRAM's max size
            sram_usage = self.compute_sram_usage(ops_relation[id][_sram_need])
            if sram_usage > sram_max_size_per_split_path:
                print("="*50)
                op_info = op.info
                print(f"size exceed at #{id} op: {op_info}")
                print(f"sram_need: {ops_relation[id][_sram_need]}")
                print(f"in: {ops_relation[id][_in]}")
                print(f"out: {ops_relation[id][_out]}")
                print(f"gen: {ops_relation[id][_gen]}")
                print(f"kill: {ops_relation[id][_kill]}")
                ops_relation[id][_out] = set()

    def compute_sram_usage(self, mem_need):
        mem_need_size = 0
        for tensor_id in mem_need:
            mem_need_size += self.need_allocate_tensors[tensor_id].size
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