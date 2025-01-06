from collections import defaultdict
from Architecture_feature import Mem_area

class tensor_memory:
    def __init__(self, tensor_id):
        self.tensor_id = tensor_id
        self.size = -1
        self.memory_storage = Mem_area.DRAM
        self.live_range = defaultdict(dict)
        self.start_address = -1
        self.end_address = -1

class memory_allocator:
    def __init__(self, graph, use_sram):
        self.graph = graph
        # DRAM's max size is 4GB
        self.DRAM_MAX_SIZE = 1 << 32
        # SRAM's max size is 2MB
        self.SRAM_MAX_SIZE = 1 << 21
        self.need_allocate_tensors = defaultdict(tensor_memory)
        self.need_allocate_tensor_ids = []
        self.allocated_tensors = defaultdict(tensor_memory)
        self.memory_allocate(use_sram)

    def memory_allocate(self, use_sram):
        self.init_all_tensors()
        # Perform DRAM's mem allocation
        allocated_tensor = self.greedy_by_size()
        # Perform SRAM's mem allocation
        # TODO: SRAM's mem allocation
        if use_sram:
            pass
        return allocated_tensor

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
        if total_size > self.DRAM_MAX_SIZE:
            raise ValueError("The total size of tensors exceeds the DRAM's max size")

        # Compute tensor's live range
        self.compute_tensor_live_range()

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
            smallest_gap = self.DRAM_MAX_SIZE

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
        return self.allocated_tensors