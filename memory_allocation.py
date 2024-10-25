from collections import defaultdict

class tenosr_memory:
    def __init__(self, tensor_id, size):
        self.tensor_id = tensor_id
        self.size = size
        self.start_address = -1
        self.end_address = -1

class memory_allocator:
    def __init__(self, model):
        self.model = model
        self.SRAM_MAX_SIZE = 1 << 63
        self.need_allocate_tensor_id = set()
        self.live_range = defaultdict(dict)
        self.allocated_tensor = self.memory_allocate()

    def compute_tensor_live_range(self):
        operators = self.model['subgraphs'][0]['operators']

        # Record each input/output tensor's first and last time used
        for opid, op_info in enumerate(operators):
            for tensor_id in op_info['inputs'] + op_info['outputs']:
                self.need_allocate_tensor_id.add(tensor_id)
                # Update this tensor's first time used
                if opid < self.live_range[tensor_id].get('first_time_used', len(operators)):
                    self.live_range[tensor_id]['first_time_used'] = opid
                # Update this tensor's last time used
                if opid > self.live_range[tensor_id].get('last_time_used', -1):
                    self.live_range[tensor_id]['last_time_used'] = opid

    def compute_tensor_size(self, tensor_id) -> int:
        tensor = self.model['subgraphs'][0]['tensors'][tensor_id]
        shape = tensor['shape']
        size = 1
        for dim in shape:
            size *= dim
        return size

    def greedy_by_size(self) -> list:
        # Step 1: Sort tensors by size
        tensor_size = {}
        for tensor_id in self.need_allocate_tensor_id:
            tensor_size[tensor_id] = self.compute_tensor_size(tensor_id)
        sorted_tensor_id = sorted(tensor_size, key=lambda x: tensor_size[x], reverse=True)

        # Step 2: Initialize memory allocation
        need_allocate_tensors = []
        for tensor_id in sorted_tensor_id:
            need_allocate_tensors.append(tenosr_memory(tensor_id, tensor_size[tensor_id]))
        total_consumed_size = 0
        size_non_inc_allocated_tensors = []
        allocated_tensors = defaultdict(tenosr_memory)

        # Step 3: Allocate memory (greedy by size)
        for tensor in need_allocate_tensors:
            prev_start = 0
            best_start = None
            smallest_gap = self.SRAM_MAX_SIZE

            for allocated_tensor in size_non_inc_allocated_tensors:
                max_first_op = max(self.live_range[tensor.tensor_id]['first_time_used'], self.live_range[allocated_tensor.tensor_id]['first_time_used'])
                min_last_op = min(self.live_range[tensor.tensor_id]['last_time_used'], self.live_range[allocated_tensor.tensor_id]['last_time_used'])
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
            allocated_tensors[tensor.tensor_id] = tensor

        return allocated_tensors

    def memory_allocate(self):
        self.compute_tensor_live_range()
        allocated_tensor = self.greedy_by_size()
        return allocated_tensor