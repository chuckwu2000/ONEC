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
        # May different to tensor_metadata's start_address and end_address, because the storage space of each tensor metadata is different
        self.start_addr = -1
        self.end_addr = -1
        self.tensors = []

class tensor_metadata:
    def __init__(self, pid, cid):
        # pid: parent opid, cid: child opid
        self.pid = pid
        self.cid = cid
        self.in_DRAM = True
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
                    find_tensor_child = False
                    child_op = self.graph.ops[child_id]
                    for tid in child_op.info['inputs']:
                        if tid == tensor_id:
                            find_tensor_child = True
                            break
                    if not find_tensor_child:
                        continue
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