from collections import defaultdict
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures

class tensor_memory:
    def __init__(self, tensor_id):
        self.tensor_id = tensor_id
        self.size = -1
        self.live_range = defaultdict(int)
        # May different to tensor_metadata's start_address and end_address, because the storage space of each tensor metadata is different
        self.sram_start_addr = -1
        self.sram_end_addr = -1
        self.dram_start_addr = -1
        self.dram_end_addr = -1
        self.tensors = []

class tensor_metadata:
    def __init__(self, pid, cid):
        # pid: parent opid, cid: child opid
        self.pid = pid
        self.cid = cid
        self.in_DRAM = True
        self.sram_start_address = -1
        self.sram_end_address = -1

class memory_allocator:
    def __init__(self, graph):
        self.graph = graph
        self.need_allocate_tensors = defaultdict(tensor_memory)
        self.need_allocate_tensor_ids = []
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

    def dram_allocate(self, tensor_info):
        # Step 1: Fetch those tensors that need to be allocated in DRAM
        need_allocated_tensor_ids = []
        for tensor_id in tensor_info:
            for tensor_metadata in tensor_info[tensor_id].tensors:
                if tensor_metadata.in_DRAM:
                    need_allocated_tensor_ids.append(tensor_id)
                    break
        # Step 2: Greedy by size
        self.greedy_by_size(tensor_info, need_allocated_tensor_ids)

    def greedy_by_size(self, tensor_info, need_allocated_tensor_ids) -> list:
        # Step 1: Initialize memory allocation
        total_used_size = 0
        ordered_allocated_tensors = []

        # Step 2: Greedy by size
        # Step 2.1: Sort tensors by size
        sorted_tensor_ids = sorted(need_allocated_tensor_ids, key=lambda x: tensor_info[x].size, reverse=True)
        # Step 2.2: Allocate memory
        for tensor_id in sorted_tensor_ids:
            prev_start = 0
            best_start = None
            smallest_gap = ArchitectureFeatures.DRAM_MAX_SIZE

            for allocated_tensor in ordered_allocated_tensors:
                max_first_op = max(tensor_info[tensor_id].live_range['first_time_used'], tensor_info[allocated_tensor].live_range['first_time_used'])
                min_last_op = min(tensor_info[tensor_id].live_range['last_time_used'], tensor_info[allocated_tensor].live_range['last_time_used'])
                # Live range has overlap (try to find the gap between this tensor's start address & previous start)
                if max_first_op <= min_last_op:
                    gap = allocated_tensor.dram_start_addr - prev_start
                    if gap >= tensor_info[tensor_id].size and gap < smallest_gap:
                        smallest_gap = gap
                        best_start = prev_start
                    prev_start = max(prev_start, tensor_info[allocated_tensor].dram_end_addr)
                # If no large enough gap can reuse or this tensor's live range overlap with all the allocated tensors,
                # prev_start will start from the highest end address
                if best_start is None:
                    best_start = prev_start
                tensor_info[tensor_id].dram_start_addr = best_start
                tensor_info[tensor_id].dram_end_addr = best_start + tensor_info[tensor_id].size
                total_used_size = max(total_used_size, tensor_info[tensor_id].dram_end_addr)
                # Check total used size whether over the DRAM's size
                if total_used_size > ArchitectureFeatures.DRAM_MAX_SIZE:
                    raise ValueError("The total used size exceeds the DRAM's max size")
                ordered_allocated_tensors.append(tensor_id)
                # Sort the allocated tensors by start address(ascending) and if the start address is the same, sort by size(descending)
                ordered_allocated_tensors = sorted(ordered_allocated_tensors, key=lambda t: (tensor_info[t].dram_start_addr, -tensor_info[t].size))