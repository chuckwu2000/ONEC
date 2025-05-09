from Architecture_feature import ArchitectureFeatures
from OpClassify import Op_Classify
from collections import defaultdict

op_classify = Op_Classify()
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops
fall_back_cpu_ops = op_classify.fall_back_cpu_ops
reduce_ops = op_classify.reduce_ops
unary_ops = op_classify.unary_ops
binary_ops = op_classify.binary_ops
trinary_ops = op_classify.trinary_ops

# At different time stamps, the tensor may allocate on different SRAMs
class Distributed_SRAM_tensor_info:
    def __init__(self, pid, cid):
        self.pid = pid
        self.cid = cid
        self.live_range = defaultdict(int)
        self.sram_idx = -1

class Distributed_SRAM_tensor:
    def __init__(self):
        self.live_range = defaultdict(int)
        self.tensors = []

class Distributed_SRAM_allocator:
    def __init__(self, graph):
        self.graph = graph
        self.total_SRAMs = 8
        self.tensor_info = defaultdict(Distributed_SRAM_tensor)
        self.need_allocate_tensor_ids = []
        self.init_all_tensors()
        self.set_tensor_live_range()

    def init_all_tensors(self):
        ordered_ops = self.graph.ordered_ops
        for op in ordered_ops:
            for tensor_id in op.info['inputs']:
                if tensor_id not in self.need_allocate_tensor_ids:
                    self.need_allocate_tensor_ids.append(tensor_id)
                    self.tensor_info[tensor_id] = Distributed_SRAM_tensor()
                buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                if len(buffer) != 0:
                    # Constant tensor
                    self.tensor_info[tensor_id].tensors.append(Distributed_SRAM_tensor_info(op.opid, op.opid))
                else:
                    # Non-constant tensor
                    for parent_id in op.parents:
                        find_tensor_parent = False
                        parent_op = self.graph.ops[parent_id]
                        for tid in parent_op.info['outputs']:
                            if tid == tensor_id:
                                find_tensor_parent = True
                                break
                        if not find_tensor_parent:
                            continue
                        tensor_info_allocated = False
                        for tensor in self.tensor_info[tensor_id].tensors:
                            if tensor.cid == op.opid and tensor.pid == parent_id:
                                tensor_info_allocated = True
                                break
                        if not tensor_info_allocated:
                            self.tensor_info[tensor_id].tensors.append(Distributed_SRAM_tensor_info(parent_id, op.opid))
            for tensor_id in op.info['outputs']:
                if tensor_id not in self.need_allocate_tensor_ids:
                    self.need_allocate_tensor_ids.append(tensor_id)
                    self.tensor_info[tensor_id] = Distributed_SRAM_tensor()
                for child_id in op.children:
                    find_tensor_child = False
                    child_op = self.graph.ops[child_id]
                    for tid in child_op.info['inputs']:
                        if tid == tensor_id:
                            find_tensor_child = True
                            break
                    if not find_tensor_child:
                        continue
                    self.tensor_info[tensor_id].tensors.append(Distributed_SRAM_tensor_info(op.opid, child_id))
        # In here, we don't check if the tensor overuse the DRAM (since OEM's NPU not connect to DRAM now)
        # TODO: If NPU connect DRAM in the future, can reference Memory_allocation.py

    def set_tensor_live_range(self):
        ordered_ops = self.graph.ordered_ops
        for order, op in enumerate(ordered_ops):
            opcode_index = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            if opcode_type in elem_wise_ops or opcode_type in mac_ops:
                for tensor_id in op.info['inputs'] + op.info['outputs']:
                    if tensor_id == -1:
                        continue
                    # Update the tensor's first time used
                    if order < self.tensor_info[tensor_id].live_range.get('first_time_used', len(ordered_ops)):
                        self.tensor_info[tensor_id].live_range['first_time_used'] = order
                    # Update the tensor's last time used
                    if order > self.tensor_info[tensor_id].live_range.get('last_time_used', -1):
                        self.tensor_info[tensor_id].live_range['last_time_used'] = order
        # Update the tensor's live range
        for tensor in self.tensor_info.values():
            # If the tensor's last time used is 0, it means the tensor is not used in the graph
            if tensor.live_range['last_time_used'] == 0:
                continue
            # Update the live range of the tensor
            for tensor_info in tensor.tensors:
                tensor_info.live_range['first_time_used'] = tensor.live_range['first_time_used']
                tensor_info.live_range['last_time_used'] = tensor.live_range['last_time_used']

    def allocate_tensors(self):
        cascade_patterns = self.find_cascade_pattern()
        visited = [False for _ in range(len(self.graph.ordered_ops))]
        for op in self.ordered_ops:
            if visited[op.opid]:
                continue
            find_in_cascade = False
            for cascade_pattern in cascade_patterns:
                # Launch concurrent run pattern
                if op.opid == cascade_pattern[0]:
                    operator_ids = [ids for ids in cascade_pattern]
                    # OEM's NPU support only 4 operators in concurrent run
                    while len(operator_ids) > 4:
                        operator_ids.pop()
                    operators = []
                    for opid in operator_ids:
                        operators.append(self.graph.ops[opid])
                    while(not self.allocate_success(operators)):
                        # If the allocation is not successful, try to decrease the number of operators
                        operator_ids.pop()
                        operators.pop()
                    # Update the visited list
                    for opid in operator_ids:
                        visited[opid] = True
                    find_in_cascade = True
                    break
            # Launch single operator
            if not find_in_cascade:
                operator = self.graph.ops[op.opid]
                operators = [operator]
                if not self.allocate_success(operators):
                    raise ValueError("The allocation of the operator is not successful")
                visited[op.opid] = True

    def find_cascade_pattern(self):
        cascade_matched_ops_list = []
        have_matched = [False for _ in range(len(self.graph.ordered_ops))]
        for order, op in enumerate(self.graph.ordered_ops):
            if have_matched[order]:
                continue
            opcode_idx = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_idx].get('builtin_code')
            # Our cascade pattern always start from the mac-main op
            if opcode_type in mac_ops:
                cascade_matched_ops = [op.opid]
                now_order = order
                next_order = order + 1
                while next_order < len(self.graph.ordered_ops):
                    now_op = self.graph.ordered_ops[now_order]
                    next_op = self.graph.ordered_ops[next_order]
                    opcode_index = next_op.info.get('opcode_index')
                    opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

                    # If the child op need to perform reduce operation, we can't let it directly consume the output of mac-main-op
                    if opcode_type in reduce_ops:
                        break

                    # Check whether have producer-consumer relationship
                    have_producer_consumer = False
                    for child in now_op.children:
                        if child == next_op.opid:
                            have_producer_consumer = True
                            break
                    if not have_producer_consumer:
                        break

                    if opcode_type == "RESHAPE":
                        now_order = next_order
                        next_order += 1
                        continue

                    # Check if the next operator is a mac-main op
                    if opcode_type in elem_wise_ops and len(cascade_matched_ops) < 4:
                        cascade_matched_ops.append(next_op.opid)
                        have_matched[next_order] = True
                        now_order = next_order
                        next_order += 1
                    else:
                        break
                # Add the cascade pattern to the list
                if len(cascade_matched_ops) > 1:
                    cascade_matched_ops_list.append(cascade_matched_ops)
            else:
                have_matched[order] = True
        return cascade_matched_ops_list
        
    def allocate_success(self, operators):
        class ColoringGraph:
            def __init__(self, colors):
                self.colors = colors
                self.edges = []
                self.nodes = []

        # The color we can allocate to the tensor (False means color not used)
        colors = [False for _ in range(self.total_SRAMs)]
        # Collect all need allocate tensors
        candidate_tensors = []
        total_operators = len(operators)
        for op_idx, op in enumerate(operators):
            next_op = operators[op_idx + 1] if op_idx + 1 < total_operators else None
            # Collect input tensors
            opcode_index = op.info['opcode_index']
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            if opcode_type in unary_ops:
                input_need = 0
            elif opcode_type in binary_ops:
                input_need = 1
            elif opcode_type in trinary_ops:
                input_need = 2
            for tensor_id in op.info['inputs'][0: input_need]:
                if tensor_id == -1:
                    continue
                # Based on pid & cid to find the tensor
                for tensor in self.tensor_info[tensor_id].tensors:
                    if tensor.cid == op.opid:
                        # Check the tensor is already allocated
                        if tensor.sram_idx != -1:
                            # If the tensor is already allocated, we use it directly and mark the color
                            colors[tensor.sram_idx] = True
                        else:
                            candidate_tensors.append(tensor_id)
                        break
            # Collect output tensors
            for tensor_id in op.info['outputs']:
                for tensor in self.tensor_info[tensor_id].tensors:
                    # Won't break as early as possible, since same tensor may be used by different ops
                    if tensor.pid == op.opid:
                        # OEM's NPU can flow output data into buffer(no need to store back to SRAM), then next op can consume the output tensor from buffer
                        # If the next op won't consume the output tensor immediately, we need to store the output tensor back to SRAM
                        if next_op != None and tensor.cid != next_op.opid:
                            candidate_tensors.append(tensor_id)
                
        # Check if the operators' tensors can be allocated in the SRAM
        # TODO: build graph...
        coloring_graph = ColoringGraph(colors)
        return True