from OpClassify import Op_Classify
from collections import defaultdict

op_classify = Op_Classify()
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops
data_layout_ops = op_classify.data_layout_ops
reduce_ops = op_classify.reduce_ops
unary_ops = op_classify.unary_ops
binary_ops = op_classify.binary_ops
trinary_ops = op_classify.trinary_ops

# At different timestamps, the tensor may be allocated to different SRAMs
class Distributed_SRAM_tensor_info:
    def __init__(self, pid, cid):
        self.pid = pid
        self.cid = cid
        self.live_range = defaultdict(int)
        self.sram_id = -1
        self.dram_access = True

class Distributed_SRAM_tensor:
    def __init__(self):
        self.live_range = defaultdict(int)
        self.tensors = []

class Distributed_SRAM_allocator:
    def __init__(self, graph):
        self.graph = graph
        self.ordered_ops = graph.ordered_ops
        self.total_SRAMs = 8
        self.tensor_info = defaultdict(Distributed_SRAM_tensor)
        self.non_allocated_cascade_patterns = []
        self.cascade_patterns = []
        self.need_allocate_tensor_ids = []
        self.init_all_tensors()
        self.set_tensor_live_range()

    def init_all_tensors(self):
        for op in self.ordered_ops:
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
                if len(op.children) == 0:
                    self.tensor_info[tensor_id].tensors.append(Distributed_SRAM_tensor_info(op.opid, op.opid))
        # For codegen convenience, we sort the tensor's info by their cid
        for tensor_id in self.need_allocate_tensor_ids:
            self.tensor_info[tensor_id].tensors.sort(key=lambda tensor: self.graph.ops[tensor.cid].schedule_order)
        # In here, we don't check if the tensor overuse the DRAM (since OEM's NPU not connect to DRAM now)
        # TODO: If NPU connect DRAM in the future, can reference Memory_allocation.py

    def set_tensor_live_range(self):
        for order, op in enumerate(self.ordered_ops):
            opcode_index = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            if opcode_type in elem_wise_ops or opcode_type in mac_ops:
                for tensor_id in op.info['inputs'] + op.info['outputs']:
                    if tensor_id == -1:
                        continue
                    # Update the tensor's first time used
                    if order < self.tensor_info[tensor_id].live_range.get('first_time_used', len(self.ordered_ops)):
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
        self.non_allocated_cascade_patterns = self.find_cascade_pattern()
        self.update_reshape_related_tensors()
        visited = [False for _ in range(len(self.graph.ordered_ops))]
        pattern_output_tensors = []
        for op in self.ordered_ops:
            if visited[op.opid]:
                continue
            # Update the last pattern's utput tensors
            last_pattern_output_tensors = pattern_output_tensors
            # Check whether the op is the start of the cascade pattern
            find_in_cascade = False
            for cascade_pattern in list(self.non_allocated_cascade_patterns):
                # Allocate concurrent run pattern
                if op.opid == cascade_pattern[0]:
                    operator_ids = [ids for ids in cascade_pattern]
                    operators = []
                    for opid in operator_ids:
                        operators.append(self.graph.ops[opid])
                    while(True):
                        allocate_success, pattern_output_tensors = self.allocate_success(operators, last_pattern_output_tensors)
                        if allocate_success:
                            break
                        # If the allocation is not successful, try to decrease the number of operators
                        operator_ids.pop()
                        operators.pop()
                    self.cascade_patterns.append(operator_ids)
                    # Update the visited list
                    for opid in operator_ids:
                        visited[opid] = True
                    find_in_cascade = True
                    break
            # Launch single operator
            if not find_in_cascade:
                # The fall back cpu op won't participate in the SRAM allocation
                opcode_idx = op.info.get('opcode_index')
                opcode_type = self.graph.opcodes[opcode_idx].get('builtin_code')
                if opcode_type in data_layout_ops:
                    continue

                operator = self.graph.ops[op.opid]
                operators = [operator]
                allocate_success, pattern_output_tensors = self.allocate_success(operators, last_pattern_output_tensors)
                if not allocate_success:
                    raise ValueError("The allocation of the single operator is not successful\n" + f"op info: {op.info}")
                visited[op.opid] = True
        return self.tensor_info

    def find_cascade_pattern(self):
        cascade_matched_ops_list = []
        # Based on the opcode type (element-wise ops)
        used_units = []
        have_matched = [False for _ in range(len(self.graph.ordered_ops))]
        for order, op in enumerate(self.graph.ordered_ops):
            if have_matched[order]:
                continue
            opcode_idx = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_idx].get('builtin_code')
            # Our cascade pattern always start from the mac-main op
            if opcode_type in mac_ops:
                cascade_matched_ops = [op.opid]
                used_units = []
                now_order = order
                next_order = order + 1
                while next_order < len(self.graph.ordered_ops):
                    # OEM's NPU can't support the cascade pattern with more than 4 ops
                    if len(cascade_matched_ops) >= 4:
                        break
                    now_op = self.graph.ordered_ops[now_order]
                    next_op = self.graph.ordered_ops[next_order]
                    opcode_index = next_op.info.get('opcode_index')
                    opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

                    if opcode_type in used_units:
                        break
                    used_units.append(opcode_type)

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

                    # Skip the reshape op (we will update the related tensors' info in later phase)
                    if opcode_type == "RESHAPE":
                        have_matched[next_order] = True
                        now_order = next_order
                        next_order += 1
                        continue

                    if opcode_type in elem_wise_ops:
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
    
    # Since we will skip the reshape op in codegen phase, we need to update the related tensors' info
    def update_reshape_related_tensors(self):
        # Steps 1: Collect all reshape ops
        reshape_ops = []
        for op in self.ordered_ops:
            opcode_index = op.info['opcode_index']
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            if opcode_type == "RESHAPE":
                reshape_ops.append(op)
        # Steps 2: Update the related tensors' info
        for op in reshape_ops:
            input_tensor_id = op.info['inputs'][0]
            for tensor in self.tensor_info[input_tensor_id].tensors:
                if tensor.cid == op.opid:
                    input_tensor = tensor
                    break
            output_tensor_id = op.info['outputs'][0]
            for tensor in self.tensor_info[output_tensor_id].tensors:
                if tensor.pid == op.opid:
                    output_tensor = tensor
                    break
            # Update the pid & cid
            input_tensor.cid = output_tensor.cid
            # Update the input tensor's live range
            input_tensor.live_range['last_time_used'] = output_tensor.live_range['last_time_used']
            # Update output tensor's tensors
            self.tensor_info[output_tensor_id].tensors = [input_tensor]
        
    # Use Chaitin-Briggs algorithm to color the graph
    # Note: In here, we only accept that SRAM be reused between the sequential patterns, 
    # otherwise, the output of the pattern will be stored back from SRAM to DRAM [TODO]
    def allocate_success(self, operators, last_pattern_output_tensors):
        class ColoringNode:
            def __init__(self):
                # The node's neighbors, which may decrease by removing the node (element's data type: Distributed_SRAM_tensor_info)
                self.neighbors = []
                # For coloring use, avoid using the same color
                self.neighbors_list = []
                self.color = -1

        class ColoringGraph:
            def __init__(self, colors):
                self.colors = colors
                self.nodes = {}

        # The color we can allocate to the tensor (False means color not used)
        colors = [False for _ in range(self.total_SRAMs)]
        # Steps 1: Collect all tensors(Distributed_SRAM_tensor_info) that need to be allocated
        candidate_tensors = []
        # Store the output tensors of the concurrent run pattern, that subsequent patterns can directly reuse the same SRAM
        pattern_output_tensors = []
        total_operators = len(operators)
        for op_idx, op in enumerate(operators):
            prev_op_idx = op_idx - 1
            prev_op = operators[prev_op_idx] if prev_op_idx >= 0 else None
            next_op_idx = op_idx + 1
            next_op = operators[next_op_idx] if next_op_idx < total_operators else None
            opcode_index = op.info['opcode_index']
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

            # Collect input tensors
            if opcode_type in unary_ops:
                input_need = 1
            elif opcode_type in binary_ops:
                input_need = 2
            elif opcode_type in trinary_ops:
                # For now, we don't consider the bias tensor
                input_need = 2
            else:
                input_need = len(op.info['inputs'])
            for tensor_id in op.info['inputs'][0: input_need]:
                if tensor_id == -1:
                    continue
                # Based on pid & cid to find the tensor
                for tensor in self.tensor_info[tensor_id].tensors:
                    if tensor.cid == op.opid:
                        # In OEM's NPU the intermediate tensor is flow to buffer not store back to SRAM
                        if prev_op != None and tensor.pid == prev_op.opid:
                            break
                        # Check the tensor whether is already allocated and for now we only accept sram be reused between the concurrent run pattern
                        if tensor in last_pattern_output_tensors and tensor.sram_id != -1:
                            # If the tensor is already allocated, we use it directly and mark the color
                            colors[tensor.sram_id] = True
                            tensor.dram_access = False
                            break
                        # No need to access the DRAM, since it will be allocated to the SRAM in the future
                        if tensor in candidate_tensors:
                            tensor.dram_access = False
                        else:
                            candidate_tensors.append(tensor)
                        break
            # Collect output tensors
            for tensor_id in op.info['outputs']:
                # May encount that output tensor consume by more than 8 ops (can't allocate each tensor_info to different SRAM)
                if len(self.tensor_info[tensor_id].tensors) >= self.total_SRAMs - 2:
                    # Such case's outputs consume timestamp usually is not very close, so we only allocate first tensor_info
                    tensor = self.tensor_info[tensor_id].tensors[0]
                    if next_op == None:
                        candidate_tensors.append(tensor)
                        pattern_output_tensors.append(tensor)
                    if next_op != None and tensor.cid != next_op.opid:
                        candidate_tensors.append(tensor)
                        pattern_output_tensors.append(tensor)
                else:
                    for tensor in self.tensor_info[tensor_id].tensors:
                        # Won't break as early as possible, since same tensor may be used by different ops
                        if tensor.pid == op.opid:
                            # If the next op won't consume the output tensor immediately, we need to store the output tensor back to SRAM
                            if next_op == None:
                                candidate_tensors.append(tensor)
                                pattern_output_tensors.append(tensor)
                            if next_op != None and tensor.cid != next_op.opid:
                                candidate_tensors.append(tensor)
                                pattern_output_tensors.append(tensor)

        # Steps 2: Build the interference graph
        coloring_graph = ColoringGraph(colors)
        for tensor in candidate_tensors:
            coloring_graph.nodes.update({tensor: ColoringNode()})
        # Use O(n-square) approach to build the graph
        # TODO: Potential improvements in future work
        for tensor1 in candidate_tensors:
            for tensor2 in candidate_tensors:
                if tensor1 == tensor2:
                    continue
                # Check if the two tensors' live range overlap
                # If the two tensors' live range overlap, we need to add an edge between them
                if tensor1.live_range['first_time_used'] <= tensor2.live_range['last_time_used'] and \
                   tensor2.live_range['first_time_used'] <= tensor1.live_range['last_time_used']:
                    coloring_graph.nodes[tensor1].neighbors.append(tensor2)
                    coloring_graph.nodes[tensor1].neighbors_list.append(tensor2)
        # Steps 3: Check which nodes can be trivially pushed to stack
        empty_colors = 0
        for color_been_used in colors:
            if not color_been_used:
                empty_colors += 1
        stack = []
        # Maintain a copy of the tensor to node mapping for final color assignment
        tensor_node_mapping = {}
        for tensor, node in coloring_graph.nodes.items():
            tensor_node_mapping.update({tensor: node})
        # Pick the node which its degree is less than empty_colors and push it to the stack
        for tensor, node in list(coloring_graph.nodes.items()):
            if len(node.neighbors) < empty_colors:
                stack.append(node)
                # Remove the edge in the graph
                for node_II in coloring_graph.nodes.values():
                    if tensor in node_II.neighbors:
                        node_II.neighbors.remove(tensor)
                # Remove the node from the graph
                coloring_graph.nodes.pop(tensor)
        # If all nodes' neighbors are smaller than empty_colors, we can definitely color the graph
        # Original Chaitin-Briggs algorithm can further consider the degree of the node which is equal to empty_colors(by spilling), 
        # but in here we don't consider it
        if len(coloring_graph.nodes) != 0:
            return False, []
        # Steps 4: Pop the nodes from stack and try to color
        while len(stack) != 0:
            node = stack.pop()
            # In here, we don't consider to reuse the same color(SRAM), since the concurrent run pattern will use those SRAMs at the same time
            if empty_colors == 0:
                return False, []
            for color_idx, color_been_used in enumerate(colors):
                if not color_been_used:
                    node.color = color_idx
                    colors[color_idx] = True
                    empty_colors -= 1
                    break
        # Steps 5: Assign the color to the tensor
        for tensor in candidate_tensors:
            tensor.sram_id = tensor_node_mapping[tensor].color
        return True, pattern_output_tensors