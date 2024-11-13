from MyGraph import Graph
import copy

class Optimizer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.visited = [False] * len(graph.ops)
        self.split_tensor_table = [list() for _ in range(len(self.graph.tensors))]
        self.spilt_transpose_opid = []
        self.need_delete = []

    def from_split_find_parent_transpose(self, current_opid):
        # For now, we try to find the transpose five steps away
        op = self.graph.ops[current_opid]
        current_opid = op.parents[0]
        steps = 5
        while((steps != 0) is True):
            op = self.graph.ops[current_opid]
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
            if opcode_type == 'TRANSPOSE':
                return current_opid
            current_opid = op.parents[0]
            steps -= 1
    
    def find_split_and_traverse_til_end(self, current_opid):
        if self.visited[current_opid]:
            return
        self.visited[current_opid] = True
        opcode_index = self.graph.ops[current_opid].info.get("opcode_index")
        opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'SPLIT':
            transpose_opid = self.from_split_find_parent_transpose(current_opid)
            if transpose_opid is not None:
                split_transpose_pattern = (current_opid, transpose_opid)
                self.spilt_transpose_opid.append(split_transpose_pattern)
        for child in self.graph.ops[current_opid].children:
            # check if it is splittable op
            self.find_split_and_traverse_til_end(child)
        return
    
    def split_tensor_by_n(self, tensor_id_in, tile_size, tile_dim):
        tensor_info = self.graph.tensors[tensor_id_in]
        buffer_id = len(self.graph.buffers)
        tensor_id = len(self.graph.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim] / tile_size), 1):
            guard = min(tile_size, tensor_info['shape'][tile_dim] - i * tile_size)
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['shape'][tile_dim] = guard
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '%d' % (i)
            self.graph.buffers.append({})
            self.graph.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1
    
    def perform_hoist_split(self):
        # Handle each split_transpose pattern
        for pattern in self.spilt_transpose_opid:
            split_opid, transpose_opid = pattern
            split_op = self.graph.ops[split_opid]
            transpose_op = self.graph.ops[transpose_opid]
            # Fetch the start and end opid
            current_opid = split_op.parents[0]
            stop_opid = transpose_op.parents[0]
            # Fetch the axis & dim val from the split op
            axis_buffer = self.graph.buffers[self.graph.tensors[split_op.info['inputs'][0]]['buffer']]
            axis = axis_buffer['data'][0]
            input_shape = self.graph.tensors[split_op.info['inputs'][1]]['shape']
            dim_val = input_shape[axis]
            # Move the split op to the parent of the transpose op
            while(current_opid != stop_opid):
                current_op = self.graph.ops[current_opid]
                current_input = current_op.info['inputs'][0]
                # According to the current op type, we need to update the input/output
                opcode_index = current_op.info.get("opcode_index")
                opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
                if opcode_type == 'TRANSPOSE':
                    # Fetch the permute buffer
                    perm_tensor = self.graph.tensors[current_op.info['inputs'][1]]
                    perm_buffer = self.graph.buffers[perm_tensor['buffer']]['data']
                    # Process buffer structure: [a, 0, 0, 0, b, 0, 0, 0, c, 0, 0, 0, d, 0, 0, 0]
                    perm = []
                    for i in range(0, len(perm_buffer), 4):
                        perm.append(perm_buffer[i])
                    # Update the axis
                    axis = perm[axis]  
                    # Split the input of the transpose op
                    self.split_tensor_by_n(current_input, 1, axis) 
                    # Copy transpose op #new_outputs times & update the input/output of split op & reshape op
                    for i, a in enumerate(self.split_tensor_table[current_input]):
                        new_op_info = copy.deepcopy(current_op.info)
                        new_op_info['inputs'][0] = a
                        new_op_info['outputs'][0] = split_op.info['outputs'][i]
                        self.graph.operators.append(new_op_info)
                        split_op.info['inputs'][1] = current_input
                        split_op.info['outputs'][i] = a
                    # Record transpose op needs to be deleted
                    self.need_delete.append(current_opid)
                elif opcode_type == 'RESHAPE':
                    # Update the axis
                    reshape_input = self.graph.tensors[current_input]['shape']
                    for dim in range(len(reshape_input)):
                        if reshape_input[dim] == dim_val:
                            axis = dim
                            break
                    # Split the input of the reshape op
                    self.split_tensor_by_n(current_input, 1, axis)
                    # Copy reshape op #new_outputs times & update the input/output of split op & reshape op
                    for i, a in enumerate(self.split_tensor_table[current_input]):
                        new_op_info = copy.deepcopy(current_op.info)
                        new_op_info['inputs'][0] = a
                        new_op_info['outputs'][0] = split_op.info['outputs'][i]
                        self.graph.operators.append(new_op_info)
                        split_op.info['inputs'][1] = current_input
                        split_op.info['outputs'][i] = a
                    # Record reshape op needs to be deleted
                    self.need_delete.append(current_opid)
                else:
                    # Split the input of the reshape op
                    self.split_tensor_by_n(current_input, 1, axis)
                    # Copy reshape op #new_outputs times & update the input/output of split op & reshape op
                    for i, a in enumerate(self.split_tensor_table[current_input]):
                        new_op_info = copy.deepcopy(current_op.info)
                        new_op_info['inputs'][0] = a
                        new_op_info['outputs'][0] = split_op.info['outputs'][i]
                        self.graph.operators.append(new_op_info)
                        split_op.info['inputs'][1] = current_input
                        split_op.info['outputs'][i] = a
                    # Record reshape op needs to be deleted
                    self.need_delete.append(current_opid)
                # Update axis buffer in split op (the original buffer will share with other ops, so we need to create a new buffer & tensor)
                buffer_id = len(self.graph.buffers)
                new_buffer = copy.deepcopy(axis_buffer)
                new_buffer['data'][0] = axis
                self.graph.buffers.append(new_buffer)
                tensor_id = len(self.graph.tensors)
                new_tensor = copy.deepcopy(self.graph.tensors[split_op.info['inputs'][0]])
                new_tensor['buffer'] = buffer_id
                self.graph.tensors.append(new_tensor)
                split_op.info['inputs'][0] = tensor_id
                buffer_id += 1
                # Update the current opid
                current_opid = current_op.parents[0]

    # For now, we assume the case is between the path from transpose to split (because tranformers usually have this pattern)
    def hoist_split(self, graph: Graph):
        # Traverse the graph from the root op
        root_id = graph.root_op_id[-1]
        self.find_split_and_traverse_til_end(root_id)
        self.perform_hoist_split()
    
    def eliminate_transpose(self, graph: Graph):
        pass

    def update_operators(self):
        for opid in self.need_delete:
            self.graph.operators[opid] = None
        # Delete the None op_info
        self.graph.operators = [op_info for op_info in self.graph.operators if op_info is not None]

    def perform_optimize(self):
        self.hoist_split(self.graph)
        self.eliminate_transpose(self.graph)
        self.update_operators()
        opt_ori_graph = Graph(self.graph.operators, self.graph.tensors, self.graph.buffers, self.graph.opcodes, self.graph.inputs, self.graph.outputs, self.graph.exec_order)
        return opt_ori_graph