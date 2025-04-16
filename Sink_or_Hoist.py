from collections import defaultdict
from MyGraph import Node, Graph
from AutoSplit import Splitter
from OpClassify import Op_Classify
import copy
import numpy as np
import math

sink_data_layout = ["PACK", "CONCATENATION"]
hoist_data_layout = ["SPLIT"]
safe_data_layout_ops = ["RESHAPE"]
reduce_ops = Op_Classify().reduce_ops
fall_back_cpu_ops = Op_Classify().fall_back_cpu_ops
mac_ops = Op_Classify().mac_ops
elementwise_ops = Op_Classify().elementwise_ops
binary_ops = Op_Classify().binary_ops

class Safe_Sinker_Hoister:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def re_init(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    # Try to sink concat or pack op
    # For now, we want to fuse the FC in self-attention with the elementwise operations that are followed by the pack operation
    def data_layout_sink(self):
        # Continue sinking until no valid pattern remains
        while True:
            # Step 1: Find the valid sink pattern
            valid_pattern = self.find_valid_sink_pattern()
            if len(valid_pattern) == 0:
                break
            for pattern in valid_pattern:
                pack_op = pattern[0]
                # Step 2: Find the number of the heads
                pack_output_tensor = self.tensors[pack_op.info['outputs'][0]]
                pack_output_shape = pack_output_tensor['shape']
                head_num = pack_output_shape[0]
                # Record the previous layer's opids for updating the new op's parents
                prev_opids = pack_op.parents
                prev_opids = sorted(prev_opids)
                # Record the last op's output tensor id to update the pack op's output tensor
                last_op_in_pattern = pattern[-1]
                last_op_output_tensor_ids = last_op_in_pattern.info['outputs']
                # Record the last op's child for updating pack op's child
                final_opids = last_op_in_pattern.children
                final_opids = sorted(final_opids)
                final_ops = []
                for final_opid in final_opids:
                    final_ops.append(self.ops[final_opid])
                # Check last op in pattern whether is the split op, which can be eliminated with pack op
                encount_split = False
                # In the initial phase, the input tensor is pack op's split output tensor & update prev_op's output tensor
                split_input_tensor_ids = self.split_tensor_by_n(pack_op.info['outputs'][0], 1, 0)
                for i, prev_opid in enumerate(prev_opids):
                    self.ops[prev_opid].info['outputs'] = [split_input_tensor_ids[i]]

                # Step 3: Try to sink the pack op (delay the head pack)
                for op in pattern[1:]:
                    # Check whether the op is a split op (usually is the last op in the pattern)
                    opcode_index = op.info.get("opcode_index")
                    opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    if opcode_type in hoist_data_layout:
                        encount_split = True
                        break

                    # We support that if both of the op's parents are pack/concatenate op, we can combine them, 
                    # this is for checking whether it needs to handle the binary op's second input
                    have_non_constant_tensor2 = False
                    # Find the head_num dimension & perform head split on output tensor
                    output_tensor_id = op.info['outputs'][0]
                    output_tensor_shape = self.tensors[output_tensor_id]['shape']
                    for dim, dim_val in enumerate(output_tensor_shape):
                        if dim_val == head_num:
                            split_output_tensor_ids = self.split_tensor_by_n(output_tensor_id, 1, dim)
                            # Update the output tensor id
                            # Check whether it needs to split the constant tensor in the binary op (subset of elementwise op)
                            opcode_index = op.info.get("opcode_index")
                            opcode_type = self.opcodes[opcode_index].get("builtin_code")
                            if opcode_type in binary_ops:
                                input_tensor2_id = op.info['inputs'][1]
                                # input_tensor2 is the constant tensor
                                if len(self.buffers[self.tensors[input_tensor2_id]['buffer']]) != 0:
                                    # For checking whether it needs to perform real split (the tensor may original needs broadcast)
                                    input_tensor_1_shape_len = len(self.tensors[split_input_tensor_ids[0]]['shape'])
                                    split_input_tensor2_ids = self.split_constant_tensor_by_n(input_tensor2_id, 1, dim, input_tensor_1_shape_len)
                                    # If not perform split, needs to copy the same tensor id (head_num - 1) times
                                    if len(split_input_tensor2_ids) != head_num:
                                        for _ in range(head_num - 1):
                                            split_input_tensor2_ids.append(split_input_tensor2_ids[0])
                                else:
                                # This case only happen once:
                                #  pack pack
                                #    |  /
                                #    add
                                    have_non_constant_tensor2 = True
                                    split_input_tensor2_ids = self.split_tensor_by_n(input_tensor2_id, 1, dim)
                                    # Update the input2 op(another pack)'s parents's output tensor
                                    for parent_id in op.parents:
                                        parent_output_tensor_ids = self.ops[parent_id].info['outputs']
                                        if input_tensor2_id in parent_output_tensor_ids:
                                            pack_op_2 = self.ops[parent_id]
                                            break
                                    prev_opids_2 = pack_op_2.parents
                                    prev_opids_2 = sorted(prev_opids_2)
                                    for i, prev_opid_2 in enumerate(prev_opids_2):
                                        self.ops[prev_opid_2].info['outputs'] = [split_input_tensor2_ids[i]]
                            break
                    # Record the next layer's opids for updating the new op's children
                    next_opids = []
                    # The followed ops need to be copy head_num times
                    for i in range(head_num):
                        opcode_index = op.info.get("opcode_index")
                        opcode_type = self.opcodes[opcode_index].get("builtin_code")
                        if opcode_type in elementwise_ops:
                            new_op_info = copy.deepcopy(op.info)
                            new_op_info['inputs'][0] = split_input_tensor_ids[i]
                            new_op_info['inputs'][1] = split_input_tensor2_ids[i]
                            new_op_info['outputs'] = [split_output_tensor_ids[i]]
                        # No need to modify the reshape's shape tensor (since we have already modify the output tensor)
                        elif opcode_type in safe_data_layout_ops:
                            new_op_info = copy.deepcopy(op.info)
                            new_op_info['inputs'][0] = split_input_tensor_ids[i]
                            new_op_info['outputs'] = [split_output_tensor_ids[i]]
                        else:
                            raise ValueError(f"Op info: {op.info} is not supported for sink in Safe_Sinker_Hoister")
                        self.operators.append(new_op_info)
                        new_opid = len(self.operators) - 1
                        self.ops.append(Node(new_op_info, new_opid))
                        next_opids.append(new_opid)

                        # Update the new_op's parents and children
                        self.ops[prev_opids[i]].children = [new_opid]
                        self.ops[new_opid].parents = [prev_opids[i]]
                        if have_non_constant_tensor2:
                            self.ops[prev_opids_2[i]].children = [new_opid]
                            self.ops[new_opid].parents.append(prev_opids_2[i])

                    split_input_tensor_ids = split_output_tensor_ids
                    prev_opids = next_opids

                # Handle tail of pattern
                if not encount_split:
                    # Step 4: Create a new concatenate op (if last op is not a split op)
                    new_concat_info = {
                        "opcode_index": self.get_opcode_index(2),
                        "inputs": split_output_tensor_ids,
                        "outputs": last_op_output_tensor_ids,
                        "builtin_options_type": "ConcatenationOptions",
                        "builtin_options": {
                            'axis': 1
                        }
                    }
                    self.operators.append(new_concat_info)
                    new_concat_opid = len(self.operators) - 1
                    self.ops.append(Node(new_concat_info, new_concat_opid))
                    # Update the concatenate op's parents and children
                    for prev_opid in prev_opids:
                        self.ops[prev_opid].children = [new_concat_opid]
                    self.ops[new_concat_opid].parents = prev_opids
                    self.ops[new_concat_opid].children = [final_opids[0]]
                    for i, parent in enumerate(final_ops[0].parents):
                        if parent == last_op_in_pattern.opid:
                            del final_ops[0].parents[i]
                    final_ops[0].parents.append(new_concat_opid)
                else:
                    # Step 5: No need to create a new concatenate op (if last op is a split op)
                    # But split's ouptut tensor may be shared with the child ops, we use a mapping table to mapping the tensor id
                    tensor_mapping = defaultdict(int)
                    split_op = pattern[-1]
                    for i, output_tensor_id in enumerate(split_op.info['outputs']):
                        tensor_mapping[output_tensor_id] = split_output_tensor_ids[i]
                    # Update the final op's input tensor id
                    for i, final_op in enumerate(final_ops):
                        final_op.info['inputs'][0] = tensor_mapping[final_op.info['inputs'][0]]
                    # Update the penultimate op's children & final op's parents
                    for i, prev_opid in enumerate(prev_opids):
                        prev_op = self.ops[prev_opid]
                        prev_op.children = []
                        for final_opid in final_opids:
                            final_op = self.ops[final_opid]
                            if prev_op.info['outputs'][0] == final_op.info['inputs'][0]:
                                prev_op.children.append(final_opid)
                                final_op.parents.append(prev_opid)
                            for j, parent in enumerate(final_op.parents):
                                if parent == split_op.opid:
                                    del final_op.parents[j]
        
            new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
            new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
            self.splitter.re_init(new_graph)
            # Must do this to update the graph, or the find_valid_sink_pattern func will deal with the old graph, and the program can't stop
            self.re_init(self.splitter)
        
    def find_valid_sink_pattern(self):
        # Find sink candidate ops
        candidate_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type in sink_data_layout:
                candidate_ops.append(op)
        # Sink the pack/candidate ops
        # Rule 1: If the path encount the op has multiple children or multiple parents, then stop sinking
        # Rule 2: If the path encount the op that is belong to the reduce op, then stop sinking
        # Rule 3: If the path doesn't have the concurrent run oppotunity, then cancel the sinking
        # Rule 3.1: If the path don't have the elementwise op, then cancel the sinking
        # Rule 3.2: If the path's elementwise op will concurrent run with other mac-main ops, then cancel the sinking
        valid_pattern = []
        for op in candidate_ops:
            tmp_pattern = [op]
            now_opid = op.opid
            now_op = self.ops[now_opid]
            # If the pack op have multiple children, then stop sinking
            if len(now_op.children) != 1:
                continue
            while(True):
                # Check the next order's op is the child of the current op
                next_opid = self.ops[now_opid].children[0]
                next_op = self.ops[next_opid]
                opcode_index = next_op.info.get("opcode_index")
                opcode_type = self.opcodes[opcode_index].get("builtin_code")
                # If pack op encount the split op, we can eliminate these pack & split pattern if split's axis is 0
                if opcode_type in hoist_data_layout:
                    axis_tensor = self.tensors[next_op.info['inputs'][0]]
                    axis_buffer = self.buffers[axis_tensor['buffer']]
                    axis = axis_buffer['data'][0]
                    if axis == 0:
                        tmp_pattern.append(next_op)
                    break
                # Avoid to sink across the reduce ops or fall back cpu ops and mac-main ops (we didn't plan to support mac-main op)
                if opcode_type in reduce_ops or opcode_type in fall_back_cpu_ops or opcode_type in mac_ops:
                    break
                # Check the next order's op has only one parent and one child
                if len(next_op.children) != 1:
                    break
                if len(next_op.parents) != 1:
                    # If both parents are pack/concatenate op, we can combine them, and pack can further sink
                    for parent in next_op.parents:
                        opcode_index = self.ops[parent].info.get("opcode_index")
                        opcode_type = self.opcodes[opcode_index].get("builtin_code")
                        if opcode_type not in sink_data_layout:
                            break
                    # To avoid handle this op multiple times, make sure handle the next op in inputs[0]'s pack
                    if next_op.info['inputs'][0] in op.info['outputs']:
                        tmp_pattern.append(next_op)
                    break
                tmp_pattern.append(next_op)
                now_opid = next_opid
            for op in tmp_pattern:
                opcode_index = op.info.get("opcode_index")
                opcode_type = self.opcodes[opcode_index].get("builtin_code")
                if opcode_type in elementwise_ops:
                    valid_pattern.append(tmp_pattern)
                    break
        return valid_pattern
    
    def data_layout_hoist(self):
        pass

    # Refer to the split_tensor_by_n in Splitter
    def split_tensor_by_n(self, tensor_id_in, tile_size, tile_dim):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        split_tensor_ids = []
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim] / tile_size), 1):
            guard = min(tile_size, tensor_info['shape'][tile_dim] - i * tile_size)
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['shape'][tile_dim] = guard
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_head_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            split_tensor_ids.append(tensor_id)
            buffer_id += 1
            tensor_id += 1
        return split_tensor_ids
    
    # Refer to the split_constant_tensor_by_n in Splitter
    def split_constant_tensor_by_n(self, tensor_id_in, tile_size, tile_dim, input_tensor_1_shape_len):
        buffer_info = self.buffers[self.tensors[tensor_id_in]['buffer']]
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_buffer_info_base = copy.deepcopy(buffer_info)
        new_tensor_info_base = copy.deepcopy(tensor_info)

        # Prepare for copy the buffer data
        ori_buffer_info = new_buffer_info_base
        if len(ori_buffer_info['data']) == 0:
            raise BaseException("The buffer data is empty, can't split the constant tensor")
        # Based on the data type to determine the numpy type
        if tensor_info.get("type") == 'INT8':
            np_type = np.int8
        else:
            np_type = np.float32
        # Convert the buffer data to numpy array
        one_dim_arr = np.frombuffer(bytes(ori_buffer_info['data']), dtype = np_type)
        # If the buffer only contain one element (may loss shape info or shape = []), let the shape be [1]
        shape = tensor_info.get("shape", [])
        # Reshape to the original shape
        np_arr = one_dim_arr.reshape(shape)

        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        split_tensor_ids = []
        # The constant tensor needs to perform broadcast
        if len(shape) != input_tensor_1_shape_len:
            for i in range(0, 1, 1):
                new_buffer_info = copy.deepcopy(new_buffer_info_base)
                new_tensor_info = copy.deepcopy(new_tensor_info_base)
                new_tensor_info['buffer'] = buffer_id
                new_tensor_info['name'] += '_split_%d' % (i)
                self.buffers.append(new_buffer_info)
                self.tensors.append(new_tensor_info)
                split_tensor_ids.append(tensor_id)
                buffer_id += 1
                tensor_id += 1
        else:
            for i in range(0, math.ceil(tensor_info['shape'][tile_dim] / tile_size), 1):
                guard = min(tile_size, tensor_info['shape'][tile_dim] - i * tile_size)
                new_buffer_info = copy.deepcopy(new_buffer_info_base)
                new_tensor_info = copy.deepcopy(new_tensor_info_base)
                new_tensor_info['shape'][tile_dim] = guard
                new_tensor_info['buffer'] = buffer_id
                new_tensor_info['name'] += '_split_%d' % (i)
                
                # Extract the data
                if len(shape) == 4:
                    if tile_dim == 0:
                        tmp_data = np_arr[i * tile_size : i * tile_size + guard, :, :, :]
                    elif tile_dim == 1:
                        tmp_data = np_arr[:, i * tile_size : i * tile_size + guard, :, :]
                    elif tile_dim == 2:
                        tmp_data = np_arr[:, :, i * tile_size : i * tile_size + guard, :]
                    elif tile_dim == 3:
                        tmp_data = np_arr[:, :, :, i * tile_size : i * tile_size + guard]
                elif len(shape) == 3:
                    if tile_dim == 0:
                        tmp_data = np_arr[i * tile_size : i * tile_size + guard, :, :]
                    elif tile_dim == 1:
                        tmp_data = np_arr[:, i * tile_size : i * tile_size + guard, :]
                    elif tile_dim == 2:
                        tmp_data = np_arr[:, :, i * tile_size : i * tile_size + guard]
                elif len(shape) == 2:
                    if tile_dim == 0:
                        tmp_data = np_arr[i * tile_size : i * tile_size + guard, :]
                    elif tile_dim == 1:
                        tmp_data = np_arr[:, i * tile_size : i * tile_size + guard]
                # Flatten the data
                new_data = tmp_data.astype(np_type).tobytes()
                new_buffer_info['data'] = list(new_data)
                
                self.buffers.append(new_buffer_info)
                self.tensors.append(new_tensor_info)
                split_tensor_ids.append(tensor_id)
                buffer_id += 1
                tensor_id += 1
        return split_tensor_ids
    
    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'