from MyGraph import Node, Graph
from AutoSplit import Splitter
from OpClassify import Op_Classify
import copy
import numpy as np
import math

sink_data_layout = ["PACK"]
hoist_data_layout = ["SPLIT"]
safe_data_layout_ops = ["RESHAPE"]
reduce_ops = Op_Classify().reduce_ops
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

    # Try to sink concat or pack op
    # For now, we want to fuse the FC in self-attention with the pack op followed elementwise ops
    def data_layout_sink(self):
        # Step 1: Find the valid sink pattern
        valid_pattern = self.find_valid_sink_pattern()
        for pattern in valid_pattern:
            pack_op = pattern[0]
            # Step 2: Find the number of the heads
            pack_output_tensor = self.tensors[pack_op.info['outputs'][0]]
            pack_output_shape = pack_output_tensor['shape']
            head_num = pack_output_shape[0]
            # Step 3: Try to sink the pack op (delay the head pack)
            # Record the previous layer's opids for updating the new op's parents
            prev_opids = pack_op.parents
            prev_opids = sorted(prev_opids)
            # Record the last op's output tensor id to update the pack op's output tensor
            last_op_in_pattern = pattern[-1]
            last_op_output_tensor_id = last_op_in_pattern.info['outputs'][0]
            # Record the last op's child for updating pack op's child
            final_opid = last_op_in_pattern.children[0]
            final_op = self.ops[final_opid]
            # In the initial phase, the input tensor is multi-head attention's fully connected op's output tensor
            split_input_tensor_ids = pack_op.info['inputs']
            for op in pattern[1:]:
                # Find the head_num dimension & perform head split on output tensor
                output_tensor_id = op.info['outputs'][0]
                output_tensor_shape = self.tensors[output_tensor_id]['shape']
                for dim, dim_val in enumerate(output_tensor_shape):
                    if dim_val == head_num:
                        split_output_tensor_ids = self.split_tensor_by_n(output_tensor_id, 1, dim)
                        # Check whether need to split the constant tensor in the binary op (subset of elementwise op)
                        opcode_index = op.info.get("opcode_index")
                        opcode_type = self.opcodes[opcode_index].get("builtin_code")
                        if opcode_type in binary_ops:
                            input_tensor2_id = op.info['inputs'][1]
                            split_input_tensor2_ids = self.split_constant_tensor_by_n(input_tensor2_id, 1, dim)
                            # Copy the same tensor id for the other
                            if len(split_input_tensor2_ids) != head_num:
                                for _ in range(head_num - 1):
                                    split_input_tensor2_ids.append(split_input_tensor2_ids[0])
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

                split_input_tensor_ids = split_output_tensor_ids
                prev_opids = next_opids
            # Step 3: Create a new concatenate op
            new_concat_info = {
                "opcode_index": self.get_opcode_index(2),
                "inputs": split_output_tensor_ids,
                "outputs": [last_op_output_tensor_id],
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
            self.ops[new_concat_opid].children = [final_opid]
            for i, parent in enumerate(final_op.parents):
                if parent == last_op_in_pattern.opid:
                    del final_op.parents[i]
            final_op.parents.append(new_concat_opid)
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)
        
    def find_valid_sink_pattern(self):
        # Find sink candidate ops
        candidate_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type in sink_data_layout:
                candidate_ops.append(op)
        # Sink the candidate ops
        # Rule 1: If the path encount the op has multiple children or multiple parents, then stop sinking
        # Rule 2: If the path encount the op that is belong to the reduce op, then stop sinking
        # Rule 3: If the path doesn't have the concurrent run oppotunity, then cancel the sinking
        # Rule 3.1: If the path don't have the elementwise op, then cancel the sinking
        # Rule 3.2: If the path's elementwise op will concurrent run with other mac-main ops, then cancel the sinking
        valid_pattern = []
        for op in candidate_ops:
            tmp_pattern = [op]
            now_opid = op.opid
            while(True):
                # Check the next order's op is the child of the current op
                next_opid = now_opid + 1
                if next_opid in self.ops[now_opid].children:
                    next_op = self.ops[next_opid]
                    # Check the next order's op has only one parent and one child
                    if len(next_op.parents) != 1 or len(next_op.children) != 1:
                        break
                    # Avoid to sink across the reduce op or mac-main op (we didn't plan to support mac-main op)
                    opcode_index = next_op.info.get("opcode_index")
                    opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    if opcode_type in reduce_ops or opcode_type in mac_ops:
                        break
                    tmp_pattern.append(next_op)
                    now_opid = next_opid
                else:
                    break
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
    def split_constant_tensor_by_n(self, tensor_id_in, tile_size, tile_dim):
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
        if shape == [] or tile_dim >= len(shape):
            for i in range(0, 1, 1):
                new_buffer_info = copy.deepcopy(new_buffer_info_base)
                new_tensor_info = copy.deepcopy(new_tensor_info_base)
                new_tensor_info['buffer'] = buffer_id
                new_tensor_info['name'] += '_split_%d' % (i)
                
                # Extract the data
                if len(shape) != 1:
                    raise BaseException("The shape of the constant tensor is not supported")
                
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