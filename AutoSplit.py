from MyGraph import Node,Graph
import copy

# bert prefer split token
# yolo prefer split height
class ModelType:
    BERT = 0
    YOLO = 1

class SplitterNode:
    def __init__(self, node:Node):
        self.node = node
        self.split_id = []
        self.visited = False

class Splitter:
    def __init__(self,ori_graph:Graph, split_height:int, model_type:int):
        self.padding_param_tensors = {}
        self.ori_graph = ori_graph
        self.split_height = split_height
        self.opcodes =  copy.deepcopy(ori_graph.opcodes)
        self.tensors = copy.deepcopy(ori_graph.tensors)
        self.buffers = copy.deepcopy(ori_graph.buffers)
        self.operators = [copy.deepcopy(n.info) for n in self.ori_graph.ops]
        self.nodes = [SplitterNode(n) for n in self.ori_graph.ops]
        self.splittable_opcode_idxes = {}
        self.model_type = model_type
        # For BERT model, assume the token size is 100
        self.token_size = 100
        for i, opcode in enumerate(self.opcodes):
            # 0: ADD
            # 2: CONCATENATION
            # 3: CONV_2D
            # 4: DEPTHWISE_CONV_2D
            # 9: FULLY_CONNECTED
            # 14: LOGISTIC
            # 17: MAX_POOL_2D
            # 18: MUL
            # 22: RESHAPE
            # 25: SOFTMAX
            # 34: PAD
            # 39: TRANSPOSE
            # 40: MEAN
            # 41: SUB
            # 49: SPLIT
            # 67: TRANSPOSE_CONV
            # 76: RSQRT
            # 82: REDUCE_MAX
            # 83: PACK
            # 97: RESIZE_NEAREST_NEIGHBOR
            # 98: LEAKY_RELU
            # 99: SQUARED_DIFFERENCE
            # 102: SPLIT_V
            # 114: QUANTIZE
            # 126: BATCH_MATMUL
            # 127: GELU
            split_candidate = [0, 2, 3, 4, 9, 14, 17, 18, 22, 25, 34, 39, 40, 41, 49, 67, 76, 82, 83, 97, 98, 99, 102, 114, 126, 127]
            if opcode.get("deprecated_builtin_code", 0) in split_candidate:
                self.splittable_opcode_idxes[opcode.get("deprecated_builtin_code", 0)] = i

    def re_init(self, ori_graph):
        self.ori_graph = ori_graph
        self.opcodes =  copy.deepcopy(ori_graph.opcodes)
        self.tensors = copy.deepcopy(ori_graph.tensors)
        self.buffers = copy.deepcopy(ori_graph.buffers)
        self.operators = [copy.deepcopy(n.info) for n in self.ori_graph.ops]
        self.nodes = [SplitterNode(n) for n in self.ori_graph.ops]
        self.splittable_opcode_idxes = {}
        for i, opcode in enumerate(self.opcodes):
            # 0: ADD
            # 2: CONCATENATION
            # 3: CONV_2D
            # 4: DEPTHWISE_CONV_2D
            # 9: FULLY_CONNECTED
            # 14: LOGISTIC
            # 17: MAX_POOL_2D
            # 18: MUL
            # 22: RESHAPE
            # 25: SOFTMAX
            # 34: PAD
            # 39: TRANSPOSE
            # 40: MEAN
            # 41: SUB
            # 49: SPLIT
            # 67: TRANSPOSE_CONV
            # 76: RSQRT
            # 82: REDUCE_MAX
            # 83: PACK
            # 97: RESIZE_NEAREST_NEIGHBOR
            # 98: LEAKY_RELU
            # 99: SQUARED_DIFFERENCE
            # 102: SPLIT_V
            # 114: QUANTIZE
            # 126: BATCH_MATMUL
            # 127: GELU
            split_candidate = [0, 2, 3, 4, 9, 14, 17, 18, 22, 25, 34, 39, 40, 41, 49, 67, 76, 82, 83, 97, 98, 99, 102, 114, 126, 127]
            if opcode.get("deprecated_builtin_code", 0) in split_candidate:
                self.splittable_opcode_idxes[opcode.get("deprecated_builtin_code", 0)] = i

    def perform_split(self, blocks = []) -> Graph:
        # There may be some manipulation will create new tensors
        self.split_tensor_table = [list() for _ in range(len(self.tensors) + 10000)]
        self.new_operators = []

        input_tile_size = self.split_height
        output_tile_size = self.split_height
        input_tile_size = 50
        output_tile_size = 50

        # Currently assume it's splittable from the root of the graph
        splittables = []
        end_ids = []
        # Fisrt, traverse from the multi inputs (Except the last input)
        if len(self.ori_graph.root_op_id) > 1:
            for root_op_id in self.ori_graph.root_op_id[: -1]:
                start_id = self.traverse_til_splittable(root_op_id)
                if start_id is not None:
                    self.split_block_input(start_id, input_tile_size)
                    self.traverse_til_not_splittable(start_id, splittables, end_ids)
        
        if len(blocks) == 0:
            # Then, traverse from the last input
            start_id = self.traverse_til_splittable(self.ori_graph.root_op_id[-1])
            # get splittable block
            self.traverse_til_not_splittable(start_id, splittables, end_ids)
            for end_id in end_ids:
                print(f"end_id info: {self.nodes[end_id].node.info}")
            # split block input
            self.split_block_input(start_id, input_tile_size)
            # start split
            for op in splittables:
                self.split_one_node(op, input_tile_size, output_tile_size)
            # concat block output
            for end_id in end_ids:
                self.concat_block_output(end_id)
        else:
            # Split the prologue
            for op in splittables:
                self.split_one_node(op, input_tile_size, output_tile_size)

            first_block = True
            epilogue_start_id = None
            # Perform TS on each block
            for block in blocks:
                start_id = block[0]
                end_id = block[1]
                splittables = []
                self.traverse_til_not_splittable_with_end_id(start_id, splittables, end_id)
                # Only first block need to split block input
                if first_block:
                    self.split_block_input(start_id, input_tile_size)
                    first_block = False
                else:
                    # Last block had splitted this block's input
                    splittables.pop(0)
                # start split block
                for op in splittables:
                    self.split_one_node(op, input_tile_size, output_tile_size)
                # concat than split block output
                outputs = self.nodes[end_id].node.info['outputs']
                for output in outputs:
                    output_name = self.tensors[output]['name']
                    for next_opid in self.nodes[end_id].node.children:
                        for i, input in enumerate(self.nodes[next_opid].node.info['inputs']):
                            # Need to match the next op's input tensor name to the output tensor name
                            if self.tensors[input]['name'] == output_name:
                                self.concat_than_split(next_opid, output_tile_size, input_idx = i)
                                break
                epilogue_start_id = end_id

            # Split the epilogue
            splittables = []
            end_ids = []
            start_id = epilogue_start_id
            self.traverse_til_not_splittable(start_id, splittables, end_ids)
            splittables.pop(0)
            # Perform TS on the rest of the graph
            for op in splittables:
                self.split_one_node(op, input_tile_size, output_tile_size)
            for end_id in end_ids:
                self.concat_block_output(end_id)
        
        new_graph = Graph ( self.new_operators, self.tensors, self.buffers,
                            self.opcodes, self.ori_graph.inputs, self.ori_graph.outputs, self.ori_graph.exec_order)

        # TODO: it will failed in self-attention model
        # new_graph.recycle_tensors_buffers()

        return new_graph

    def traverse_til_end(self, current_opid):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None

        self.new_operators.append(self.nodes[current_opid].node.info)

        self.nodes[current_opid].visited = True
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            result = self.traverse_til_end(child)
            if result is not None:
                return result
        return None

    def traverse_til_splittable(self, current_opid):
        if self.nodes[current_opid].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
            return current_opid
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None

        self.new_operators.append(self.nodes[current_opid].node.info)

        self.nodes[current_opid].visited = True
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            result = self.traverse_til_splittable(child)
            if result is not None:
                return result
        return None

    # Assume that always have one non-splittable op
    # Support there have multiple outputs
    def traverse_til_not_splittable(self, current_opid, splittables, end_ids):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None
        self.nodes[current_opid].visited = True
        if current_opid not in splittables:
            splittables.append(current_opid)
        else:
            return None
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            if self.nodes[child].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
                result = self.traverse_til_not_splittable(child, splittables, end_ids)
                if result is not None:
                    return result
            else:
                return child
        # To avoid the model with zero splittable op
        if len(self.nodes[current_opid].node.children) == 0:
            # Last op no need to split
            splittables.pop(-1)
            end_ids.append(current_opid)
        return None
    
    def traverse_til_not_splittable_with_end_id(self, current_opid, splittables, end_id):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None
            
        self.nodes[current_opid].visited = True
        if current_opid not in splittables:
            splittables.append(current_opid)

        # Traverse to the end_id
        if current_opid == end_id:
            # Different to the traverse_til_not_splittable, it keep the end_id in splittables
            return end_id        
        
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            if self.nodes[child].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
                result = self.traverse_til_not_splittable_with_end_id(child, splittables, end_id)
                if result is not None:
                    return result
            else:
                return None
        return None

    def split_tensor(self, tensor_id_in):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']
        new_tensor_info_base['shape'][1] = 1
        for i in range(tensor_info['shape'][1]):
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_split_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1

    def split_tensor_by_n(self, tensor_id_in, tile_size, tile_dim):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim] / tile_size), 1):
            guard = min(tile_size, tensor_info['shape'][tile_dim] - i * tile_size)
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['shape'][tile_dim] = guard
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_split_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1

    def split_tensor_by_nxn(self, tensor_id_in, tile_size, tile_dim1, tile_dim2):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim1]/tile_size), 1):
            guard1 = min(tile_size, tensor_info['shape'][tile_dim1] - i * tile_size)
            for j in range(0, math.ceil(tensor_info['shape'][tile_dim2]/tile_size), 1):
                guard2 = min(tile_size, tensor_info['shape'][tile_dim2] - j * tile_size)
                new_tensor_info = copy.deepcopy(new_tensor_info_base)
                new_tensor_info['shape'][tile_dim1] = guard1
                new_tensor_info['shape'][tile_dim2] = guard2
                new_tensor_info['buffer'] = buffer_id
                new_tensor_info['name'] += '_split_%d_%d' % (i, j)
                self.buffers.append({})
                self.tensors.append(new_tensor_info)
                self.split_tensor_table[tensor_id_in].append(tensor_id)
                buffer_id += 1
                tensor_id += 1

    def split_one_node(self, opid, input_split, output_split):
        opcode_idx = self.nodes[opid].node.info.get("opcode_index",0)

        if opcode_idx == self.splittable_opcode_idxes.get(0, -1):
            self.split_add(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(2, -1):
            self.split_concatenation(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(3 , -1):
            self.split_conv(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(4 , -1):
            self.split_dwconv(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(9, -1):
            self.split_fullyconnected(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(14, -1):
            self.split_logistic(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(17, -1):
            self.split_max_pool(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(18, -1):
            self.split_mul(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(22, -1):
            self.split_reshape(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(25, -1):
            self.split_softmax(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(34, -1):
            self.split_pad(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(39, -1):
            self.split_transpose(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(40, -1):
            self.split_mean(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(41, -1):
            self.split_sub(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(49, -1):
            self.split_split(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(67, -1):
            self.split_trconv(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(76, -1):
            self.split_rsqrt(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(83, -1):
            self.split_pack(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(97, -1):
            self.split_resize_nearest_neighbor(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(98, -1):
            self.split_leaky_relu(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(99, -1):
            self.split_squared_difference(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(102, -1):
            self.split_split_v(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(114, -1):
            self.split_quantize(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(126, -1):
            self.split_batch_matmul(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(127, -1):
            self.split_gelu(opid, output_split)

    def split_pad(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            i_len = len(self.split_tensor_table[inputs[0]])
            for a, b, c in zip(self.split_tensor_table[inputs[0]],
                               [inputs[1] for i in range(i_len)],
                               self.split_tensor_table[outputs[0]]):
                padding_index = b
                padding_tensor = self.tensors[padding_index]
                buffer_index = padding_tensor['buffer']
                buffer_data = self.buffers[buffer_index]
                buffer_data['data'][8] = 0
                buffer_data['data'][12] = 0

                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a, b]
                new_op_info['outputs'] = [c]
                op = Node(new_op_info, split_op_id)

                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    # TODO: Figure out how to control the split_dim
    # We follow the number of split input tensor to split the output tensor
    # For now, not support the split_dim decomposed to multiple dimension
    def split_reshape(self, opid, output_split):
        info = self.nodes[opid].node.info
        input = info['inputs'][0]
        output = info['outputs'][0]
        input_shape = self.tensors[input]['shape']
        output_shape = self.tensors[output]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            # Keep track the split_dim
            # Ex: split_dim = 2, output_split = 50, [1x64x100] -> [1x6400] => [1x64x50] -> [1x3200]
            # Ex: split_dim = 2, output_split = 10, [1x320x20x20] -> [1x10x32x20x20] => [1x320x10x20] -> [1x10x32x10x20]
            # Find out what split_dim is compose to output dimension
            input_split_dim = []
            if len(input_shape) != len(output_shape):
                if len(input_shape) > len(output_shape):
                    input_idx = 0
                    output_idx = 0
                    input_sum = 1
                    while output_idx < len(output_shape):
                        input_sum *= input_shape[input_idx]
                        input_split_dim.append(input_idx)
                        input_idx += 1
                        if output_shape[output_idx] == input_sum:
                            if self.split_dim in input_split_dim:
                                self.split_dim = output_idx
                                print(f"after reshape: split_dim: {self.split_dim}")
                                break
                            output_idx += 1
                            input_split_dim = []
                            input_sum = 1
                else:
                    split_dim_value = input_shape[self.split_dim]
                    for dim in range(len(output_shape)):
                        if output_shape[dim] == split_dim_value:
                            self.split_dim = dim
                            break

                    split_dim = idx
                    self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
                    for child in self.nodes[opid].node.children:
                        self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

            # Modify the split_tensor_by_n
            tensor_info = self.tensors[info['outputs'][0]]
            buffer_id = len(self.buffers)
            tensor_id = len(self.tensors)
            new_tensor_info_base = copy.deepcopy(tensor_info)
            if 'shape_signature' in new_tensor_info_base:
                del new_tensor_info_base['shape_signature']
            for i in range(len(self.split_tensor_table[info['inputs'][0]])):
                new_tensor_info = copy.deepcopy(new_tensor_info_base)
                if input_split_dim != []:
                    input_split_tensor_id = self.split_tensor_table[info['inputs'][0]][i]
                    sum = 1
                    for dim in input_split_dim:
                        sum *= self.tensors[input_split_tensor_id]['shape'][dim]
                    new_tensor_info['shape'][self.split_dim] = sum
                else:
                    new_tensor_info['shape'][self.split_dim] = output_split
                new_tensor_info['buffer'] = buffer_id
                new_tensor_info['name'] += '_split_%d' % (i)
                self.buffers.append({})
                self.tensors.append(new_tensor_info)
                self.split_tensor_table[info['outputs'][0]].append(tensor_id)
                buffer_id += 1
                tensor_id += 1
            
        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_transpose(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            perm_tensor = self.tensors[info['inputs'][1]]
            perm_buffer = self.buffers[perm_tensor['buffer']]['data']
            # Process buffer structure: [a, 0, 0, 0, b, 0, 0, 0, c, 0, 0, 0, d, 0, 0, 0]
            perm = []
            for i in range(0, len(perm_buffer), 4):
                perm.append(perm_buffer[i])
            
            split_dim = self.nodes[opid].node.split_dim
            for idx, i in enumerate(perm):
                if i == split_dim:
                    split_dim = idx
                    self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
                    for child in self.nodes[opid].node.children:
                        self.nodes[child].node.split_dim = split_dim     
                    break

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_softmax(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_split(self, opid, output_split):
        info = self.nodes[opid].node.info
        inputs = info['inputs']
        outputs = info['outputs']
        output_shape = self.tensors[outputs[0]]['shape']

        # split have multi outputs, each output need to be split
        axis_buffer = self.buffers[self.tensors[inputs[0]]['buffer']]
        axis = axis_buffer['data'][0]
        input_shape = self.tensors[inputs[1]]['shape']
        for i in range(input_shape[axis]):
            if self.model_type == ModelType.BERT:
                split_dim_value = self.token_size
                for dim, dim_value in enumerate(output_shape):
                    if dim_value == split_dim_value:
                        self.split_tensor_by_n(info['outputs'][i], output_split, dim)
                        break
            else:
                split_dim = self.nodes[opid].node.split_dim
                self.split_tensor_by_n(info['outputs'][i], output_split, split_dim)
                for child in self.nodes[opid].node.children:
                    self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != input_shape[axis]:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            
            for idx1, a in enumerate(self.split_tensor_table[inputs[1]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][1] = a
                new_op_output = []
                for idx2 in range(input_shape[axis]):
                    b = self.split_tensor_table[outputs[idx2]][idx1]
                    new_op_output.append(b)
                new_op_info['outputs'] = new_op_output
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_split_v(self, opid, output_split):
        info = self.nodes[opid].node.info
        inputs = info['inputs']
        outputs = info['outputs']

        split_dim = self.nodes[opid].node.split_dim
        for output_idx in range(len(outputs)):
            self.split_tensor_by_n(info['outputs'][output_idx], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        if len(inputs) != 3:
            raise "wrong input number"
        elif len(outputs) != size_splits:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            
            for idx1, a in enumerate(self.split_tensor_table[inputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_output = []
                for idx2 in range(size_splits):
                    b = self.split_tensor_table[outputs[idx2]][idx1]
                    new_op_output.append(b)
                new_op_info['outputs'] = new_op_output
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_batch_matmul(self, opid, output_split):
        info = self.nodes[opid].node.info
        inputs = info['inputs']
        outputs = info['outputs']

        self.split_tensor_by_nxn(info['outputs'][0], output_split, self.split_dim, self.split_dim + 1)

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            
            # Need to order the batch matmul's input tensor
            a_step = self.tensors[info['outputs'][0]]['shape'][2] / output_split
            b_step = self.tensors[info['outputs'][0]]['shape'][3] / output_split
            count = 0
            for c in self.split_tensor_table[outputs[0]]:
                a_offset = int(count // a_step)
                a = self.split_tensor_table[inputs[0]][a_offset]
                b_offset = int(count % b_step)
                b = self.split_tensor_table[inputs[1]][b_offset]
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a, b]
                new_op_info['outputs'] = [c]
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1
                count += 1

        # Concate the split output tensor & split again from here
        next_opid = self.nodes[opid].node.children[0]
        self.concat_than_split(next_opid, output_split, input_idx = 0)

    def split_conv(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) not in [2,3,4]:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            if(len(inputs) == 4):
                tokens = self.tensors[info['inputs'][3]]['name'].split('_')
                if(tokens[0] != 'padding'):
                    raise "wrong custom padding setting tensor data"
                paddings_H = int(tokens[1])
                paddings_W = int(tokens[2])
            else:
                # calculate padding
                if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                    total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
                    total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[2] - in_shape[2]
                    paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                    paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
                else:
                    paddings_H = 0
                    paddings_W = 0

            # generate splitted conv for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[1]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))

                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[0]][in_y])

                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)
                if (len(inputs) == 4):
                    new_op_info['inputs'][3] = padding_param_tensor
                    new_op_info['inputs'] += new_inputs
                else:
                    new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][4]

                # outputs
                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][int(out_y)//input_split]]

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_dwconv(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split, 1)

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) not in [2,3,4]:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            if(len(inputs) == 4):
                tokens = self.tensors[info['inputs'][3]]['name'].split('_')
                if(tokens[0] != 'padding'):
                    raise "wrong custom padding setting tensor data"
                paddings_H = int(tokens[1])
                paddings_W = int(tokens[2])
            else:
                # calculate padding
                if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                    total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
                    total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[2] - in_shape[2]
                    paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                    paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
                else:
                    paddings_H = 0
                    paddings_W = 0

            # generate splitted conv for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[1]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))

                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[0]][in_y])

                new_op_info = copy.deepcopy(new_op_info_base)
                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)

                if (len(inputs) == 4):
                    new_op_info['inputs'][3] = padding_param_tensor
                    new_op_info['inputs'] += new_inputs
                else:
                    new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][4]

                # outputs
                # for out_inner_y in range(guard_inner_y):
                    # new_op_info['outputs'].append(self.split_tensor_table[outputs[0]][out_y + out_inner_y])

                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][out_y//input_split]]

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_trconv(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split, 1)

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) != 3:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            tr_shape = self.tensors[inputs[0]]['shape']
            in_shape = self.tensors[inputs[2]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            # calculate padding
            if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                # total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
                # total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[2] - in_shape[2]
                total_padding_H = (in_shape[1] - 1) * stride_h + ker_shape[1] - out_shape[1]
                total_padding_W = (in_shape[2] - 1) * stride_w + ker_shape[2] - out_shape[2]
                paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
            else:
                paddings_H = 0
                paddings_W = 0

            # generate splitted trconv for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[1]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))

                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[2]][in_y])

                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)
                new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                #new_op_info['inputs'][0] = new_op_info['inputs'][5]

                # outputs
                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][int(out_y)//input_split]]

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_fullyconnected(self, opid, output_split):
        info = self.nodes[opid].node.info

        inputs = info['inputs']
        outputs = info['outputs']

        # Check fc's two origin inputs whether need splitted (by experience, it has the number in the name's last character)
        # TODO: need to fixed the need_split_nxn decide method
        need_split_nxn = True
        for input in inputs:
            # If the input is none, skip it
            if input == -1:
                continue
            input_name = self.tensors[input]['name']
            if not input_name[-1].isdigit():
                need_split_nxn = False
                break

        split_dim = self.nodes[opid].node.split_dim
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    if need_split_nxn:
                        self.split_tensor_by_nxn(info['outputs'][0], output_split, dim, dim + 1)
                    else:
                        self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            if need_split_nxn:
                self.split_tensor_by_nxn(info['outputs'][0], output_split, split_dim, split_dim + 1)
            else:
                self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        if len(inputs) != 3:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            if need_split_nxn:
                # Need to order the fullyconnected's input tensor
                a_step = self.tensors[info['outputs'][0]]['shape'][split_dim] / output_split
                b_step = self.tensors[info['outputs'][0]]['shape'][split_dim + 1] / output_split
                count = 0
                for c in self.split_tensor_table[outputs[0]]:
                    a_offset = int(count // a_step)
                    a = self.split_tensor_table[inputs[0]][a_offset]
                    b_offset = int(count % b_step)
                    b = self.split_tensor_table[inputs[1]][b_offset]
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['inputs'][1] = b
                    new_op_info['outputs'] = [c]
                    op = Node(new_op_info, split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1
                    count += 1
                
                # Concate the split output tensor & split again from here
                # If the child op have multi inputs(ex:pack), we need to assign the matched input idx to concat_then_split
                output_name = self.tensors[outputs[0]]['name']
                next_opid = self.nodes[opid].node.children[0]
                for i, input in enumerate(self.nodes[next_opid].node.info['inputs']):
                    if self.tensors[input]['name'] == output_name:
                        self.concat_than_split(next_opid, output_split, input_idx = i)
                        break
            else:
                for a, b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_max_pool(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = (info['builtin_options']['filter_height'], info['builtin_options']['filter_width'])
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            # calculate padding
            if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[0] - in_shape[1]
                total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[1] - in_shape[2]
                paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
            else:
                paddings_H = 0
                paddings_W = 0

            # generate splitted max_pool for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[0]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))

                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[0]][in_y])

                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)
                new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][2]

                # outputs
                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][int(out_y)//input_split]]

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_add(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = -1
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    split_dim = dim
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim
        
        # Consider the condition that input need to broadcast
        need_broadcast = False
        size1 = 1
        size2 = 1
        input1_shape = self.tensors[info['inputs'][0]]['shape']
        input2_shape = self.tensors[info['inputs'][1]]['shape']
        for dim in input1_shape:
            size1 *= dim
        for dim in input2_shape:
            size2 *= dim
        if size1 != size2:
            need_broadcast = True
        
        # Add with constant value, constant value also need to be splitted
        if len(self.split_tensor_table[info['inputs'][1]]) == 0:
            self.split_tensor_by_n(info['inputs'][1], output_split, split_dim)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not need_broadcast and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.nodes)
            # We don't split the constant value, if it need to be broadcast
            if need_broadcast:
                for a,b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1
            else:
                for a,b,c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_sub(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = -1
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    split_dim = dim
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim
    
        # Consider the condition that input need to broadcast
        need_broadcast = False
        size1 = 1
        size2 = 1
        input1_shape = self.tensors[info['inputs'][0]]['shape']
        input2_shape = self.tensors[info['inputs'][1]]['shape']
        for dim in input1_shape:
            size1 *= dim
        for dim in input2_shape:
            size2 *= dim
        if size1 != size2:
            need_broadcast = True

        # Sub with constant value, constant value also need to be splitted
        if len(self.split_tensor_table[info['inputs'][1]]) == 0:      
            self.split_tensor_by_n(info['inputs'][1], output_split, split_dim)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not need_broadcast and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            print(f"info: {info}")
            print(f"split number of two operand is not equal: {len(self.split_tensor_table[inputs[0]])} != {len(self.split_tensor_table[inputs[1]])}")
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.nodes)
            # We don't split the constant value, if it need to be broadcast
            if need_broadcast:
                for a,b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1
            else:
                for a,b,c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_mul(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = -1
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    split_dim = dim
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        # Consider the condition that input need to broadcast
        need_broadcast = False
        size1 = 1
        size2 = 1
        input1_shape = self.tensors[info['inputs'][0]]['shape']
        input2_shape = self.tensors[info['inputs'][1]]['shape']
        for dim in input1_shape:
            size1 *= dim
        for dim in input2_shape:
            size2 *= dim
        if size1 != size2:
            need_broadcast = True
        
        # Mul with constant value, constant value also need to be splitted
        if len(self.split_tensor_table[info['inputs'][1]]) == 0:
            self.split_tensor_by_n(info['inputs'][1], output_split, split_dim)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not need_broadcast and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.nodes)
            # We don't split the constant value, if it need to be broadcast
            if need_broadcast:
                for a,b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1
            else:
                for a,b,c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    op = Node(new_op_info,split_op_id)
                    self.nodes.append(SplitterNode(op))
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_logistic(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_leaky_relu(self, opid, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split, 1)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_mean(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_squared_difference(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.nodes)
            for a,b,c in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[inputs[1]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a,b]
                new_op_info['outputs'] = [c]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_rsqrt(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_gelu(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_quantize(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 1:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    # Need to consider whether it really can be splitted
    def split_resize_nearest_neighbor(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1
    
    def split_reduce_max(self, opid, output_split):
        info = self.nodes[opid].node.info
        axis_buffer = self.buffers[self.tensors[inputs[1]]['buffer']]
        axis = axis_buffer['data'][0]

        split_dim = self.nodes[opid].node.split_dim
        if axis < split_dim:
            split_dim -= 1
        
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    # For now, assume concat's input only have two tensors
    def split_concatenation(self, opid, output_split):
        info = self.nodes[opid].node.info
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            for idx, c in enumerate(self.split_tensor_table[outputs[0]]):
                new_op_input = []
                for input_idx in inputs:
                    new_op_input.append(self.split_tensor_table[input_idx][idx])
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = new_op_input
                new_op_info['outputs'] = [c]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_pack(self, opid, output_split):
        info = self.nodes[opid].node.info
        output_shape = self.tensors[info['outputs'][0]]['shape']
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(output_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(info['outputs'][0], output_split, dim)
                    break
        else:
            split_dim = self.nodes[opid].node.split_dim
            split_dim += 1
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            for child in self.nodes[opid].node.children:
                self.nodes[child].node.split_dim = split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        if len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            for c in self.split_tensor_table[outputs[0]]:
                new_op_info = copy.deepcopy(info)

                new_op_input = []
                for input_idx in range(len(inputs)):
                    for a in self.split_tensor_table[inputs[input_idx]]:
                        new_op_input.append(a)
                
                new_op_info['inputs'] = new_op_input
                new_op_info['outputs'] = [c]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_block_input(self, start_opid, input_split):
        info = self.nodes[start_opid].node.info
        # Depend on the input shape, split the input tensor
        inputs = info['inputs'][0]
        input_shape = self.tensors[inputs]['shape']
        
        if self.model_type == ModelType.BERT:
            split_dim_value = self.token_size
            for dim, dim_value in enumerate(input_shape):
                if dim_value == split_dim_value:
                    self.split_tensor_by_n(inputs, input_split, dim)
                    break
        else:
            # Split from te first dim that larger than 1
            for dim, dim_value in enumerate(input_shape):
                if dim_value > 1:
                    split_dim = dim
                    self.split_tensor_by_n(inputs, input_split, split_dim)
                    for child in self.nodes[start_opid].node.children:
                        self.nodes[child].node.split_dim = split_dim
                    break

        axis_tensor = {
            "shape": [

            ],
            "type": "INT32",
            "buffer": len(self.buffers),
            "name": self.tensors[info['inputs'][0]]['name']+"_split_axis_tensor",
            "quantization": {
            },

          }
        axis_buffer = {
            "data": self.int_list_to_byte_list([1])
        }
        self.tensors.append(axis_tensor)
        self.buffers.append(axis_buffer)

        outputs = copy.deepcopy(self.split_tensor_table[info['inputs'][0]])
        # outputs = [x for i,x in enumerate(outputs) if i % input_split == 0]

        new_op_info = {
            "opcode_index": self.get_opcode_index(49),
            "inputs": [len(self.tensors)-1, info['inputs'][0]],
            "outputs": outputs,
            "builtin_options_type": "SplitOptions",
            "builtin_options": {
                "num_splits" : len(outputs)
            }
        }
        self.new_operators.append(new_op_info)

    def concat_block_output(self, end_opid):
        info = self.nodes[end_opid].node.info
        new_op_info = {
            "opcode_index": self.get_opcode_index(2),
            "inputs": copy.deepcopy(self.split_tensor_table[info['inputs'][0]]),
            "outputs": info['outputs'],
            "builtin_options_type": "ConcatenationOptions",
            "builtin_options": {
                'axis': 1
            }
        }
        self.new_operators.append(new_op_info)

    # Parent op(multi output) -> concat -> split -> child op
    def concat_than_split(self, start_opid, output_split, input_idx):
        # child's info
        info = self.nodes[start_opid].node.info

        # First, create a concate node
        immediate_tensor = copy.deepcopy(self.tensors[info['inputs'][input_idx]])
        self.tensors.append(immediate_tensor)
        immediate_tensor_id = len(self.tensors) - 1
        concate_op_info = {
            "opcode_index": self.get_opcode_index(2),
            "inputs": copy.deepcopy(self.split_tensor_table[info['inputs'][input_idx]]),
            "outputs": [immediate_tensor_id],
            "builtin_options_type": "ConcatenationOptions",
            "builtin_options": {
                'axis': 1
            }
        }
        self.new_operators.append(concate_op_info)
        
        # Second, create a split node
        inputs = immediate_tensor_id
        input_shape = self.tensors[inputs]['shape']
        for dim, dim_value in enumerate(input_shape):
            if dim_value > 1:
                self.split_tensor_by_n(inputs, output_split, dim)
                break
        
        axis_tensor = {
            "shape": [],
            "type": "INT32",
            "buffer": len(self.buffers),
            "name": self.tensors[immediate_tensor_id]['name']+"_split_axis_tensor!!",
            "quantization": {},
          }
        axis_buffer = {
            "data": self.int_list_to_byte_list([1])
        }
        self.tensors.append(axis_tensor)
        self.buffers.append(axis_buffer)

        outputs = copy.deepcopy(self.split_tensor_table[immediate_tensor_id])
        split_op_info = {
            "opcode_index": self.get_opcode_index(49),
            "inputs": [immediate_tensor_id],
            "outputs": outputs,
            "builtin_options_type": "SplitOptions",
            "builtin_options": {
                "num_splits" : len(outputs)
            }
        }
        self.new_operators.append(split_op_info)

        # Final, update start_op's input
        info['inputs'][input_idx] = immediate_tensor_id

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'

    def get_padding_param_tensor(self, padTop, padSide):
        if (padTop not in self.padding_param_tensors or
            padSide not in self.padding_param_tensors[padTop]):
            new_buffer_info = {
                "data": self.int_list_to_byte_list([padTop, padSide])
            }
            new_tensor_info = {
                "shape": [
                    2
                ],
                "type": "INT32",
                "buffer": len(self.buffers),
                "name": "padding_%d_%d" % (padTop, padSide),

                "quantization": {
                }
            }

            if padTop not in self.padding_param_tensors:
                self.padding_param_tensors[padTop] = {padSide:len(self.tensors)}
            else:
                self.padding_param_tensors[padTop][padSide] = len(self.tensors)

            self.buffers.append(new_buffer_info)
            self.tensors.append(new_tensor_info)

        return self.padding_param_tensors[padTop][padSide]

    def PaddingFusion(self):
        def get_pad_param(pad_data):
            byte_data = bytes(pad_data)
            int_data = [ int.from_bytes(byte_data[4*i:4*i+4], byteorder='little') for i in range(len(byte_data)//4)]
            if False in [int_data[i] == 0 for i in [0,1,6,7]]:
                raise "pad in N or C"
            elif int_data[2]!=int_data[3] or int_data[4]!=int_data[5]:
                raise "asymmetric pad in H or W"
            return int_data[2], int_data[4]


        def apply_fusion(pad_opid, conv_opid):
            pad_op_info = self.ori_graph.ops[pad_opid].info
            conv_op_info = self.ori_graph.ops[conv_opid].info
            pad_H, pad_W = get_pad_param(self.buffers[self.tensors[pad_op_info['inputs'][1]]['buffer']]['data'])

            inputs = conv_op_info['inputs']
            outputs = conv_op_info['outputs']
            in_shape = self.tensors[pad_op_info['inputs'][0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = conv_op_info['builtin_options']['stride_h']
            total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
            total_padding_W = (out_shape[2] - 1) * stride_h + ker_shape[2] - in_shape[2]
            calculated_padding_H = total_padding_H // 2 if total_padding_H > 0 else 0
            calculated_padding_W = total_padding_W// 2 if total_padding_W > 0 else 0

            conv_op_info['inputs'][0] = pad_op_info['inputs'][0]
            if calculated_padding_H==pad_H and calculated_padding_W==pad_W:
                conv_op_info['builtin_options']['padding'] = 'SAME'
            elif len(conv_op_info['inputs']) == 3:
                conv_op_info['inputs'].append(self.get_padding_param_tensor(pad_H,pad_W))
            elif len(conv_op_info['inputs']) < 3:
                 BaseException("worng inputs format: length < 3")
            elif len(conv_op_info['inputs']) > 3:
                 BaseException("overriding padding param already exist")



        def PaddingFusion_dfs(cur_opid, visited, deprecated):
            # check visited
            if visited[cur_opid]:
                return deprecated
            visited[cur_opid] = True

            # is cur_op a pad op?
            is_pad = self.opcodes[self.ori_graph.ops[cur_opid].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) == 34

            # visit all children
            is_deprecated = 0
            need_fusion = []
            for child_id in self.ori_graph.ops[cur_opid].children:
                if(is_pad and self.opcodes[self.ori_graph.ops[child_id].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) in [3, 4]):
                    need_fusion.append(child_id)
                    is_deprecated = 1 if is_deprecated == 0 else -1
                else:
                    is_deprecated = -1
                PaddingFusion_dfs(child_id, visited, deprecated)


            if is_deprecated == 1:
                deprecated.append(cur_opid)
                for opid in need_fusion:
                    apply_fusion(cur_opid, opid)

            return deprecated


        visited = [False for _ in range(len(self.ori_graph.ops))]
        deprecated = []
        while 1 :
            deprecated_new = []
            PaddingFusion_dfs(self.ori_graph.root_op_id, visited, deprecated_new)
            # TODO: remove deprecated operators
            if len(deprecated_new) == 0:
                break
            for opid in deprecated_new:
                if opid == self.ori_graph.root_op_id:
                    self.ori_graph.root_op_id = self.ori_graph.ops[opid].children[0]

                if len(self.ori_graph.ops[opid].parents) > 0:
                    parent = self.ori_graph.ops[opid].parents[0]
                    for i in range(len(self.ori_graph.ops[parent].children)):
                        if self.ori_graph.ops[parent].children[i] == opid:
                            del self.ori_graph.ops[parent].children[i]
                            self.ori_graph.ops[parent].children += self.ori_graph.ops[opid].children
                            break
                    for child in self.ori_graph.ops[opid].children:
                        for i in range(len(self.ori_graph.ops[child].children)):
                            if self.ori_graph.ops[child].parents[i] == opid:
                                self.ori_graph.ops[child].parents[i] = parent
                                break
                else:
                    for child in self.ori_graph.ops[opid].children:
                        for i in range(len(self.ori_graph.ops[child].children)):
                            if self.ori_graph.ops[child].parents[i] == opid:
                                del self.ori_graph.ops[child].parents[i]
                                break
            deprecated += deprecated_new
        self.ori_graph.buffers = self.buffers
        self.ori_graph.tensors = self.tensors
        self.ori_graph.remove_deprecated_op(deprecated)
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.ori_graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, self.ori_graph.exec_order)
        self.re_init(new_graph)

    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length=4, byteorder='little')]
        return out