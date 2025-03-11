from collections import defaultdict
from MyGraph import Node,Graph
import numpy as np
import copy

# bert prefer split token
# CNN prefer split height
class ModelType:
    BERT = 0
    CNN = 1

class SplitterNode:
    def __init__(self, node:Node):
        self.node = node
        self.split_id = []
        self.visited = False
        # For BERT model, we need to avoid the split on the node which on the path betwween the V and til the FC
        self.avoid_split = False

class Splitter:
    def __init__(self,ori_graph:Graph, split_height:int, model_type:int, token_size:int):
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
        # For BERT model, let the token size decided by user
        self.token_size = token_size
        # same_layer_next_opids[x] = (#ops in layer, x's head_opid, x's next_opid, #ops in layer) 
        self.same_layer_next_opids = defaultdict(list)
        for i, opcode in enumerate(self.opcodes):
            # 0: ADD
            # 2: CONCATENATION
            # 3: CONV_2D
            # 4: DEPTHWISE_CONV_2D
            # 6: DEQUANTIZE
            # 9: FULLY_CONNECTED
            # 14: LOGISTIC
            # 17: MAX_POOL_2D
            # 18: MUL
            # 22: RESHAPE
            # 25: SOFTMAX
            # 28: TANH
            # 34: PAD
            # 39: TRANSPOSE
            # 40: MEAN
            # 41: SUB
            # 42: DIV
            # 49: SPLIT
            # 65: SLICE
            # 76: RSQRT
            # 78: POW
            # 82: REDUCE_MAX
            # 83: PACK
            # 97: RESIZE_NEAREST_NEIGHBOR
            # 99: SQUARED_DIFFERENCE
            # 102: SPLIT_V
            # 114: QUANTIZE
            # 126: BATCH_MATMUL
            # 127: GELU
            split_candidate = [0, 2, 3, 4, 6, 9, 14, 17, 18, 22, 25, 28, 34, 39, 40, 41, 42, 49, 65, 76, 78, 82, 83, 97, 99, 102, 114, 126, 127]
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
            # See the corresponding number in the __init__ function
            split_candidate = [0, 2, 3, 4, 6, 9, 14, 17, 18, 22, 25, 28, 34, 39, 40, 41, 42, 49, 65, 76, 78, 82, 83, 97, 99, 102, 114, 126, 127]
            if opcode.get("deprecated_builtin_code", 0) in split_candidate:
                self.splittable_opcode_idxes[opcode.get("deprecated_builtin_code", 0)] = i

    def perform_split(self) -> Graph:
        self.split_tensor_table = defaultdict(list)
        self.new_operators = []

        input_tile_size = self.split_height
        output_tile_size = self.split_height

        # Currently assume it's splittable from the root of the graph
        splittables = []
        end_ids = []
        # Fisrt, traverse from the multi inputs (Except the last input)
        if len(self.ori_graph.root_op_ids) > 1:
            for root_op_id in self.ori_graph.root_op_ids[: -1]:
                start_id = self.traverse_til_splittable(root_op_id)
                if start_id is not None:
                    self.split_block_input(start_id, input_tile_size)
                    self.traverse_til_not_splittable(start_id, splittables, end_ids)
        
        # Then, traverse from the last input
        start_id = self.traverse_til_splittable(self.ori_graph.root_op_ids[-1])
        # Get splittable block
        self.traverse_til_not_splittable(start_id, splittables, end_ids)
        # Add non-visited op into new_operators
        for end_id in end_ids:
            self.traverse_til_end(end_id)
        # Split block input
        self.split_block_input(start_id, input_tile_size)
        # Start split
        for op in splittables:
            self.split_one_node(op, input_tile_size, output_tile_size)
        # Concat block output
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
            self.traverse_til_end(child)
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
            # Check if it is the reshape op
            opcode_index = self.nodes[child].node.info.get("opcode_index")
            opcode_type = self.ori_graph.opcodes[opcode_index].get("builtin_code")
            if self.model_type != ModelType.BERT and opcode_type == "RESHAPE":
                # Check if it is splittable op
                input_shape = copy.deepcopy(self.tensors[self.nodes[child].node.info['inputs'][0]]['shape'])
                output_shape = copy.deepcopy(self.tensors[self.nodes[child].node.info['outputs'][0]]['shape'])
                # For now, we check whether the input shape have at least two dimentions that equal to the output shape
                # If not, we assume it is non-splittable op
                count = 0
                for i in range(1, len(input_shape)):
                    for j in range(1, len(output_shape)):
                        if input_shape[i] == output_shape[j]:
                            count += 1
                            output_shape[j] = -1
                            break
                if count < 2:
                    if child not in end_ids:
                        end_ids.append(child)
                    return None
                
            # Check if mean's keep_dims is True (if not, it is non-splittable op)
            if opcode_type == "MEAN":
                info = self.nodes[child].node.info
                input_shape = self.tensors[info['inputs'][0]]['shape']
                output_shape = self.tensors[info['outputs'][0]]['shape']
                if len(input_shape) != len(output_shape):
                    if child not in end_ids:
                        end_ids.append(child)
                    return None

            # Check if it is splittable op
            if self.nodes[child].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
                self.traverse_til_not_splittable(child, splittables, end_ids)
            else:
                end_ids.append(child)
        
        # To avoid the model with zero splittable op
        if len(self.nodes[current_opid].node.children) == 0:
            # Last op no need to split
            splittables.pop(-1)
            end_ids.append(current_opid)
        return None
    
    # Not use now, idea is prepare for block_based option use (option had been removed in commit bd7874d36eea8e3966b9fbb83e743cbf548a5713)
    def traverse_til_not_splittable_with_end_id(self, current_opid, splittables, end_id, end_ids):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None
            
        self.nodes[current_opid].visited = True
        if current_opid not in splittables:
            splittables.append(current_opid)

        # Traverse to the end_id
        if current_opid == end_id:
            # Different to the traverse_til_not_splittable, it keep the end_id in splittables
            return True
        
        for child in self.nodes[current_opid].node.children:
            # Check if it is the reshape op
            opcode_index = self.nodes[child].node.info.get("opcode_index")
            opcode_type = self.ori_graph.opcodes[opcode_index].get("builtin_code")
            if self.model_type != ModelType.BERT and opcode_type == "RESHAPE":
                # Check if it is splittable op
                input_shape = copy.deepcopy(self.tensors[self.nodes[child].node.info['inputs'][0]]['shape'])
                output_shape = copy.deepcopy(self.tensors[self.nodes[child].node.info['outputs'][0]]['shape'])
                # For now, we check whether the input shape have at least two dimentions that equal to the output shape
                # If not, we assume it is non-splittable op
                count = 0
                for i in range(1, len(input_shape)):
                    for j in range(1, len(output_shape)):
                        if input_shape[i] == output_shape[j]:
                            count += 1
                            output_shape[j] = -1
                            break
                if count < 2:
                    if child not in end_ids:
                        end_ids.append(child)
                    return False
            
            # check if it is splittable op
            if self.nodes[child].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
                til_block_end = self.traverse_til_not_splittable_with_end_id(child, splittables, end_id, end_ids)
        return til_block_end

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
        shape = tensor_info.get("shape", [1])
        if shape == []:
            shape = [1]
        # Reshape to the original shape
        np_arr = one_dim_arr.reshape(shape)

        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        if tile_dim >= len(shape):
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
                self.split_tensor_table[tensor_id_in].append(tensor_id)
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
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim1] / tile_size), 1):
            guard1 = min(tile_size, tensor_info['shape'][tile_dim1] - i * tile_size)
            for j in range(0, math.ceil(tensor_info['shape'][tile_dim2] / tile_size), 1):
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

    # This function will generate (size[tile_dim]/tile_size) tensors but with the same tensor info
    def split_tensor_by_n_with_same_info(self, tensor_id_in, tile_size, tile_dim):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        for i in range(0, math.ceil(tensor_info['shape'][tile_dim] / tile_size), 1):
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_split_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1

    # Set the ops on the path from V to pack(which bert model usually perform multi-head self attention) with avoid_split
    # TODO: may need to handle the case that the pack is not the last op
    def set_flow_with_avoid_split(self, opid):
        opcode_index = self.nodes[opid].node.info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type == "PACK":
            return
        self.nodes[opid].avoid_split = True
        for child in self.nodes[opid].node.children:
            self.set_flow_with_avoid_split(child)

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
        elif opcode_idx == self.splittable_opcode_idxes.get(6, -1):
            self.split_dequantize(opid, output_split)
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
        elif opcode_idx == self.splittable_opcode_idxes.get(28, -1):
            self.split_tanh(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(34, -1):
            self.split_pad(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(39, -1):
            self.split_transpose(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(40, -1):
            self.split_mean(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(41, -1):
            self.split_sub(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(42, -1):
            self.split_div(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(49, -1):
            self.split_split(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(65, -1):
            self.split_slice(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(76, -1):
            self.split_rsqrt(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(78, -1):
            self.split_pow(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(82, -1):
            self.split_reduce_max(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(83, -1):
            self.split_pack(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(97, -1):
            self.split_resize_nearest_neighbor(opid, output_split)
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

        # Record the split opid in the same layer
        pre_opid = self.nodes[opid].split_id[0]
        head_opid = self.nodes[opid].split_id[0]
        for now_opid in self.nodes[opid].split_id[1:]:
            self.same_layer_next_opids[pre_opid] = [head_opid, now_opid, len(self.nodes[opid].split_id)]
            pre_opid = now_opid
        last_opid = self.nodes[opid].split_id[-1]
        self.same_layer_next_opids[last_opid] = [head_opid, -1, len(self.nodes[opid].split_id)]

    def split_pad(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
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
            split_op_id = len(self.new_operators)
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
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    # We follow the number of split input tensor to split the output tensor
    # For now, not support the split_dim decomposed to multiple dimension
    def split_reshape(self, opid, output_split):
        info = self.nodes[opid].node.info
        input = info['inputs'][0]
        output = info['outputs'][0]
        input_shape = self.tensors[input]['shape']
        output_shape = self.tensors[output]['shape']

        # Keep track the split_dim
        # Not support the split_dim decomposed to multiple dimension or composed to one dimension
        split_dim = self.nodes[opid].node.split_dim
        tmp_value = 1
        tmp_idx = 0
        if len(output_shape) > len(input_shape):
            for idx, dim_value in enumerate(output_shape):
                tmp_value *= dim_value
                if tmp_idx >= len(input_shape):
                    raise f"out of range happen in split_reshape, reshape info: {info}"
                if tmp_value == input_shape[tmp_idx]:
                    if tmp_idx == split_dim:
                        split_dim = idx
                        for child in self.nodes[opid].node.children:
                            self.nodes[child].node.split_dim = split_dim
                        break
                    tmp_idx += 1
                    tmp_value = 1
        elif len(output_shape) < len(input_shape):
            for idx, dim_value in enumerate(input_shape):
                tmp_value *= dim_value
                if tmp_idx >= len(output_shape):
                    raise f"out of range happen in split_reshape, reshape info: {info}"
                if tmp_value == output_shape[tmp_idx]:
                    if idx == split_dim:
                        split_dim = tmp_idx
                        for child in self.nodes[opid].node.children:
                            self.nodes[child].node.split_dim = split_dim
                        break
                    tmp_idx += 1
                    tmp_value = 1
        # Special case happen in the DistilGPT2 model
        elif len(output_shape) == len(input_shape):
            split_dim_val = input_shape[split_dim]
            for idx, dim_value in enumerate(output_shape):
                if dim_value == split_dim_val:
                    split_dim = idx
                    for child in self.nodes[opid].node.children:
                        self.nodes[child].node.split_dim = split_dim
                    break

        # Since QK and V's GEMM will take V's token dimension as product dimension, which should not be splitted
        # So we perform split_tensor_by_n_with_same_info to keep the size of the splitted tensor
        output_name = self.tensors[output]['name']
        # Check this reshape is from V's output
        is_value_reshape = False
        if 'attn/Reshape_2' in output_name:
            is_value_reshape = True
            self.set_flow_with_avoid_split(opid)
        if is_value_reshape:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        elif self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
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
            split_op_id = len(self.new_operators)
            if is_value_reshape:
                parent_opid = self.nodes[opid].node.parents[0]
                immediate_tensor = copy.deepcopy(self.tensors[inputs[0]])
                self.tensors.append(immediate_tensor)
                immediate_tensor_id = len(self.tensors) - 1
                # Add a concate op before the reshape op
                concate_op_info = {
                    "opcode_index": self.get_opcode_index(2),
                    "inputs": copy.deepcopy(self.split_tensor_table[inputs[0]]),
                    "outputs": [immediate_tensor_id],
                    "builtin_options_type": "ConcatenationOptions",
                    "builtin_options": {
                        "axis": self.nodes[parent_opid].node.split_dim
                    }
                }
                self.new_operators.append(concate_op_info)
            
                for a in self.split_tensor_table[outputs[0]]:
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = immediate_tensor_id
                    new_op_info['outputs'] = [a]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1
            else:
                for a, b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1

    def split_transpose(self, opid, output_split):
        info = self.nodes[opid].node.info
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
                if self.nodes[opid].avoid_split:
                    self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
                else:
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
            split_op_id = len(self.new_operators)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_softmax(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_split(self, opid, output_split):
        info = self.nodes[opid].node.info
        inputs = info['inputs']
        outputs = info['outputs']

        # Some case: <1, 128, 32> -> split_dim = 1 -> <128, 16> break our split_dim tracking
        input_shape = self.tensors[inputs[1]]['shape']
        output_shape = self.tensors[outputs[0]]['shape']
        split_dim_value = input_shape[self.nodes[opid].node.split_dim]
        find_split_dim = False
        for i in range(len(output_shape)):
            if output_shape[i] == split_dim_value:
                self.nodes[opid].node.split_dim = i
                find_split_dim = True
                break
        if not find_split_dim:
            raise "Can't split the dim which is split op's split axis"

        split_dim = self.nodes[opid].node.split_dim
        # split have multi outputs, each output need to be split
        num_splits = info['builtin_options']['num_splits']
        for i in range(num_splits):
            if self.nodes[opid].avoid_split:
                self.split_tensor_by_n_with_same_info(info['outputs'][i], output_split, split_dim)
            else:
                self.split_tensor_by_n(info['outputs'][i], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != num_splits:
            raise "wrong output number"
        else:
            split_op_id = len(self.new_operators)
            for idx1, a in enumerate(self.split_tensor_table[inputs[1]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][1] = a
                new_op_output = []
                for idx2 in range(num_splits):
                    b = self.split_tensor_table[outputs[idx2]][idx1]
                    new_op_output.append(b)
                new_op_info['outputs'] = new_op_output
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_split_v(self, opid, output_split):
        info = self.nodes[opid].node.info
        inputs = info['inputs']
        outputs = info['outputs']

        size_splits = info['builtin_options']['num_splits']

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
            split_op_id = len(self.new_operators)
            for idx1, a in enumerate(self.split_tensor_table[inputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_output = []
                for idx2 in range(size_splits):
                    b = self.split_tensor_table[outputs[idx2]][idx1]
                    new_op_output.append(b)
                new_op_info['outputs'] = new_op_output
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_slice(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        if self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']

        input_shape = self.tensors[inputs[0]]['shape']
        
        size_buffer_id = self.tensors[inputs[2]]['buffer']
        size_buffer = self.buffers[size_buffer_id]
        # Process buffer structure: [a, 0, 0, 0, b, 0, 0, 0, c, 0, 0, 0, d, 0, 0, 0]
        if size_buffer['data'][split_dim * 4] != input_shape[split_dim]:
            raise "split_dim size not equal to input size"
        
        for output_idx in range(len(outputs)):
            self.split_tensor_by_n(info['outputs'][output_idx], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        if len(inputs) != 3:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.new_operators)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                # Change split_dim's size to output_split
                input2_buffer = self.buffers[size_buffer_id]
                input2_buffer['data'][split_dim * 4] = output_split
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

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
            split_op_id = len(self.new_operators)
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
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1
                count += 1

        # Concate the split output tensor & split again from here
        next_opid = self.nodes[opid].node.children[0]
        self.concat_than_split(opid, next_opid, output_split, input_idx = 0)

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
            split_op_id = len(self.new_operators)
            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            # Had perform padding fusion
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
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_dwconv(self, opid, input_split, output_split):
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
            split_op_id = len(self.new_operators)
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
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_fullyconnected(self, opid, output_split):
        info = self.nodes[opid].node.info

        inputs = info['inputs']
        outputs = info['outputs']

        # Check fc's weight whether had been splitted
        need_split_nxn = True
        if(self.split_tensor_table.get(inputs[1]) == None):
            need_split_nxn = False

        if self.model_type == ModelType.BERT:
            # Handle bert's split_dim is very annoying (input & weight may have diff shape -> break our split_dim tracking), use special case to handle
            output_shape = self.tensors[info['outputs'][0]]['shape']
            for i in range(len(output_shape)):
                if output_shape[i] > 1:
                    self.nodes[opid].node.split_dim = i
                    split_dim = i
                    break
            # Since QK and V's GEMM will take V's token dimension as product dimension, which should not be splitted
            # So we perform split_tensor_by_n_with_same_info to keep the size of the splitted tensor
            output_name = self.tensors[info['outputs'][0]]['name']
            # Check this FC is compute the V
            is_value_fc = False
            if 'value' in output_name:
                is_value_fc = True
                self.set_flow_with_avoid_split(opid)
            if is_value_fc:
                self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
            # Encount the QKV's FC, only split the QK part
            elif self.nodes[opid].avoid_split:
                self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
            elif need_split_nxn:
                self.split_tensor_by_nxn(info['outputs'][0], output_split, split_dim, split_dim + 1)
            else:
                self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        else:
            split_dim = self.nodes[opid].node.split_dim
            is_value_fc = False
            if need_split_nxn:
                self.split_tensor_by_nxn(info['outputs'][0], output_split, split_dim, split_dim + 1)
            else:
                self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = split_dim

        if len(inputs) != 3:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.new_operators)
            if is_value_fc:
                # For V's FC, we decide to avoid split, perform fake concatenate
                for a in self.split_tensor_table[outputs[0]]:
                    new_op_info = copy.deepcopy(info)
                    for b in self.split_tensor_table[inputs[0]]:
                        new_op_info['inputs'].append(b)
                    new_op_info['inputs'][0] = self.split_tensor_table[inputs[0]][0]
                    new_op_info['outputs'] = [a]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1
            elif self.nodes[opid].avoid_split and need_split_nxn:
                for a, b, c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['inputs'][1] = b
                    new_op_info['outputs'] = [c]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1
            elif need_split_nxn:
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
                        self.concat_than_split(opid, next_opid, output_split, input_idx = i)
                        break
            else:
                for a, b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1

    def split_max_pool(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) not in [1,2]:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.new_operators)
            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = (info['builtin_options']['filter_height'], info['builtin_options']['filter_width'])
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            # calculate padding
            if(len(inputs) == 2):
                tokens = self.tensors[info['inputs'][1]]['name'].split('_')
                if(tokens[0] != 'padding'):
                    raise "wrong custom padding setting tensor data"
                paddings_H = int(tokens[1])
                paddings_W = int(tokens[2])
            else:
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
                if(len(inputs) == 2):
                    new_op_info['inputs'][1] = padding_param_tensor
                    new_op_info['inputs'] += new_inputs
                else:
                    new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][2]

                # outputs
                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][int(out_y)//input_split]]

                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_add(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        if self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim
        
        # Add with constant value, constant value may also need to be splitted
        have_constant = False
        input1_is_constant = False
        input2_is_constant = False
        need_broadcast = False
        input1_need_broadcast = False
        input2_need_broadcast = False
        if len(self.buffers[self.tensors[info['inputs'][0]]['buffer']]) != 0:
            input1_is_constant = True
            if self.split_tensor_table[info['inputs'][0]] == []:
                self.split_constant_tensor_by_n(info['inputs'][0], output_split, split_dim)
        if len(self.buffers[self.tensors[info['inputs'][1]]['buffer']]) != 0:
            input2_is_constant = True
            if self.split_tensor_table[info['inputs'][1]] == []:
                self.split_constant_tensor_by_n(info['inputs'][1], output_split, split_dim)
        have_constant = input1_is_constant or input2_is_constant
        if have_constant:
            input1_split_len = len(self.split_tensor_table[info['inputs'][0]])
            input2_split_len = len(self.split_tensor_table[info['inputs'][1]])
            if input1_split_len != input2_split_len:
                input1_need_broadcast = True if input1_split_len == 1 else False
                input2_need_broadcast = True if input2_split_len == 1 else False
                if not input1_need_broadcast and not input2_need_broadcast:
                    raise BaseException("In add op, split number of two operand is not equal and the min split number is not 1")
                need_broadcast = True

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not have_constant and len(self.tensors[info['inputs'][0]]['shape']) != len(self.tensors[info['inputs'][1]]['shape']):
            raise BaseException("not support different dim of two operand")
        elif not have_constant and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.new_operators)
            # We don't split the constant value, if it need to be broadcast
            if have_constant and need_broadcast:
                if input1_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[1]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = self.split_tensor_table[inputs[0]][0]
                        new_op_info['inputs'][1] = a
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
                elif input2_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[0]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = a
                        new_op_info['inputs'][1] = self.split_tensor_table[inputs[1]][0]
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
            else:
                for a, b, c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_sub(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        if self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim

        # Sub with constant value, constant value may also need to be splitted
        have_constant = False
        input1_is_constant = False
        input2_is_constant = False
        need_broadcast = False
        input1_need_broadcast = False
        input2_need_broadcast = False
        if len(self.buffers[self.tensors[info['inputs'][0]]['buffer']]) != 0:
            input1_is_constant = True
            if self.split_tensor_table[info['inputs'][0]] == []:
                self.split_constant_tensor_by_n(info['inputs'][0], output_split, split_dim)
        if len(self.buffers[self.tensors[info['inputs'][1]]['buffer']]) != 0:
            input2_is_constant = True
            if self.split_tensor_table[info['inputs'][1]] == []:
                self.split_constant_tensor_by_n(info['inputs'][1], output_split, split_dim)
        have_constant = input1_is_constant or input2_is_constant
        if have_constant:
            input1_split_len = len(self.split_tensor_table[info['inputs'][0]])
            input2_split_len = len(self.split_tensor_table[info['inputs'][1]])
            if input1_split_len != input2_split_len:
                input1_need_broadcast = True if input1_split_len == 1 else False
                input2_need_broadcast = True if input2_split_len == 1 else False
                if not input1_need_broadcast and not input2_need_broadcast:
                    raise BaseException("In sub op, split number of two operand is not equal and the min split number is not 1")
                need_broadcast = True

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not have_constant and len(self.tensors[info['inputs'][0]]['shape']) != len(self.tensors[info['inputs'][1]]['shape']):
            raise BaseException("not support different dim of two operand")
        elif not have_constant and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.new_operators)
            # We don't split the constant value, if it need to be broadcast
            if have_constant and need_broadcast:
                if input1_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[1]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = self.split_tensor_table[inputs[0]][0]
                        new_op_info['inputs'][1] = a
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
                elif input2_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[0]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = a
                        new_op_info['inputs'][1] = self.split_tensor_table[inputs[1]][0]
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
            else:
                for a, b, c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_mul(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        if self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim
        
        # Mul with constant value, constant value may also need to be splitted
        have_constant = False
        input1_is_constant = False
        input2_is_constant = False
        need_broadcast = False
        input1_need_broadcast = False
        input2_need_broadcast = False
        if len(self.buffers[self.tensors[info['inputs'][0]]['buffer']]) != 0:
            input1_is_constant = True
            if self.split_tensor_table[info['inputs'][0]] == []:
                self.split_constant_tensor_by_n(info['inputs'][0], output_split, split_dim)
        if len(self.buffers[self.tensors[info['inputs'][1]]['buffer']]) != 0:
            input2_is_constant = True
            if self.split_tensor_table[info['inputs'][1]] == []:
                self.split_constant_tensor_by_n(info['inputs'][1], output_split, split_dim)
        have_constant = input1_is_constant or input2_is_constant
        if have_constant:
            input1_split_len = len(self.split_tensor_table[info['inputs'][0]])
            input2_split_len = len(self.split_tensor_table[info['inputs'][1]])
            if input1_split_len != input2_split_len:
                input1_need_broadcast = True if input1_split_len == 1 else False
                input2_need_broadcast = True if input2_split_len == 1 else False
                if not input1_need_broadcast and not input2_need_broadcast:
                    raise BaseException("In mul op, split number of two operand is not equal and the min split number is not 1")
                need_broadcast = True

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not have_constant and len(self.tensors[info['inputs'][0]]['shape']) != len(self.tensors[info['inputs'][1]]['shape']):
            raise BaseException("not support different dim of two operand")
        elif not have_constant and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.new_operators)
            # We don't split the constant value, if it need to be broadcast
            if have_constant and need_broadcast:
                if input1_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[1]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = self.split_tensor_table[inputs[0]][0]
                        new_op_info['inputs'][1] = a
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
                elif input2_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[0]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = a
                        new_op_info['inputs'][1] = self.split_tensor_table[inputs[1]][0]
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
            else:
                for a, b, c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id+=1

    def split_div(self, opid, output_split):
        info = self.nodes[opid].node.info

        split_dim = self.nodes[opid].node.split_dim
        if self.nodes[opid].avoid_split:
            self.split_tensor_by_n_with_same_info(info['outputs'][0], output_split, split_dim)
        else:
            self.split_tensor_by_n(info['outputs'][0], output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = self.nodes[opid].node.split_dim
        
        # Mul with constant value, constant value may also need to be splitted
        have_constant = False
        input1_is_constant = False
        input2_is_constant = False
        need_broadcast = False
        input1_need_broadcast = False
        input2_need_broadcast = False
        if len(self.buffers[self.tensors[info['inputs'][0]]['buffer']]) != 0:
            input1_is_constant = True
            if self.split_tensor_table[info['inputs'][0]] == []:
                self.split_constant_tensor_by_n(info['inputs'][0], output_split, split_dim)
        if len(self.buffers[self.tensors[info['inputs'][1]]['buffer']]) != 0:
            input2_is_constant = True
            if self.split_tensor_table[info['inputs'][1]] == []:
                self.split_constant_tensor_by_n(info['inputs'][1], output_split, split_dim)
        have_constant = input1_is_constant or input2_is_constant
        if have_constant:
            input1_split_len = len(self.split_tensor_table[info['inputs'][0]])
            input2_split_len = len(self.split_tensor_table[info['inputs'][1]])
            if input1_split_len != input2_split_len:
                input1_need_broadcast = True if input1_split_len == 1 else False
                input2_need_broadcast = True if input2_split_len == 1 else False
                if not input1_need_broadcast and not input2_need_broadcast:
                    raise BaseException("In mul op, split number of two operand is not equal and the min split number is not 1")
                need_broadcast = True

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif not have_constant and len(self.tensors[info['inputs'][0]]['shape']) != len(self.tensors[info['inputs'][1]]['shape']):
            raise BaseException("not support different dim of two operand")
        elif not have_constant and len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.new_operators)
            # We don't split the constant value, if it need to be broadcast
            if have_constant and need_broadcast:
                if input1_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[1]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = self.split_tensor_table[inputs[0]][0]
                        new_op_info['inputs'][1] = a
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
                elif input2_need_broadcast:
                    for a, b in zip(self.split_tensor_table[inputs[0]],
                                    self.split_tensor_table[outputs[0]]):
                        new_op_info = copy.deepcopy(info)
                        new_op_info['inputs'][0] = a
                        new_op_info['inputs'][1] = self.split_tensor_table[inputs[1]][0]
                        new_op_info['outputs'] = [b]
                        self.new_operators.append(new_op_info)
                        self.nodes[opid].split_id.append(split_op_id)
                        split_op_id+=1
            else:
                for a, b, c in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[inputs[1]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'] = [a,b]
                    new_op_info['outputs'] = [c]
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_mean(self, opid, output_split):
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
            # Check if mean's reduce axis have split_dim, if so, we need to concat the split tensor first
            axis_buffer = self.buffers[self.tensors[info['inputs'][1]]['buffer']]
            if split_dim in axis_buffer['data']:
                intermediate_tensor = copy.deepcopy(self.tensors[info['inputs'][0]])
                self.tensors.append(intermediate_tensor)
                intermediate_tensor_id = len(self.tensors) - 1
                concate_op_info = {
                    "opcode_index": self.get_opcode_index(2),
                    "inputs": copy.deepcopy(self.split_tensor_table[info['inputs'][0]]),
                    "outputs": [intermediate_tensor_id],
                    "builtin_options_type": "ConcatenationOptions",
                    "builtin_options": {
                        'axis': split_dim
                    }
                }
                self.new_operators.append(concate_op_info)

                split_op_id = len(self.new_operators)
                for a in self.split_tensor_table[outputs[0]]:
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = intermediate_tensor_id
                    new_op_info['outputs'] = [a]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1
            else:
                split_op_id = len(self.new_operators)
                for a,b in zip(self.split_tensor_table[inputs[0]],
                                self.split_tensor_table[outputs[0]]):
                    new_op_info = copy.deepcopy(info)
                    new_op_info['inputs'][0] = a
                    new_op_info['outputs'] = [b]
                    self.new_operators.append(new_op_info)
                    self.nodes[opid].split_id.append(split_op_id)
                    split_op_id += 1

    def split_squared_difference(self, opid, output_split):
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
        elif len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.new_operators)
            for a,b,c in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[inputs[1]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a,b]
                new_op_info['outputs'] = [c]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_rsqrt(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_gelu(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_pow(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_tanh(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a, b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_dequantize(self, opid, output_split):
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a]
                new_op_info['outputs'] = [b]
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1
    
    def split_reduce_max(self, opid, output_split):
        info = self.nodes[opid].node.info
        axis_buffer = self.buffers[self.tensors[info['inputs'][1]]['buffer']]
        axis = axis_buffer['data'][0]

        split_dim = self.nodes[opid].node.split_dim
        if axis == split_dim:
            raise f"reduce_max can't split on the axis {axis}, it will produces incorrect results"
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
            split_op_id = len(self.new_operators)
            for a,b in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'][0] = a
                new_op_info['outputs'] = [b]
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
            split_op_id = len(self.new_operators)
            for idx, c in enumerate(self.split_tensor_table[outputs[0]]):
                new_op_input = []
                for input_idx in inputs:
                    new_op_input.append(self.split_tensor_table[input_idx][idx])
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = new_op_input
                new_op_info['outputs'] = [c]
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_pack(self, opid, output_split):
        info = self.nodes[opid].node.info

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
            split_op_id = len(self.new_operators)
            for idx, c in enumerate(self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)

                new_op_input = []
                for input_idx in inputs:
                    new_op_input.append(self.split_tensor_table[input_idx][idx])
                
                new_op_info['inputs'] = new_op_input
                new_op_info['outputs'] = [c]
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
                    split_dim = dim
                    self.split_tensor_by_n(inputs, input_split, split_dim)
                    self.nodes[start_opid].node.split_dim = split_dim
                    break
        else:
            # Split from te first dim that larger than 1
            for dim, dim_value in enumerate(input_shape):
                if dim_value > 1:
                    split_dim = dim
                    self.split_tensor_by_n(inputs, input_split, split_dim)
                    self.nodes[start_opid].node.split_dim = split_dim
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
        number_of_concate = 1
        # If end_op is Pack, then number of concatenate should be the number of inputs (for gpt2 model)
        opcode_index = info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'PACK':
            number_of_concate = len(info['inputs'])

        for i in range(number_of_concate):
            new_op_info = {
                "opcode_index": self.get_opcode_index(2),
                "inputs": copy.deepcopy(self.split_tensor_table[info['inputs'][i]]),
                "outputs": [info['inputs'][i]],
                "builtin_options_type": "ConcatenationOptions",
                "builtin_options": {
                    'axis': 1
                }
            }
            self.new_operators.append(new_op_info)

    # Parent op(multi output) -> concat -> split -> child op
    def concat_than_split(self, opid, next_opid, output_split, input_idx):
        opid_info = self.nodes[opid].node.info
        next_opid_info = self.nodes[next_opid].node.info

        # First, create a concate node
        concate_op_info = {
            "opcode_index": self.get_opcode_index(2),
            "inputs": copy.deepcopy(self.split_tensor_table[opid_info['outputs'][0]]),
            "outputs": [opid_info['outputs'][0]],
            "builtin_options_type": "ConcatenationOptions",
            "builtin_options": {
                'axis': 1
            }
        }
        self.new_operators.append(concate_op_info)
        
        # Second, create a split node
        # immediate_tensor is prepare for splitting then become split op's output
        immediate_tensor = copy.deepcopy(self.tensors[opid_info['outputs'][0]])
        self.tensors.append(immediate_tensor)
        immediate_tensor_id = len(self.tensors) - 1
        split_dim = self.nodes[opid].node.split_dim
        self.split_tensor_by_n(immediate_tensor_id, output_split, split_dim)
        for child in self.nodes[opid].node.children:
            self.nodes[child].node.split_dim = split_dim
        
        axis_tensor = {
            "shape": [],
            "type": "INT32",
            "buffer": len(self.buffers),
            "name": self.tensors[immediate_tensor_id]['name']+"_split_axis_tensor",
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
            "inputs": [opid_info['outputs'][0]],
            "outputs": outputs,
            "builtin_options_type": "SplitOptions",
            "builtin_options": {
                "num_splits" : len(outputs)
            }
        }
        self.new_operators.append(split_op_info)

        # Final, update start_op's input
        next_opid_info['inputs'][input_idx] = immediate_tensor_id

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
            int_data = [ int.from_bytes(byte_data[4 * i: 4 * i + 4], byteorder='little') for i in range(len(byte_data) // 4)]
            if False in [int_data[i] == 0 for i in [0,1,6,7]]:
                raise "pad in N or C"
            elif int_data[2] != int_data[3] or int_data[4] != int_data[5]:
                raise "asymmetric pad in H or W"
            return int_data[2], int_data[4]

        def conv_apply_fusion(pad_opid, conv_opid):
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
            calculated_padding_W = total_padding_W // 2 if total_padding_W > 0 else 0

            conv_op_info['inputs'][0] = pad_op_info['inputs'][0]
            if calculated_padding_H == pad_H and calculated_padding_W == pad_W:
                conv_op_info['builtin_options']['padding'] = 'SAME'
            elif len(conv_op_info['inputs']) == 3:
                conv_op_info['inputs'].append(self.get_padding_param_tensor(pad_H,pad_W))
            elif len(conv_op_info['inputs']) < 3:
                BaseException("worng inputs format: length < 3")
            elif len(conv_op_info['inputs']) > 3:
                BaseException("overriding padding param already exist")

        def pool_apply_fusion(pad_opid, pool_opid):
            pad_op_info = self.ori_graph.ops[pad_opid].info
            pool_op_info = self.ori_graph.ops[pool_opid].info
            pad_H, pad_W = get_pad_param(self.buffers[self.tensors[pad_op_info['inputs'][1]]['buffer']]['data'])

            outputs = pool_op_info['outputs']
            in_shape = self.tensors[pad_op_info['inputs'][0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = [pool_op_info['builtin_options']['filter_height'], pool_op_info['builtin_options']['filter_width']]
            stride_h = pool_op_info['builtin_options']['stride_h']
            total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[0] - in_shape[1]
            total_padding_W = (out_shape[2] - 1) * stride_h + ker_shape[1] - in_shape[2]
            calculated_padding_H = total_padding_H // 2 if total_padding_H > 0 else 0
            calculated_padding_W = total_padding_W // 2 if total_padding_W > 0 else 0

            pool_op_info['inputs'][0] = pad_op_info['inputs'][0]
            if calculated_padding_H == pad_H and calculated_padding_W == pad_W:
                pool_op_info['builtin_options']['padding'] = 'SAME'
            elif len(pool_op_info['inputs']) == 1:
                pool_op_info['inputs'].append(self.get_padding_param_tensor(pad_H,pad_W))
            elif len(pool_op_info['inputs']) < 1:
                BaseException("worng inputs format: length < 1")
            elif len(pool_op_info['inputs']) > 1:
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
            conv_need_fusion = []
            pool_need_fusion = []
            for child_id in self.ori_graph.ops[cur_opid].children:
                if(is_pad and self.opcodes[self.ori_graph.ops[child_id].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) in [3, 4]):
                    conv_need_fusion.append(child_id)
                    is_deprecated = 1 if is_deprecated == 0 else -1
                elif(is_pad and self.opcodes[self.ori_graph.ops[child_id].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) == 17):
                    pool_need_fusion.append(child_id)
                    is_deprecated = 1 if is_deprecated == 0 else -1
                else:
                    is_deprecated = -1
                PaddingFusion_dfs(child_id, visited, deprecated)

            if is_deprecated == 1:
                deprecated.append(cur_opid)
                for opid in conv_need_fusion:
                    conv_apply_fusion(cur_opid, opid)
                for opid in pool_need_fusion:
                    pool_apply_fusion(cur_opid, opid)

            return deprecated

        visited = [False for _ in range(len(self.ori_graph.ops))]
        deprecated = []
        while 1 :
            deprecated_new = []
            # Usually BERT model won't have pad & conv need to be fused
            PaddingFusion_dfs(self.ori_graph.root_op_ids[0], visited, deprecated_new)
            if len(deprecated_new) == 0:
                break
            for opid in deprecated_new:
                for i, root_opid in enumerate(self.ori_graph.root_op_ids):
                    if opid == root_opid:
                        self.ori_graph.root_op_ids[i] = self.ori_graph.ops[opid].children[0]

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
    
    # Customized modify for BERT model
    def Elminate_useless_data_layout_op(self):
        pattern = ["RESHAPE", "TRANSPOSE", "RESHAPE", "SPLIT"]
        have_new_buffer = {"first": False, "id": 0}

        def traversal_dfs(cur_opid, pattern_id, visited, tmp_deprecated, deprecated):
            # Check visited
            if visited[cur_opid]:
                return deprecated
            visited[cur_opid] = True

            # Is {reshape -> transpose -> reshape -> split} from cur_opid and its children?
            cur_op = self.ori_graph.ops[cur_opid]
            opcode_index = cur_op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            in_pattern = opcode_type == pattern[pattern_id]
            if in_pattern:
                tmp_deprecated.append(cur_opid)
                pattern_id += 1
            else:
                tmp_deprecated = []
                pattern_id = 0

            # Target pattern found
            if pattern_id == 4:
                # Check whether in V path (which need to stop at transpose)
                value_path = False
                deprecated_first_op = self.ori_graph.ops[tmp_deprecated[0]]
                head_op = self.ori_graph.ops[deprecated_first_op.parents[0]]
                h_output_name = self.tensors[head_op.info['outputs'][0]]['name']
                head_parent_op = self.ori_graph.ops[head_op.parents[0]]
                hp_output_name = self.tensors[head_parent_op.info['outputs'][0]]['name']
                if 'value' in h_output_name or 'value' in hp_output_name:
                    value_path = True
                # No need to eliminate the split op
                tmp_deprecated.pop(-1)
                is_deprecated = 1
                split_op = cur_op
                # Traverse each output of the split op
                for idx, output_id in enumerate(split_op.info['outputs']):
                    # Record the child order of the split op
                    child_id = 0
                    for i, child in enumerate(split_op.children):
                        if output_id == self.ori_graph.ops[child].info['inputs'][0]:
                            child_op = self.ori_graph.ops[child]
                            child_id = i
                            break
                    pre_op = split_op
                    steps = 0

                    while(steps < 3):
                        opcode_index = child_op.info.get("opcode_index")
                        opcode_type = self.opcodes[opcode_index].get("builtin_code")
                        # Can't eliminate the transpose op if in value path
                        if opcode_type == "TRANSPOSE" and value_path:
                            # Update the output tensors of the split to the input tensor of the transpose
                            if pre_op.opid == split_op.opid:
                                pass
                            else:
                                split_op.info['outputs'][idx] = pre_op.info['outputs'][0]
                                split_op.children[child_id] = child_op.opid
                                child_op.parents[0] = split_op.opid
                            # Update the input tensor of the split to the tmp_deprecated[0]'s parent
                            split_op.info['inputs'][1] = head_op.info['outputs'][0]
                            split_op.parents[0] = head_op.opid
                            head_op.children[0] = split_op.opid
                            break
                        elif opcode_type == "FULLY_CONNECTED" and not value_path:
                            # Update the output tensors of the split to the input tensor of the fully_connected
                            if pre_op.opid == split_op.opid:
                                pass
                            else:
                                split_op.info['outputs'][idx] = pre_op.info['outputs'][0]
                                split_op.children[child_id] = child_op.opid
                                for i in range(len(child_op.parents)):
                                    if child_op.parents[i] == pre_op.opid:
                                        child_op.parents[i] = split_op.opid
                            # Update the input tensor of the split to the tmp_deprecated[0]'s parent
                            split_op.info['inputs'][1] = head_op.info['outputs'][0]
                            split_op.parents[0] = head_op.opid
                            head_op.children[0] = split_op.opid
                            break
                        else:
                            tmp_deprecated.append(child_op.opid)
                            pre_op = child_op
                            child_op = self.ori_graph.ops[child_op.children[0]]
                            steps += 1
                    if steps == 3:
                        is_deprecated = 0
                        break
                if is_deprecated == 1:
                    # Create new axis tensor and buffer for split dim 2 (multi-head attention usually split this dim)
                    if not have_new_buffer["first"]:
                        new_axis_tensor = {
                            "shape": [],
                            "type": "INT32",
                            "buffer": len(self.buffers),
                            "name": "split_axis_tensor",
                            "quantization": {},
                        }
                        new_axis_buffer = {
                            "data": self.int_list_to_byte_list([2])
                        }
                        self.tensors.append(new_axis_tensor)
                        self.buffers.append(new_axis_buffer)
                        have_new_buffer["first"] = True
                        have_new_buffer["id"] = len(self.tensors) - 1
                    split_op.info['inputs'][0] = have_new_buffer["id"]
                    deprecated += tmp_deprecated
                tmp_deprecated = []
                pattern_id = 0
            for child_id in self.ori_graph.ops[cur_opid].children:
                traversal_dfs(child_id, pattern_id, visited, tmp_deprecated, deprecated)
            return deprecated

        visited = [False for _ in range(len(self.ori_graph.ops))]
        deprecated = []
        for i in range(len(self.ori_graph.root_op_ids)):
            deprecated_new = []
            traversal_dfs(self.ori_graph.root_op_ids[i], 0, visited, [], deprecated_new)
            deprecated += deprecated_new
        self.ori_graph.buffers = self.buffers
        self.ori_graph.tensors = self.tensors
        self.ori_graph.remove_deprecated_op(deprecated)
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.ori_graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, self.ori_graph.exec_order)
        self.re_init(new_graph)