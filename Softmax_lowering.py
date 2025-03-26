from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import copy

class SoftMax:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    def softmax_lowering(self):
        # Find all softmax ops
        softmax_ops = self.find_softmax_op()
        for i, op in enumerate(softmax_ops):
            softmax_input_tensor_id = op.info['inputs'][0]
            softmax_output_tensor_id = op.info['outputs'][0]

            # Step 1: Create max_pool op to compute the max value of the input tensor in last axis
            max_pool_output_tensor = copy.deepcopy(self.tensors[softmax_input_tensor_id])
            max_pool_output_tensor['name'] = 'softmax_lower_max_pool_%d' % i
            reduce_size = max_pool_output_tensor['shape'][-1]
            # Reduce the last axis to 1
            max_pool_output_tensor['shape'][-1] = 1
            self.tensors.append(max_pool_output_tensor)
            max_pool_output_tensor_id = len(self.tensors) - 1
            max_pool_op = {
                "opcode_index": self.get_opcode_index(17),
                "inputs": [softmax_input_tensor_id],
                "outputs": [max_pool_output_tensor_id],
                "builtin_options_type": "Pool2DOptions",
                "builtin_options": {
                    "padding": "VALID",
                    "stride_w": 1,
                    "stride_h": 1,
                    "filter_width": reduce_size,
                    "filter_height": 1
                }
            }
            self.operators.append(max_pool_op)
            max_pool_opid = len(self.operators) - 1
            self.ops.append(Node(max_pool_op, max_pool_opid))

            # Step 2: Subtract the max value from the input tensor
            subtract_output_tensor = copy.deepcopy(self.tensors[softmax_input_tensor_id])
            subtract_output_tensor['name'] = 'softmax_lower_subtract_%d' % i
            self.tensors.append(subtract_output_tensor)
            subtract_output_tensor_id = len(self.tensors) - 1
            subtract_op = {
                "opcode_index": self.get_opcode_index(41),
                "inputs": [softmax_input_tensor_id, max_pool_output_tensor_id],
                "outputs": [subtract_output_tensor_id],
                "builtin_options_type": "SubOptions"
            }
            self.operators.append(subtract_op)
            subtract_opid = len(self.operators) - 1
            self.ops.append(Node(subtract_op, subtract_opid))

            # Step 3: Compute the exponential of the subtracted tensor
            exp_output_tensor = copy.deepcopy(self.tensors[subtract_output_tensor_id])
            exp_output_tensor['name'] = 'softmax_lower_exp_%d' % i
            self.tensors.append(exp_output_tensor)
            exp_output_tensor_id = len(self.tensors) - 1
            exp_op = {
                "opcode_index": self.get_opcode_index(47),
                "inputs": [subtract_output_tensor_id],
                "outputs": [exp_output_tensor_id],
                "builtin_options_type": "ExpOptions"
            }
            self.operators.append(exp_op)
            exp_opid = len(self.operators) - 1
            self.ops.append(Node(exp_op, len(self.operators) - 1))

            # Step 4: Compute the sum of the exponential tensor in the last axis
            sum_output_tensor = copy.deepcopy(self.tensors[exp_output_tensor_id])
            sum_output_tensor['name'] = 'softmax_lower_sum_%d' % i
            sum_output_tensor['shape'][-1] = 1
            self.tensors.append(sum_output_tensor)
            sum_output_tensor_id = len(self.tensors) - 1
            sum_op = {
                "opcode_index": self.get_opcode_index(74),
                "inputs": [exp_output_tensor_id],
                "outputs": [sum_output_tensor_id]
            }
            self.operators.append(sum_op)
            sum_opid = len(self.operators) - 1
            self.ops.append(Node(sum_op, len(self.operators) - 1))

            # Step 5: Divide the exponential tensor by the sum tensor
            divide_op = {
                "opcode_index": self.get_opcode_index(42),
                "inputs": [exp_output_tensor_id, sum_output_tensor_id],
                "outputs": [softmax_output_tensor_id],
                "builtin_options_type": "DivOptions"
            }
            self.operators.append(divide_op)
            divide_opid = len(self.operators) - 1
            self.ops.append(Node(divide_op, divide_opid))

            # Update the parent and children of the lower ops
            # Orginal softmax op's input tensor now is the max_pool op & subtract op's input tensor\
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(max_pool_opid)
                    self.ops[max_pool_opid].parents.append(parent_opid)
                    self.ops[parent_opid].children.append(subtract_opid)
                    self.ops[subtract_opid].parents.append(parent_opid)
                    break
            self.ops[max_pool_opid].children.append(subtract_opid)
            self.ops[subtract_opid].parents.append(max_pool_opid)
            self.ops[subtract_opid].children.append(exp_opid)
            self.ops[exp_opid].parents.append(subtract_opid)
            self.ops[exp_opid].children.append(sum_opid)
            self.ops[exp_opid].children.append(divide_opid)
            self.ops[sum_opid].parents.append(exp_opid)
            self.ops[sum_opid].children.append(divide_opid)
            self.ops[divide_opid].parents.append(exp_opid)
            self.ops[divide_opid].parents.append(sum_opid)
            # Original softmax op's output tensor now is the divide op's output tensor
            child_opid = op.children[0]
            for i, opid in enumerate(self.ops[child_opid].parents):
                if opid == op.opid:
                    del self.ops[child_opid].parents[i]
                    self.ops[child_opid].parents.append(divide_opid)
                    self.ops[divide_opid].children.append(child_opid)
                    break
            
            new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
            new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
            self.splitter.re_init(new_graph)

    def find_softmax_op(self):
        softmax_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "SOFTMAX":
                softmax_ops.append(op)
        return softmax_ops

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'
    
    def remove_deprecated_op(self, deprecated):
        new_opid = []
        id = 0
        for i in range(len(self.ops)):
            if i in deprecated:
                new_opid.append(-1)
            else:
                new_opid.append(id)
                id += 1
        for op in self.ops:
            op.opid = new_opid[op.opid]
            for i in range(len(op.parents)):
                op.parents[i] = new_opid[op.parents[i]]
            for i in range(len(op.children)):
                op.children[i] = new_opid[op.children[i]]
        self.ops = [op for op in self.ops if op.opid != -1]
        self.operators = [op for idx, op in enumerate(self.operators) if new_opid[idx] != -1]

    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length = 4, byteorder = 'little')]
        return out