from AutoSplit import Splitter
from MyGraph import Graph
from MyGraph import Node
import copy
import numpy as np

class Logistic:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    # This lowering may loss precision
    def logistic_lowering(self):
        # Find all logistic ops
        logistic_ops = self.find_logistic_op()
        for i, op in enumerate(logistic_ops):
            logistic_input_tensor_id = op.info['inputs'][0]
            logistic_output_tensor_id = op.info['outputs'][0]

            # Step 1: Create sub op to use 0 to subtract the input tensor to compute the negative value
            # Step 1.1: Create the zero tensor (scale and zero point is same to input tensor)
            sub_input1_buffer = {"data": [0]}
            self.buffers.append(sub_input1_buffer)
            sub_input1_tensor = copy.deepcopy(self.tensors[logistic_input_tensor_id])
            sub_input1_tensor['name'] = 'logistic_lower_zero_sub_%d' % i
            sub_input1_tensor['shape'] = [1, 1, 1, 1]
            sub_input1_tensor['buffer'] = len(self.buffers) - 1
            self.tensors.append(sub_input1_tensor)
            sub_input1_tensor_id = len(self.tensors) - 1
            # Output tensor's scale and zero point is same to input tensor
            sub_output_tensor = copy.deepcopy(self.tensors[logistic_input_tensor_id])
            sub_output_tensor['name'] = 'logistic_lower_sub_%d' % i
            self.tensors.append(sub_output_tensor)
            sub_output_tensor_id = len(self.tensors) - 1
            # Step 1.2: Create the sub op
            sub_op = {
                "opcode_index": self.get_opcode_index(41),
                "inputs": [sub_input1_tensor_id, logistic_input_tensor_id],
                "outputs": [sub_output_tensor_id],
                "builtin_options_type": "SubOptions"
            }
            self.operators.append(sub_op)
            sub_opid = len(self.operators) - 1
            self.ops.append(Node(sub_op, sub_opid))

            # Step 2: Create the exp op to compute the exponential of the negative tensor and plus real value 1 (1 + exp(-x))
            # Step 2.1: Calculate the output tensor's scale and zero point
            # Since x > 0: exp(-x)'s range is (0, 1], x < 0: exp(-x)'s range is (1, large_value), if quantize it, the precision will be lost
            # So we set the output value limit to (0, 10), and set the output scale to 10 / 127.0
            exp_output_tensor = copy.deepcopy(self.tensors[sub_output_tensor_id])
            exp_output_tensor['name'] = 'logistic_lower_exp_%d' % i
            exp_output_scale = 10.0 / 127.0
            np_int_scale = np.float32(exp_output_scale).view('int32')
            np_arr = np.array([np_int_scale], dtype=np.int32)
            int_scale = np_arr.tolist()
            exp_output_tensor['quantization']['scale'] = int_scale
            # Step 2.2: real value 1 turn to zero point than perform bias folding, 1 = (0 - zero_point) * scale -> zero point = -1.0 / scale
            exp_output_zero_point = round(-1.0 / exp_output_scale)
            exp_output_tensor['quantization']['zero_point'] = [exp_output_zero_point]
            self.tensors.append(exp_output_tensor)
            exp_output_tensor_id = len(self.tensors) - 1
            # Step 2.3: Create the exp op
            exp_op = {
                "opcode_index": self.get_opcode_index(47),
                "inputs": [sub_output_tensor_id],
                "outputs": [exp_output_tensor_id],
                "builtin_options_type": "ExpOptions"
            }
            self.operators.append(exp_op)
            exp_opid = len(self.operators) - 1
            self.ops.append(Node(exp_op, len(self.operators) - 1))

            # Step 3: Create the reciprocal op to compute the 1 / (1 + exp(-x))
            reciprocal_output_tensor = self.tensors[logistic_output_tensor_id]
            reciprocal_output_tensor['name'] = 'logistic_lower_reciprocal_%d' % i
            reciprocal_output_tensor_id = logistic_output_tensor_id
            # Create the reciprocal op
            reciprocal_op = {
                "opcode_index": self.get_opcode_index(300),
                "inputs": [exp_output_tensor_id],
                "outputs": [reciprocal_output_tensor_id],
                "builtin_options_type": "ReciprocalOptions"
            }
            self.operators.append(reciprocal_op)
            reciprocal_opid = len(self.operators) - 1
            self.ops.append(Node(reciprocal_op, reciprocal_opid))

            # Update the parent and children of the lower ops
            parent_opid = op.parents[0]
            for i, opid in enumerate(self.ops[parent_opid].children):
                if opid == op.opid:
                    del self.ops[parent_opid].children[i]
                    self.ops[parent_opid].children.append(sub_opid)
                    self.ops[sub_opid].parents.append(parent_opid)
                    break
            self.ops[sub_opid].children.append(exp_opid)
            self.ops[exp_opid].parents.append(sub_opid)
            self.ops[exp_opid].children.append(reciprocal_opid)
            self.ops[reciprocal_opid].parents.append(exp_opid)

            child_opid = op.children[0]
            for i, opid in enumerate(self.ops[child_opid].parents):
                if opid == op.opid:
                    del self.ops[child_opid].parents[i]
                    self.ops[child_opid].parents.append(reciprocal_opid)
                    self.ops[reciprocal_opid].children.append(child_opid)
                    break
            
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)

    def find_logistic_op(self):
        logistic_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            # For now, can't support that softmax op is the last op
            if opcode_type == "LOGISTIC":
                logistic_ops.append(op)
        return logistic_ops

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'