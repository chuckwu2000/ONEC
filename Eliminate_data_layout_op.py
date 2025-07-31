# Use hardcoded manual decide pattern to eliminate useless data layout ops in transformer self-attention part

from AutoSplit import Splitter
from MyGraph import Graph
from OpClassify import Op_Classify

data_layout_op = Op_Classify().data_layout_ops

class Eliminater:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    # Customized modify for self-attention part, must be MHA
    def Eliminate_useless_data_layout_op(self):
        new_buffer = {"first": False, "id": 0}
        deprecated = []
        QKV_ops = self.find_QKV_op()
        for QKV_op in QKV_ops:
            tmp_deprecated = []
            is_MHA = False
            # The optimization's start not guarantee is the fc op, e.g.,(FC -> ADD -> ... -> SPLIT), in this case, the head_op is the ADD op
            head_op = QKV_op
            # Keep traverse until the split op
            cur_op_id = QKV_op.children[0]
            while True:
                cur_op = self.graph.ops[cur_op_id]
                opcode_index = cur_op.info.get("opcode_index")
                opcode_type = self.opcodes[opcode_index].get("builtin_code")
                if opcode_type == "SPLIT":
                    split_op  = cur_op
                    is_MHA = True
                    break
                elif opcode_type == "FULLY_CONNECTED" or opcode_type == "MATMUL":
                    break
                elif opcode_type not in data_layout_op:
                    head_op = cur_op
                else:
                    tmp_deprecated.append(cur_op_id)
                cur_op_id = self.graph.ops[cur_op_id].children[0]

            # If the self-attention part is not MHA, there is usually no opportunity for elimination
            if not is_MHA or len(tmp_deprecated) == 0:
                continue

            # Check whether the fc op is deal with the V (which need to stop at transpose)
            value_path = False
            output_name = self.tensors[QKV_op.info["outputs"][0]]["name"]
            if "value" in output_name:
                value_path = True

            # Update the first half of the path (before split op)
            split_op.info['inputs'][1] = head_op.info['outputs'][0]
            split_op.parents[0] = head_op.opid
            head_op.children[0] = split_op.opid

            # Update the second half of the path (after split op)
            path_deprecated = True
            for output_idx, output_tensor_id in enumerate(split_op.info['outputs']):
                for idx, child in enumerate(split_op.children):
                    if output_tensor_id == self.graph.ops[child].info['inputs'][0]:
                        child_op = self.graph.ops[child]
                        child_idx = idx
                        break
                prev_op = split_op
                # Usually the eliminate opportunity won't over 3 ops
                steps = 0
                while(steps < 3):
                    opcode_index = child_op.info.get("opcode_index")
                    opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    # Can't eliminate the transpose op if in value path
                    if opcode_type == "TRANSPOSE" and value_path:
                        if prev_op.opid == split_op.opid:
                            pass
                        else:
                            split_op.info['outputs'][output_idx] = prev_op.info['outputs'][0]
                            split_op.children[child_idx] = child_op.opid
                            child_op.parents[0] = split_op.opid
                        break
                    elif opcode_type == "FULLY_CONNECTED" and not value_path:
                        if prev_op.opid == split_op.opid:
                            pass
                        else:
                            split_op.info['outputs'][output_idx] = prev_op.info['outputs'][0]
                            split_op.children[child_idx] = child_op.opid
                            for i in range(len(child_op.parents)):
                                if child_op.parents[i] == prev_op.opid:
                                    child_op.parents[i] = split_op.opid
                        break
                    else:
                        tmp_deprecated.append(child_op.opid)
                        prev_op = child_op
                        child_op = self.graph.ops[child_op.children[0]]
                        steps += 1
                if steps == 3:
                    path_deprecated = False
                    break
            if path_deprecated == True:
                # Update split op's spilt axis
                split_input_shape = self.tensors[split_op.info['inputs'][1]]["shape"]
                split_output_shape = self.tensors[split_op.info['outputs'][0]]["shape"]
                for dim, dim_val in enumerate(split_input_shape):
                    if dim_val > 1 and dim_val not in split_output_shape:
                        axis = dim
                        break
                # Only create axis tensor once
                if not new_buffer["first"]:
                    new_axis_tensor = {
                        "shape": [],
                        "type": "INT32",
                        "buffer": len(self.buffers),
                        "name": "split_axis_tensor",
                        "quantization": {},
                    }
                    new_axis_buffer = {
                        "data": self.int_list_to_byte_list([axis])
                    }
                    self.tensors.append(new_axis_tensor)
                    self.buffers.append(new_axis_buffer)
                    new_buffer["first"] = True
                    new_buffer["id"] = len(self.tensors) - 1
                split_op.info['inputs'][0] = new_buffer["id"]
                deprecated += tmp_deprecated
        self.graph.remove_deprecated_op(deprecated)
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, "DF")
        self.splitter.re_init(new_graph)
                        
    # We only find the compute Q, K, V's fc ops
    def find_QKV_op(self):
        target = ["query", "key", "value"]
        QKV_ops = []
        for op in self.ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "FULLY_CONNECTED" or opcode_type == "CONV_2D":
                output_name = self.tensors[op.info["outputs"][0]]["name"]
                for name in target:
                    if name in output_name:
                        QKV_ops.append(op)
                        break
        return QKV_ops
    
    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length = 4, byteorder = 'little')]
        return out