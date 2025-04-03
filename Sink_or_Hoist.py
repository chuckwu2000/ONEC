from AutoSplit import Splitter
from OpClassify import Op_Classify

sink_data_layout = ["PACK"]
hoist_data_layout = ["SPLIT"]
reduce_ops = Op_Classify().reduce_ops
elementwise_ops = Op_Classify().elementwise_ops

class Safe_Hoister_Splitter:
    def __init__(self, splitter: Splitter):
        self.splitter = splitter
        self.graph = splitter.ori_graph
        self.buffers = self.graph.buffers
        self.tensors = self.graph.tensors
        self.opcodes = self.graph.opcodes
        self.operators = self.graph.operators
        self.ops = self.graph.ops

    # Try to sink concat or pack op
    # For now, want to fuse the FC in self-attention with the element-wise ops
    def data_layout_sink(self):
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
        valid_pattern = []
        for op in candidate_ops:
            tmp_pattern = [op]
            now_opid = op.opid
            while(True):
                # Check the next order's op is the child of the current op
                next_opid = now_opid + 1
                if next_opid in self.ops[now_opid].children:
                    # Avoid to sink across the reduce op
                    next_op = self.ops[next_opid]
                    opcode_index = next_op.info.get("opcode_index")
                    opcode_type = self.opcodes[opcode_index].get("builtin_code")
                    if opcode_type in reduce_ops:
                        break
                    # Check the next order's op has only one parent and one child
                    if len(next_op.parents) != 1 or len(next_op.children) != 1:
                        break
                    tmp_pattern.append(next_op)
                    now_opid = next_opid
                else:
                    break
            for op in tmp_pattern:
                opcode_index = next_op.info.get("opcode_index")
                opcode_type = self.opcodes[opcode_index].get("builtin_code")
                if opcode_type in elementwise_ops:
                    valid_pattern.append(tmp_pattern)
                    break

    def data_layout_hoist(self):
        pass