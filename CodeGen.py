from OPGen import OPGen
from MyGraph import Graph

class CodeGen:
    def __init__(self, graph: Graph, allocated_tensors, cascade_matched_ops):
        self.graph = graph
        self.ordered_ops = graph.ordered_ops
        self.allocated_tensors = allocated_tensors
        self.npu_code = ""
        self.cascade_matched_ops = cascade_matched_ops
        self.visited = [False for _ in range(len(graph.ordered_ops))]

    def code_gen(self):
        op_code_generator = OPGen(self.graph, self.allocated_tensors)
        # Follow the order of ops in the graph
        for op in self.ordered_ops:
            if self.visited[op.opid]:
                continue
            # Traverse the cascade_matched_ops
            find_in_cascade = False
            for cascade_matched_pattern in self.cascade_matched_ops:
                ops = []
                if op.opid == cascade_matched_pattern[0]:
                    # Generate code for the patterns
                    for opid in cascade_matched_pattern:
                        ops.append(self.graph.ops[opid])
                    op_code = op_code_generator.op_code_gen(ops)
                    self.npu_code += op_code
                    for opid in cascade_matched_pattern:
                        self.visited[opid] = True
                    find_in_cascade = True
                    break
            if not find_in_cascade:
                # Generate code for the operator
                operator = self.graph.ops[op.opid]
                ops = [operator]
                op_code = op_code_generator.op_code_gen(ops)
                self.npu_code += op_code
                self.visited[op.opid] = True
        # Magic number
        self.npu_code += str(1234567890)