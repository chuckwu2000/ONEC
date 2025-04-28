from OPGen import OPGen

class CodeGen:
    def __init__(self, graph, allocated_tensors, cascade_matched_ops):
        self.graph = graph
        self.operators = graph.operators
        self.allocated_tensors = allocated_tensors
        self.npu_code = ""
        self.cascade_matched_ops = cascade_matched_ops
        self.visited = [False for _ in range(len(graph.ops))]

    def code_gen(self):
        op_code_generator = OPGen(self.graph, self.allocated_tensors)
        # Follow the order of operators in the graph
        for operator in self.operators:
            if self.visited[operator.opid]:
                continue
            # Traverse the cascade_matched_ops, and 
            for cascade_matched_pattern in self.cascade_matched_ops:
                if operator.opid == cascade_matched_pattern[0]:
                    # Generate code for the patterns
                    operators = cascade_matched_pattern
                    op_code = op_code_generator.op_code_gen(operators)
                    self.npu_code += op_code
                    for opid in cascade_matched_pattern:
                        self.visited[opid] = True
                else:
                    # Generate code for the operator
                    op_code = op_code_generator.op_code_gen(operator)
                    self.npu_code += op_code
                    self.visited[operator.opid] = True