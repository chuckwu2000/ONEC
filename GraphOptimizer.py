from MyGraph import Graph

class Optimizer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.visited = [False] * len(graph.ops)
        self.spilt_transpose_opid = []
        self.hoist_split(graph)
        self.eliminate_transpose(graph)

    def from_split_find_parent_transpose(self, current_opid):
        # For now, we try to find the transpose five steps away
        op = self.graph.ops[current_opid]
        current_opid = op.parents[0]
        steps = 5
        while((steps != 0) is True):
            op = self.graph.ops[current_opid]
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
            if opcode_type == 'TRANSPOSE':
                return current_opid
            current_opid = op.parents[0]
            steps -= 1
    
    def find_split_and_traverse_til_end(self, current_opid):
        if self.visited[current_opid]:
            return
        self.visited[current_opid] = True
        opcode_index = self.graph.ops[current_opid].info.get("opcode_index")
        opcode_type = self.graph.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'SPLIT':
            transpose_opid = self.from_split_find_parent_transpose(current_opid)
            if transpose_opid is not None:
                split_transpose_pattern = (current_opid, transpose_opid)
                self.spilt_transpose_opid.append(split_transpose_pattern)
        for child in self.graph.ops[current_opid].children:
            # check if it is splittable op
            self.find_split_and_traverse_til_end(child)
        return
    
    def perform_hoist_split(self):
        pass

    def hoist_split(self, graph: Graph):
        # Traverse the graph from the root op
        root_id = graph.root_op_id[-1]
        self.find_split_and_traverse_til_end(root_id)
        self.perform_hoist_split()
    
    def eliminate_transpose(self, graph: Graph):
        pass