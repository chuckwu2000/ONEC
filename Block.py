from MyGraph import Graph

class Block:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.ops = graph.ops
        self.opcodes = graph.opcodes
        self.QKV_block = []
        self.ln_ffn_block = []
        self.blocks = []
        self.visited = [False] * len(self.ops)
        self.QKV_start = False
        self.ln_ffn_start = False
        self.traverse_graph_from_root()

    # For now, we assume the case is encoder in transformer
    def DFS_find_ADD(self, current_id):
        if self.visited[current_id]:
            return
        self.visited[current_id] = True
        opcode_index = self.ops[current_id].info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'ADD' and len(self.ops[current_id].children) > 2:
            # Start of LN_FFN block
            if not self.ln_ffn_start and len(self.ops[current_id].children) == 3:
                self.ln_ffn_block.append(current_id)
                self.ln_ffn_start = True
            elif self.ln_ffn_start and len(self.ops[current_id].children) == 4:
                self.ln_ffn_block.append(current_id)
                self.blocks.append(self.ln_ffn_block)
                self.ln_ffn_block = []
                self.ln_ffn_start = False
            # Start of QKV block
            if len(self.ops[current_id].children) == 4:
                self.QKV_block.append(current_id)
                self.QKV_start = True
            elif self.QKV_start:
                self.QKV_block.append(current_id)
                self.blocks.append(self.QKV_block)
                self.QKV_block = []
                self.QKV_start = False
            
        for child in self.ops[current_id].children:
            self.DFS_find_ADD(child)
        return

    def traverse_graph_from_root(self):
        # Assume the last input is the input_ids
        root_opid = self.graph.root_op_id[-1]
        # Traverse the graph from the root op
        self.DFS_find_ADD(root_opid)