from MyGraph import Graph
from AutoSplit import ModelType

class Block:
    def __init__(self, graph: Graph, model_type):
        self.graph = graph
        self.model_type = model_type
        self.data_layout_type = ['CONCATENATION', 'SPLIT', 'SPLIT_V']
        # To avoid blocks inside block
        self.skip_block_type = ['SPLIT', 'SPLIT_V']
        self.ops = graph.ops
        self.opcodes = graph.opcodes
        # --------------------------
        # For BERT model use
        self.QKV_block = []
        self.ln_ffn_block = []
        # --------------------------
        # For other models use
        self.block = []
        # --------------------------
        self.blocks = []
        self.visited = [False] * len(self.ops)
        # --------------------------
        # For BERT model use
        self.QKV_start = False
        self.ln_ffn_start = False
        # --------------------------
        # For other models use
        self.block_start = False
        self.skip_block_start = False
        # --------------------------
        self.traverse_graph_from_root()

    # For now, we assume the case is encoder in transformer
    def DFS_find_ADD(self, current_id):
        if self.visited[current_id]:
            return
        self.visited[current_id] = True
        opcode_index = self.ops[current_id].info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if opcode_type == 'ADD' and len(self.ops[current_id].children) > 2:
            # Start of LN_FFN block (usually followed by mean, squared_diff, and mul)
            if not self.ln_ffn_start and len(self.ops[current_id].children) == 3:
                self.ln_ffn_block.append(current_id)
                self.ln_ffn_start = True
            elif self.ln_ffn_start and len(self.ops[current_id].children) == 4:
                self.ln_ffn_block.append(current_id)
                self.blocks.append(self.ln_ffn_block)
                self.ln_ffn_block = []
                self.ln_ffn_start = False
            # Start of QKV block (usually followed by Q, K, V, and residual connection)
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
    
    def DFS_find_blocks(self, current_id):
        if self.visited[current_id]:
            return
        for parent in self.ops[current_id].parents:
            if not self.visited[parent]:
                return
        self.visited[current_id] = True
        opcode_index = self.ops[current_id].info.get("opcode_index")
        opcode_type = self.opcodes[opcode_index].get("builtin_code")
        if not self.block_start and not opcode_type in self.data_layout_type:
            self.block_start = True
            self.block.append(current_id)
        # We don't support that have blocks inside block (normally start with SPLIT with multi-child, end with CONCATENATION)
        if self.block_start and opcode_type in self.skip_block_type and len(self.ops[current_id].children) > 4:
            self.skip_block_start = True
        elif self.skip_block_start and opcode_type == 'CONCATENATION':
            self.skip_block_start = False
        elif self.block_start and opcode_type in self.data_layout_type:
            self.block.append(current_id)
            self.blocks.append(self.block)
            self.block = []
            self.block_start = False
        
        for child in self.ops[current_id].children:
            self.DFS_find_blocks(child)
        return

    def traverse_graph_from_root(self):
        # Assume the last input is the input_ids
        root_opid = self.graph.root_op_ids[-1]
        # Traverse the graph from the root op
        if self.model_type == ModelType.BERT:
            self.DFS_find_ADD(root_opid)
        elif self.model_type == ModelType.CNN:
            self.DFS_find_blocks(root_opid)