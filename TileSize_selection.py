from Architecture_feature import ArchitectureFeatures
from AutoSplit import ModelType
from OpClassify import Op_Classify
import math

op_classify = Op_Classify()
mac_ops = op_classify.mac_ops
weight_reuse_ops = op_classify.need_weights_ops

# Use binary search to find the best tile size
class TileSizeSelection:
    def __init__(self, ori_graph, model_type):
        self.graph = ori_graph
        self.model_type = model_type

    # Pick the best split height for this model
    def pick_best_tile_size(self, token_size):
        sram_usage_threshold = 0.5 * ArchitectureFeatures.SRAM_MAX_SIZE
        # As for BERT model, the better systolic array's utilization, the better performance
        if self.model_type == ModelType.BERT:
            # Find all the factors of the token size
            candidate_tile_size = []
            for i in range(1, int(math.sqrt(token_size)) + 1):
                if token_size % i == 0:
                    candidate_tile_size.append(i)
                    if i != token_size // i:
                        candidate_tile_size.append(token_size // i)
            candidate_tile_size = sorted(candidate_tile_size)
            # Default to set the tile_size equal to MAC height, which can better utilize the MAC units
            # Or set the tile_size equal to the token size, if the token size is smaller than MAC height
            max_tile_size = min(ArchitectureFeatures.MAC_height, token_size)
            for id, size in enumerate(candidate_tile_size):
                if size == max_tile_size:
                    max_tile_id = id
                    break
                elif size > max_tile_size:
                    max_tile_id = id - 1
                    break
            min_tile_id = 0
            # Use binary search to find the best tile size
            valid_id = 0
            while min_tile_id <= max_tile_id:
                mid_tile_id = (min_tile_id + max_tile_id) // 2
                tile_size = candidate_tile_size[mid_tile_id]
                tile_count = token_size // tile_size
                valid_tile_size = self.check_tile_size(tile_count, sram_usage_threshold)
                if valid_tile_size:
                    valid_id = mid_tile_id
                    min_tile_id = mid_tile_id + 1
                else:
                    max_tile_id = mid_tile_id - 1
            return candidate_tile_size[valid_id]
        # As for CNN model, the more tile path the better performance
        elif self.model_type == ModelType.CNN:
            height_size = token_size
            # Find all the factors of the height size
            candidate_tile_size = []
            for i in range(1, int(math.sqrt(height_size)) + 1):
                if height_size % i == 0:
                    candidate_tile_size.append(i)
                    if i != height_size // i:
                        candidate_tile_size.append(height_size // i)
            sorted(candidate_tile_size)
            max_tile_id = len(candidate_tile_size) - 1 
            min_tile_id = 2
            # Use binary search to find the best tile size
            while min_tile_id <= max_tile_id:
                mid_tile_id = (min_tile_id + max_tile_id) // 2
                tile_size = candidate_tile_size[mid_tile_id]
                tile_count = token_size // tile_size
                valid_tile_size = self.check_tile_size(tile_count, sram_usage_threshold)
                if valid_tile_size:
                    valid_id = mid_tile_id
                    max_tile_id = mid_tile_id - 1
                else:
                    min_tile_id = mid_tile_id + 1
            return candidate_tile_size[valid_id]
        else:
            raise ValueError("Need to assign model type if you don't set the tile size!!")
        
    def check_tile_size(self, tile_count, sram_usage_threshold):
        # Graph is non-split graph
        ordered_ops = self.graph.ordered_ops
        for op in ordered_ops:
            total_tensor_size = 0
            opcode_index = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')

            if opcode_type in weight_reuse_ops:
                weight_size = self.compute_tensor_size(op.info['inputs'][1])
                if weight_size > sram_usage_threshold:
                    pass
                else:
                    total_tensor_size += weight_size
                total_tensor_size += self.compute_tensor_size(op.info['inputs'][0])
                total_tensor_size += self.compute_tensor_size(op.info['inputs'][2])
                total_tensor_size += self.compute_tensor_size(op.info['outputs'][0])
                if (total_tensor_size / tile_count) > sram_usage_threshold:
                    return False
            elif opcode_type in mac_ops:
                for tensor_id in op.info['inputs']:
                    tensor_size = self.compute_tensor_size(tensor_id)
                    total_tensor_size += tensor_size
                for tensor_id in op.info['outputs']:
                    tensor_size = self.compute_tensor_size(tensor_id)
                    total_tensor_size += tensor_size
                if (total_tensor_size / tile_count) > sram_usage_threshold:
                    return False
        return True
        
    def compute_tensor_size(self, tensor_id) -> int:
        if tensor_id == -1:
            return 0
        tensor = self.graph.tensors[tensor_id]
        shape = tensor['shape']
        if tensor.get("type") == "INT8":
            elem_size = 8
        else:
            elem_size = 32
        element = 1
        for dim in shape:
            element *= dim
        size = element * (elem_size // 8)
        return size