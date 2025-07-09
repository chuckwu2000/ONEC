from MyGraph import Graph
from OpClassify import Op_Classify

op_classify = Op_Classify()
# Classify refer to the tandem processor's paper
# The elementwise main op
# elem_wise_ops = ["ADD", "SUB", "MUL", "LOGISTIC", "RSQRT", "SQUARED_DIFFERENCE", "SOFTMAX", "GELU", "LEAKY_RELU", \
#                 "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", "TANH", "POW", "MAX_POOL_2D", "RESHAPE", "EXP", "SUM"]
# The mac main op
#mac_ops = ["MEAN", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV", "BATCH_MATMUL"]
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops

def genesys_schedule(split_graph: Graph, weights_reuse_need_allocate_tensors, genesys_options: dict):
    # GeneSys have three rules:
    # 1. The single GEMM layer
    # 2. The group of non-GEMM layers
    # 3. The GEMM layer followed by the group of non-GEMM layers: drawback is that the GEMM layer can't have multiple children
    # The block's output need to store back to DRAM
    def rule_schedule(split_graph: Graph):
        for idx, candidate_op in enumerate(split_graph.ordered_ops):
            if candidate_op.have_fully_matched:
                continue
            # Handle Rule 1 and Rule 3
            if candidate_op.is_mac_main_op:
                candidate_op.have_fully_matched = True
                
                cascade_matched_ops = [candidate_op.opid]
                now_idx = idx
                child_idx = idx + 1
                # Have at least one followed elem-wise op
                while child_idx < len(split_graph.ordered_ops):
                    now_op = split_graph.ordered_ops[now_idx]
                    # Can't keep fit Rule 3
                    if genesys_options["can_not_followed_multi_child"]:
                        if len(now_op.children) > 1:
                            break

                    child_op = split_graph.ordered_ops[child_idx]
                    opcode_index = child_op.info.get("opcode_index")
                    opcode_type = split_graph.opcodes[opcode_index].get("builtin_code")
                    if opcode_type not in elem_wise_ops:
                        break

                    # Check whether have producer-consumer relationship
                    have_producer_consumer = False
                    for child in now_op.children:
                        if child == child_op.opid:
                            have_producer_consumer = True
                            break
                    if not have_producer_consumer:
                        break
                    
                    cascade_matched_ops.append(child_op.opid)
                    child_op.have_fully_matched = True
                    now_idx = child_idx
                    child_idx += 1

                # Rule 3
                if len(cascade_matched_ops) > 1:
                    rule_three(cascade_matched_ops)
                else:
                    rule_one(candidate_op)
            # Handle rule 2
            elif candidate_op.is_elem_wise_main_op:
                candidate_op.have_fully_matched = True
                
                now_idx = idx
                child_idx = idx + 1
                while child_idx < len(split_graph.ordered_ops):
                    now_op = split_graph.ordered_ops[now_idx]
                    # Can't keep fit Rule 2
                    if genesys_options["can_not_followed_multi_child"]:
                        if len(now_op.children) > 1:
                            rule_two(now_op)
                            break

                    child_op = split_graph.ordered_ops[child_idx]
                    opcode_index = child_op.info.get("opcode_index")
                    opcode_type = split_graph.opcodes[opcode_index].get("builtin_code")
                    if opcode_type not in elem_wise_ops:
                        rule_two(now_op)
                        break

                    # Check whether have producer-consumer relationship
                    have_producer_consumer = False
                    for child in now_op.children:
                        if child == child_op.opid:
                            have_producer_consumer = True
                            break
                    if not have_producer_consumer:
                        rule_two(now_op)
                        break

                    child_op.have_fully_matched = True
                    now_idx = child_idx
                    child_idx += 1
                
    def rule_one(op):
        if genesys_options["no_intermediate_tensor_reuse"]:
            for tensor_metadata in weights_reuse_need_allocate_tensors[op.info['outputs'][0]].tensors:
                if tensor_metadata.pid == op.opid:
                    tensor_metadata.in_DRAM = True

    def rule_two(op):
        if genesys_options["no_intermediate_tensor_reuse"]:
            for tensor_metadata in weights_reuse_need_allocate_tensors[op.info['outputs'][0]].tensors:
                if tensor_metadata.pid == op.opid:
                    tensor_metadata.in_DRAM = True

    def rule_three(casecade_matched_ops):
        split_graph.cascade_matched_ops.append(casecade_matched_ops)
        if genesys_options["no_intermediate_tensor_reuse"]:
            block_end_op = split_graph.ops[casecade_matched_ops[-1]]
            block_end_output = block_end_op.info['outputs'][0]
            for tensor_metadata in weights_reuse_need_allocate_tensors[block_end_output].tensors:
                if tensor_metadata.pid == block_end_op.opid:
                    tensor_metadata.in_DRAM = True
    
    # Start to perform the GeneSys schedule
    rule_schedule(split_graph)
    
    #return pipeline_split_graph
    split_graph.pipeline_schedule = True
    return split_graph
