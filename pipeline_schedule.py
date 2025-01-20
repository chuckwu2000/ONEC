from MyGraph import Graph
from Architecture_feature import ArchitectureFeatures

# The elementwise main op
elem_wise_ops = ["ADD", "SUB", "MUL", "LOGISTIC", "RSQRT", "SQUARED_DIFFERENCE", "SOFTMAX", "GELU", "LEAKY_RELU", "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", "TANH", "POW"]
# The mac main op
mac_ops = ["MEAN", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV", "MAX_POOL_2D", "BATCH_MATMUL"]
# The memory main op
mem_ops = ["RESHAPE"]
# The elementwise main op but contain the reduce behavior
reduce_ops = ["MEAN", "REDUCE_MAX", "SOFTMAX"]

def set_active_engine(graph: Graph):
    operators = graph.ordered_ops
    for op in operators:
        opcode_index = op.info.get("opcode_index")
        opcode_type = graph.opcodes[opcode_index].get("builtin_code")
        if opcode_type in elem_wise_ops:
            op.is_mac_main_op = False
            op.is_elem_wise_main_op = True
        elif opcode_type in mac_ops:
            op.is_mac_main_op = True
            op.is_elem_wise_main_op = False
        elif opcode_type in mem_ops:
            op.is_mem_main_op = True
        else:
            op.is_mac_main_op = False
            op.is_elem_wise_main_op = False

def weight_reuse_schedule(split_graph: Graph, weight_reuse_mapping):
    for op in split_graph.ops:
        op.schedule_order = weight_reuse_mapping[op.opid]
    # Update the schedule order in the ordered_ops
    split_graph.ordered_ops = sorted(split_graph.ops, key=lambda x: x.schedule_order)
    # Update the operators in the split_graph
    split_graph.operators = []
    for op in split_graph.ordered_ops:
        split_graph.operators.append(op.info)
    return split_graph

def pipeline_schedule(split_graph: Graph):
    # First priority schedule:
    # Try to reuse the output of mac-main-op as the input of elem-wise-main-op & these two ops can be executed concurrently
    def first_priority_schedule(split_graph: Graph):
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for idx, candidate_op in enumerate(split_graph.ordered_ops):
            # Skip the op that has been matched
            if candidate_op.have_fully_matched:
                continue

            # We start from the mac-main-op, since its output can directlly be used by elem-wise-main-op
            # Conversely, elem-wise-main-op's output can't be directly used by mac-main-op (it normally need neighbor elems to be calculated)
            if candidate_op.is_mac_main_op:
                estimated_total_cycles = candidate_op.estimated_total_cycles
                cascade_matched_ops = [candidate_op.opid]
                now_idx = idx
                child_idx = idx + 1
                while child_idx < len(split_graph.ordered_ops):
                    now_op = split_graph.ordered_ops[now_idx]
                    child_op = split_graph.ordered_ops[child_idx]
                    opcode_index = child_op.info.get("opcode_index")
                    opcode_type = split_graph.opcodes[opcode_index].get("builtin_code")
                    # If the child op need to perform reduce operation, we can't let it directly consume the output of mac-main-op
                    if opcode_type in reduce_ops:
                        break

                    # Check whether have producer-consumer relationship
                    have_producer_consumer = False
                    for child in now_op.children:
                        if child == child_op.opid:
                            have_producer_consumer = True
                            break
                    if not have_producer_consumer:
                        break

                    if child_op.is_mem_main_op:
                        # Skip the mem_main_op
                        now_idx = child_idx
                        child_idx += 1
                        continue
                    
                    if child_op.is_elem_wise_main_op and not child_op.have_fully_matched:
                        if estimated_total_cycles < 0:
                            break
                        else:
                            estimated_total_cycles -= child_op.estimated_total_cycles
                        cascade_matched_ops.append(child_op.opid)
                        child_op.have_fully_matched = True
                        now_idx = child_idx
                        child_idx += 1
                    else:
                        break
                if len(cascade_matched_ops) > 1:
                    candidate_op.have_fully_matched = True
                    split_graph.cascade_matched_ops.append(cascade_matched_ops)

    # Second priority schedule:
    # Try to hoist the elem-wise-main-op to the position that can be executed concurrently with the mac-main-op
    # Or, hoist the mac-main-op to the position that can be executed concurrently with the elem-wise-main-op
    def second_priority_schedule(split_graph: Graph):
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for candidate_op in split_graph.ordered_ops:
            # Skip the op that has been matched
            if candidate_op.have_fully_matched:
                continue

            # Before re-schedule this op, update its hoist_min_schedule_order, since their parents may have been re-scheduled
            candidate_op.hoist_min_schedule_order = -1
            for parent_id in candidate_op.parents:
                # Consider that the parent may be a mem_main_op, it won't be executed in the MAC/Elem-wise engine
                if split_graph.ops[parent_id].is_mem_main_op:
                    parent_id = split_graph.ops[parent_id].parents[0]
                candidate_op.hoist_min_schedule_order = max(candidate_op.hoist_min_schedule_order, round(split_graph.ops[parent_id].schedule_order))

            # Check whether this op can be hoisted
            if candidate_op.is_mac_main_op:
                # Find the insert position
                for insert_pos in range(candidate_op.hoist_min_schedule_order, candidate_op.schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent_id in candidate_op.parents:
                        if split_graph.ops[parent_id].is_mem_main_op:
                            parent_id = split_graph.ops[parent_id].parents[0]
                        if split_graph.ops[parent_id].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_op = split_graph.ordered_ops[insert_pos]
                    if insert_pos_op.is_elem_wise_main_op and insert_pos_op.have_fully_matched is False:
                        # node[opid] can fully cover node[insert_pos_opid]
                        if insert_pos_op.estimated_total_cycles <= candidate_op.estimated_total_cycles:
                            insert_pos_op.have_fully_matched = True
                            insert_pos_op.non_overlap_cycles = 0
                            # To differentiate the schedule order between original op and its next op
                            candidate_op.schedule_order = insert_pos + 0.6
                            candidate_op.non_overlap_cycles -= insert_pos_op.estimated_total_cycles
                            matched_ops = [candidate_op.opid]
                            matched_ops.append(insert_pos_op.opid)
                            # Find more node[insert_pos + n] that can be covered by node[opid]
                            while True:
                                # Find the node[insert_pos + 1]
                                if insert_pos + 1 < len(split_graph.ordered_ops):
                                    insert_pos_child_op = split_graph.ordered_ops[insert_pos + 1]

                                if insert_pos_child_op.is_elem_wise_main_op and insert_pos_child_op.have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this mac_main op
                                    # 1.7 is just a default threshold
                                    if insert_pos_child_op.estimated_total_cycles <= 1.7 * candidate_op.non_overlap_cycles:
                                        insert_pos_child_op.have_fully_matched = True
                                        insert_pos_child_op.non_overlap_cycles = 0
                                        candidate_op.non_overlap_cycles -= insert_pos_child_op.estimated_total_cycles
                                        matched_ops.append(insert_pos_child_op.opid)
                                        # Upadte the insert_pos
                                        insert_pos += 1
                                        continue
                                break
                            candidate_op.have_fully_matched = True
                            candidate_op.non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
                        # node[insert_pos_opid] can fully cover node[opid]
                        else:
                            candidate_op.have_fully_matched = True
                            candidate_op.non_overlap_cycles = 0
                            candidate_op_pos = candidate_op.schedule_order
                            # To differentiate the schedule order between original op and its next op
                            candidate_op.schedule_order = insert_pos + 0.6
                            insert_pos_op.non_overlap_cycles -= candidate_op.estimated_total_cycles
                            matched_ops = [insert_pos_op.opid]
                            matched_ops.append(candidate_op.opid)
                            seq_offset = 0.01
                            # Find more node[opid + n] that can be covered by node[insert_pos_opid]
                            while True:
                                # Find the node[opid + 1]
                                if candidate_op_pos + 1 < len(split_graph.ordered_ops):
                                    candidate_op_child_op = split_graph.ordered_ops[candidate_op_pos + 1]

                                if candidate_op_child_op.is_mac_main_op and candidate_op_child_op.have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this elem_wise_main op
                                    # 1.7 is just a default threshold
                                    if candidate_op_child_op.estimated_total_cycles <= 1.7 * insert_pos_op.non_overlap_cycles:
                                        candidate_op_child_op.have_fully_matched = True
                                        candidate_op_child_op.non_overlap_cycles = 0
                                        candidate_op_child_op.schedule_order = insert_pos + 0.6 + seq_offset
                                        seq_offset += 0.01
                                        insert_pos_op.non_overlap_cycles -= candidate_op_child_op.estimated_total_cycles
                                        matched_ops.append(candidate_op_child_op.opid)
                                        # Upadte the ori_opid_pos
                                        candidate_op_pos += 1
                                        continue
                                break
                            insert_pos_op.have_fully_matched = True
                            insert_pos_op.non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
            elif candidate_op.is_elem_wise_main_op:
                # Find the insert position
                for insert_pos in range(candidate_op.hoist_min_schedule_order, candidate_op.schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent_id in candidate_op.parents:
                        if split_graph.ops[parent_id].is_mem_main_op:
                            parent_id = split_graph.ops[parent_id].parents[0]
                        if split_graph.ops[parent_id].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_op = split_graph.ordered_ops[insert_pos]
                    if insert_pos_op.is_mac_main_op and insert_pos_op.have_fully_matched is False:
                        # node[opid] can fully cover node[insert_pos_opid]
                        if insert_pos_op.estimated_total_cycles <= candidate_op.estimated_total_cycles:
                            insert_pos_op.have_fully_matched = True
                            insert_pos_op.non_overlap_cycles = 0
                            # To differentiate the schedule order between original op and its next op
                            candidate_op.schedule_order = insert_pos + 0.6
                            candidate_op.non_overlap_cycles -= insert_pos_op.estimated_total_cycles
                            matched_ops = [candidate_op.opid]
                            matched_ops.append(insert_pos_op.opid)
                            # Find more node[insert_pos + n] that can be covered by node[opid]
                            while True:
                                # Find the node[insert_pos + 1]
                                if insert_pos + 1 < len(split_graph.ordered_ops):
                                    insert_pos_child_op = split_graph.ordered_ops[insert_pos + 1]

                                if insert_pos_child_op.is_mac_main_op and insert_pos_child_op.have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this mac_main op
                                    # 1.7 is just a default threshold
                                    if insert_pos_child_op.estimated_total_cycles <= 1.7 * candidate_op.non_overlap_cycles:
                                        insert_pos_child_op.have_fully_matched = True
                                        insert_pos_child_op.non_overlap_cycles = 0
                                        candidate_op.non_overlap_cycles -= insert_pos_child_op.estimated_total_cycles
                                        matched_ops.append(insert_pos_child_op.opid)
                                        # Upadte the insert_pos
                                        insert_pos += 1
                                        continue
                                break
                            candidate_op.have_fully_matched = True
                            candidate_op.non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
                        # node[insert_pos_opid] can fully cover node[opid]
                        else:
                            candidate_op.have_fully_matched = True
                            candidate_op.non_overlap_cycles = 0
                            candidate_op_pos = candidate_op.schedule_order
                            # To differentiate the schedule order between original op and its next op
                            candidate_op.schedule_order = insert_pos + 0.6
                            insert_pos_op.non_overlap_cycles -= candidate_op.estimated_total_cycles
                            matched_ops = [insert_pos_op.opid]
                            matched_ops.append(candidate_op.opid)
                            seq_offset = 0.01
                            # Find more node[opid + n] that can be covered by node[insert_pos_opid]
                            while True:
                                # Find the node[opid + 1]
                                if candidate_op_pos + 1 < len(split_graph.ordered_ops):
                                    candidate_op_child_op = split_graph.ordered_ops[candidate_op_pos + 1]

                                if candidate_op_child_op.is_elem_wise_main_op and candidate_op_child_op.have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this elem_wise_main op
                                    # 1.7 is just a default threshold
                                    if candidate_op_child_op.estimated_total_cycles <= 1.7 * insert_pos_op.non_overlap_cycles:
                                        candidate_op_child_op.have_fully_matched = True
                                        candidate_op_child_op.non_overlap_cycles = 0
                                        candidate_op_child_op.schedule_order = insert_pos + 0.6 + seq_offset
                                        seq_offset += 0.01
                                        insert_pos_op.non_overlap_cycles -= candidate_op_child_op.estimated_total_cycles
                                        matched_ops.append(candidate_op_child_op.opid)
                                        # Upadte the ori_opid_pos
                                        candidate_op_pos += 1
                                        continue
                                break
                            insert_pos_op.have_fully_matched = True
                            insert_pos_op.non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
    
    # Start to piepline schedule
    first_priority_schedule(split_graph)
    second_priority_schedule(split_graph)
    
    #return pipeline_split_graph
    split_graph.pipeline_schedule = True
    return split_graph

def set_new_operators(split_graph: Graph):
        # Store the new schedule order into a list, it contain some ops have the same schedule order
        split_graph.operators = []
        new_ops = []
        # Store the correspondence between old opid and new opid
        old_new_opid_correspondence = {}
        # For tmp use
        new_schedule_order_list = []
        for i, op in enumerate(split_graph.ops):
            new_schedule_order_list.append((i, op.schedule_order))
        # Sort the list by schedule order & update the schedule order
        order = 0
        for elem in sorted(new_schedule_order_list, key=lambda pair: pair[1]):
            op = split_graph.ops[elem[0]]
            old_new_opid_correspondence[op.opid] = order
            op.schedule_order = order
            order += 1
            new_ops.append(op)
            split_graph.operators.append(op.info)
        split_graph.ops = new_ops
        split_graph.ordered_ops = new_ops

        # Update the opid in cascade_matched_ops and matched_ops
        new_cascade_matched_ops = []
        new_matched_ops = []
        for cascade_matched_pattern in split_graph.cascade_matched_ops:
            new_cascade_pattern = []
            for opid in cascade_matched_pattern:
                new_opid = old_new_opid_correspondence[opid]
                new_cascade_pattern.append(split_graph.ops[new_opid].schedule_order)
            new_cascade_matched_ops.append(new_cascade_pattern)
        for matched_pattern in split_graph.matched_ops:
            new_matched_pattern = []
            for opid in matched_pattern:
                new_opid = old_new_opid_correspondence[opid]
                new_matched_pattern.append(split_graph.ops[new_opid].schedule_order)
            new_matched_ops.append(new_matched_pattern)
        split_graph.cascade_matched_ops = new_cascade_matched_ops
        split_graph.matched_ops = new_matched_ops

        # Update the opid & parents & children & op_lookup_input
        op_lookup_input = {}
        for new_opid, op in enumerate(split_graph.ops):
            # Update the opid
            op.opid = new_opid
            # Update the parents & children
            new_parents = list()
            new_children = list()
            for parent_id in op.parents:
                new_parents.append(old_new_opid_correspondence[parent_id])
            for child_id in op.children:
                new_children.append(old_new_opid_correspondence[child_id])
            op.parents = new_parents
            op.children = new_children
            # Update the op_lookup_input
            for in_id in op.info['inputs']:
                if in_id not in op_lookup_input.keys():
                    op_lookup_input[in_id] = []
                op_lookup_input[in_id].append(new_opid)
        split_graph.op_lookup_input = op_lookup_input
        # Update the root_op_ids
        new_root_op_ids = []
        for in_tensor_id in split_graph.inputs:
            new_root_op_ids.append(op_lookup_input[in_tensor_id][0])
        split_graph.root_op_ids = new_root_op_ids