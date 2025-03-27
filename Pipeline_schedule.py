from MyGraph import Graph
from OpClassify import Op_Classify

op_classify = Op_Classify()
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops
mem_ops = ["RESHAPE"]
reduce_ops = op_classify.reduce_ops

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
                    # We don't plus one on op's hoist_min_schedule_order, since it may loss some optimize opportunity
                    # But we need to check that op won't have same order with its parents
                    illegal = False
                    for parent_id in candidate_op.parents:
                        if split_graph.ops[parent_id].is_mem_main_op:
                            parent_id = split_graph.ops[parent_id].parents[0]
                        if split_graph.ops[parent_id].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_op = split_graph.ordered_ops[insert_pos]
                    # The second priority schedule's matched_ops should belong to the same block
                    # Since we had ensured the block's memory usage won't exceed the SRAM's capacity
                    if candidate_op.block_id != insert_pos_op.block_id:
                        continue

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

                                # The second priority schedule's matched_ops should belong to the same block
                                if candidate_op.block_id != insert_pos_child_op.block_id:
                                    break

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

                                # The second priority schedule's matched_ops should belong to the same block
                                if candidate_op_child_op.block_id != insert_pos_op.block_id:
                                    break

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
                    # The second priority schedule's matched_ops should belong to the same block
                    # Since we had ensured the block's memory usage won't exceed the SRAM's capacity
                    if candidate_op.block_id != insert_pos_op.block_id:
                        continue

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

                                # The second priority schedule's matched_ops should belong to the same block
                                if candidate_op.block_id != insert_pos_child_op.block_id:
                                    break

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

                                # The second priority schedule's matched_ops should belong to the same block
                                if candidate_op_child_op.block_id != insert_pos_op.block_id:
                                    break

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

    def update_schedule_order(split_graph: Graph):
        # Update the schedule order in the ordered_ops
        split_graph.ordered_ops = sorted(split_graph.ops, key=lambda x: x.schedule_order)
        # Update the operators in the split_graph
        split_graph.operators = []
        for op in split_graph.ordered_ops:
            split_graph.operators.append(op.info)
    
    # Start to piepline schedule
    first_priority_schedule(split_graph)
    second_priority_schedule(split_graph)

    # Update the schedule order
    update_schedule_order(split_graph)
    
    #return pipeline_split_graph
    split_graph.pipeline_schedule = True
    return split_graph