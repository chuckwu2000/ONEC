from MyGraph import Graph

# The elementwise main op
elem_wise_ops = ["ADD", "SUB", "MUL", "LOGISTIC", "RSQRT", "SQUARED_DIFFERENCE", "SOFTMAX", "GELU", "LEAKY_RELU", "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", "TANH", "POW"]
# The mac main op
mac_ops = ["MEAN", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV", "MAX_POOL_2D", "BATCH_MATMUL"]
# The memory main op
mem_ops = ["RESHAPE"]
# The elementwise main op but contain the reduce behavior
reduce_ops = ["MEAN", "REDUCE_MAX", "SOFTMAX"]

def set_active_engine(graph: Graph):
    operators = graph.ordered_opid
    for opid in operators:
        opcode_index = graph.ops[opid].info.get("opcode_index")
        opcode_type = graph.opcodes[opcode_index].get("builtin_code")
        if opcode_type in elem_wise_ops:
            graph.ops[opid].is_mac_main_op = False
            graph.ops[opid].is_elem_wise_main_op = True
        elif opcode_type in mac_ops:
            graph.ops[opid].is_mac_main_op = True
            graph.ops[opid].is_elem_wise_main_op = False
        elif opcode_type in mem_ops:
            graph.ops[opid].is_mem_main_op = True
        else:
            graph.ops[opid].is_mac_main_op = False
            graph.ops[opid].is_elem_wise_main_op = False

def pipeline_schedule(split_graph: Graph):
    # First priority schedule:
    # Try to reuse the output of mac-main-op as the input of elem-wise-main-op & these two ops can be executed concurrently
    # TODO: need to check that sequencial elem-wise-main-ops have producer-consumer relationship
    def first_priority_schedule(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for idx, opid in enumerate(new_operators):
            # Skip the op that has been matched
            if split_graph.ops[opid].have_fully_matched:
                continue

            # We start from the mac-main-op, since its output can directlly be used by elem-wise-main-op
            # Conversely, elem-wise-main-op's output can't be directly used by mac-main-op (it normally need neighbor elems to be calculated)
            if split_graph.ops[opid].is_mac_main_op:
                estimated_total_cycles = split_graph.ops[opid].estimated_total_cycles
                cascade_matched_ops = [opid]
                child_idx = idx + 1
                if child_idx >= len(new_operators):
                    continue
                child_opid = new_operators[child_idx]
                while child_opid < len(new_operators):
                    opcode_index = split_graph.ops[child_opid].info.get("opcode_index")
                    opcode_type = split_graph.opcodes[opcode_index].get("builtin_code")
                    # If the child op need to perform reduce operation, we can't let it directly consume the output of mac-main-op
                    if opcode_type in reduce_ops:
                        break
                    if split_graph.ops[child_opid].is_elem_wise_main_op and split_graph.ops[child_opid].have_fully_matched is False:
                        # Check whether child_opid is the opid's child
                        keep_search = False
                        for child in split_graph.ops[opid].children:
                            if child_opid == child:
                                if estimated_total_cycles <= split_graph.ops[child_opid].estimated_total_cycles:
                                    keep_search = False
                                else:
                                    estimated_total_cycles -= split_graph.ops[child_opid].estimated_total_cycles
                                    keep_search = True
                                cascade_matched_ops.append(child_opid)
                                split_graph.ops[child_opid].have_fully_matched = True
                                break
                        if not keep_search:
                            break
                        child_idx += 1
                        child_opid = new_operators[child_idx]
                    else:
                        break
                if len(cascade_matched_ops) > 1:
                    split_graph.ops[opid].have_fully_matched = True
                    split_graph.cascade_matched_ops.append(cascade_matched_ops)

    # Second priority schedule:
    # Try to hoist the elem-wise-main-op to the position that can be executed concurrently with the mac-main-op
    # Or, hoist the mac-main-op to the position that can be executed concurrently with the elem-wise-main-op
    def second_priority_schedule(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for opid in new_operators:
            # Skip the op that has been matched
            if split_graph.ops[opid].have_fully_matched:
                continue

            # Before re-schedule this op, update its hoist_min_schedule_order, since their parents may have been re-scheduled
            split_graph.ops[opid].hoist_min_schedule_order = -1
            for parent in split_graph.ops[opid].parents:
                # Consider that the parent may be a mem_main_op, it won't be executed in the MAC/Elem-wise engine
                if split_graph.ops[parent].is_mem_main_op:
                    parent = split_graph.ops[parent].parents[0]
                split_graph.ops[opid].hoist_min_schedule_order = max(split_graph.ops[opid].hoist_min_schedule_order, round(split_graph.ops[parent].schedule_order))

            # Check whether this op can be hoisted
            if split_graph.ops[opid].is_mac_main_op:
                # Find the insert position
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent in split_graph.ops[opid].parents:
                        if split_graph.ops[parent].is_mem_main_op:
                            parent = split_graph.ops[parent].parents[0]
                        if split_graph.ops[parent].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_elem_wise_main_op and split_graph.ops[insert_pos_opid].have_fully_matched is False:
                        # node[opid] can fully cover node[insert_pos_opid]
                        if split_graph.ops[insert_pos_opid].estimated_total_cycles <= split_graph.ops[opid].estimated_total_cycles:
                            split_graph.ops[insert_pos_opid].have_fully_matched = True
                            split_graph.ops[insert_pos_opid].non_overlap_cycles = 0
                            # To differentiate the schedule order between original op and its next op
                            split_graph.ops[opid].schedule_order = insert_pos + 0.6
                            split_graph.ops[opid].non_overlap_cycles -= split_graph.ops[insert_pos_opid].estimated_total_cycles
                            matched_ops = [opid]
                            matched_ops.append(insert_pos_opid)
                            # Find more node[insert_pos + n] that can be covered by node[opid]
                            while True:
                                # Find the node[insert_pos + 1]
                                if insert_pos + 1 < len(split_graph.ordered_opid):
                                    insert_pos_child_id = split_graph.ordered_opid[insert_pos + 1]

                                if split_graph.ops[insert_pos_child_id].is_elem_wise_main_op and split_graph.ops[insert_pos_child_id].have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this mac_main op
                                    # 1.7 is just a default threshold
                                    if split_graph.ops[insert_pos_child_id].estimated_total_cycles <= 1.7 * split_graph.ops[opid].non_overlap_cycles:
                                        split_graph.ops[insert_pos_child_id].have_fully_matched = True
                                        split_graph.ops[insert_pos_child_id].non_overlap_cycles = 0
                                        split_graph.ops[opid].non_overlap_cycles -= split_graph.ops[insert_pos_child_id].estimated_total_cycles
                                        matched_ops.append(insert_pos_child_id)
                                        # Upadte the insert_pos
                                        insert_pos += 1
                                        continue
                                break
                            split_graph.ops[opid].have_fully_matched = True
                            split_graph.ops[opid].non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
                        # node[insert_pos_opid] can fully cover node[opid]
                        else:
                            split_graph.ops[opid].have_fully_matched = True
                            split_graph.ops[opid].non_overlap_cycles = 0
                            opid_pos = split_graph.ops[opid].schedule_order
                            # To differentiate the schedule order between original op and its next op
                            split_graph.ops[opid].schedule_order = insert_pos + 0.6
                            split_graph.ops[insert_pos_opid].non_overlap_cycles -= split_graph.ops[opid].estimated_total_cycles
                            matched_ops = [insert_pos_opid]
                            matched_ops.append(opid)
                            seq_offset = 0.01
                            # Find more node[opid + n] that can be covered by node[insert_pos_opid]
                            while True:
                                # Find the node[opid + 1]
                                if opid_pos + 1 < len(split_graph.ordered_opid):
                                    opid_child_id = split_graph.ordered_opid[opid_pos + 1]

                                if split_graph.ops[opid_child_id].is_mac_main_op and split_graph.ops[opid_child_id].have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this elem_wise_main op
                                    # 1.7 is just a default threshold
                                    if split_graph.ops[opid_child_id].estimated_total_cycles <= 1.7 * split_graph.ops[insert_pos_opid].non_overlap_cycles:
                                        split_graph.ops[opid_child_id].have_fully_matched = True
                                        split_graph.ops[opid_child_id].non_overlap_cycles = 0
                                        split_graph.ops[opid_child_id].schedule_order = insert_pos + 0.6 + seq_offset
                                        seq_offset += 0.01
                                        split_graph.ops[insert_pos_opid].non_overlap_cycles -= split_graph.ops[opid_child_id].estimated_total_cycles
                                        matched_ops.append(opid_child_id)
                                        # Upadte the ori_opid_pos
                                        opid_pos += 1
                                        continue
                                break
                            split_graph.ops[insert_pos_opid].have_fully_matched = True
                            split_graph.ops[insert_pos_opid].non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
            elif split_graph.ops[opid].is_elem_wise_main_op:
                # Find the insert position
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent in split_graph.ops[opid].parents:
                        if split_graph.ops[parent].is_mem_main_op:
                            parent = split_graph.ops[parent].parents[0]
                        if split_graph.ops[parent].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_mac_main_op and split_graph.ops[insert_pos_opid].have_fully_matched is False:
                        # node[opid] can fully cover node[insert_pos_opid]
                        if split_graph.ops[insert_pos_opid].estimated_total_cycles <= split_graph.ops[opid].estimated_total_cycles:
                            split_graph.ops[insert_pos_opid].have_fully_matched = True
                            split_graph.ops[insert_pos_opid].non_overlap_cycles = 0
                            # To differentiate the schedule order between original op and its next op
                            split_graph.ops[opid].schedule_order = insert_pos + 0.6
                            split_graph.ops[opid].non_overlap_cycles -= split_graph.ops[insert_pos_opid].estimated_total_cycles
                            matched_ops = [opid]
                            matched_ops.append(insert_pos_opid)
                            # Find more node[insert_pos + n] that can be covered by node[opid]
                            while True:
                                # Find the node[insert_pos + 1]
                                if insert_pos + 1 < len(split_graph.ordered_opid):
                                    insert_pos_child_id = split_graph.ordered_opid[insert_pos + 1]

                                if split_graph.ops[insert_pos_child_id].is_mac_main_op and split_graph.ops[insert_pos_child_id].have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this mac_main op
                                    # 1.7 is just a default threshold
                                    if split_graph.ops[insert_pos_child_id].estimated_total_cycles <= 1.7 * split_graph.ops[opid].non_overlap_cycles:
                                        split_graph.ops[insert_pos_child_id].have_fully_matched = True
                                        split_graph.ops[insert_pos_child_id].non_overlap_cycles = 0
                                        split_graph.ops[opid].non_overlap_cycles -= split_graph.ops[insert_pos_child_id].estimated_total_cycles
                                        matched_ops.append(insert_pos_child_id)
                                        # Upadte the insert_pos
                                        insert_pos += 1
                                        continue
                                break
                            split_graph.ops[opid].have_fully_matched = True
                            split_graph.ops[opid].non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
                        # node[insert_pos_opid] can fully cover node[opid]
                        else:
                            split_graph.ops[opid].have_fully_matched = True
                            split_graph.ops[opid].non_overlap_cycles = 0
                            opid_pos = split_graph.ops[opid].schedule_order
                            # To differentiate the schedule order between original op and its next op
                            split_graph.ops[opid].schedule_order = insert_pos + 0.6
                            split_graph.ops[insert_pos_opid].non_overlap_cycles -= split_graph.ops[opid].estimated_total_cycles
                            matched_ops = [insert_pos_opid]
                            matched_ops.append(opid)
                            seq_offset = 0.01
                            # Find more node[opid + n] that can be covered by node[insert_pos_opid]
                            while True:
                                # Find the node[opid + 1]
                                if opid_pos + 1 < len(split_graph.ordered_opid):
                                    opid_child_id = split_graph.ordered_opid[opid_pos + 1]

                                if split_graph.ops[opid_child_id].is_elem_wise_main_op and split_graph.ops[opid_child_id].have_fully_matched is False:
                                    # Ensure that won't take another longer op to cover this elem_wise_main op
                                    # 1.7 is just a default threshold
                                    if split_graph.ops[opid_child_id].estimated_total_cycles <= 1.7 * split_graph.ops[insert_pos_opid].non_overlap_cycles:
                                        split_graph.ops[opid_child_id].have_fully_matched = True
                                        split_graph.ops[opid_child_id].non_overlap_cycles = 0
                                        split_graph.ops[opid_child_id].schedule_order = insert_pos + 0.6 + seq_offset
                                        seq_offset += 0.01
                                        split_graph.ops[insert_pos_opid].non_overlap_cycles -= split_graph.ops[opid_child_id].estimated_total_cycles
                                        matched_ops.append(opid_child_id)
                                        # Upadte the ori_opid_pos
                                        opid_pos += 1
                                        continue
                                break
                            split_graph.ops[insert_pos_opid].have_fully_matched = True
                            split_graph.ops[insert_pos_opid].non_overlap_cycles = 0
                            split_graph.matched_ops.append(matched_ops)
                            break
          
    def set_new_operators(split_graph: Graph):
        # Store the new schedule order into a list, it contain some op have the same schedule order
        split_graph.operators = []
        split_graph.ordered_opid = []
        # For tmp use
        new_schedule_order_list = []
        for i, op in enumerate(split_graph.ops):
            new_schedule_order_list.append((i, op.schedule_order))
        # Sort the list by schedule order
        order = 0
        for elem in sorted(new_schedule_order_list, key=lambda pair: pair[1]):
            op = split_graph.ops[elem[0]]
            op.schedule_order = order
            order += 1
            split_graph.operators.append(op.info)
            split_graph.ordered_opid.append(op.opid)

        # Update the opid in cascade_matched_ops and matched_ops
        new_cascade_matched_ops = []
        new_matched_ops = []
        for cascade_matched_pattern in split_graph.cascade_matched_ops:
            new_cascade_pattern = []
            for opid in cascade_matched_pattern:
                new_cascade_pattern.append(split_graph.ops[opid].schedule_order)
            new_cascade_matched_ops.append(new_cascade_pattern)
        for matched_pattern in split_graph.matched_ops:
            new_matched_pattern = []
            for opid in matched_pattern:
                new_matched_pattern.append(split_graph.ops[opid].schedule_order)
            new_matched_ops.append(new_matched_pattern)
        split_graph.cascade_matched_ops = new_cascade_matched_ops
        split_graph.matched_ops = new_matched_ops
    
    # Start to piepline schedule
    first_priority_schedule(split_graph)
    second_priority_schedule(split_graph)

    # Set model's new operators & update the cascade_matched_ops and matched_ops
    set_new_operators(split_graph)
    
    #return pipeline_split_graph
    split_graph.pipeline_schedule = True
    return split_graph