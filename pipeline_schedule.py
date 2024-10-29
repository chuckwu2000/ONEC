from MyGraph import Graph
import queue

def pipeline_schedule(split_graph: Graph):
    # TODO: Implement pipeline scheduling

    def set_active_engine(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        for opid in new_operators:
            opcode_index = split_graph.ops[opid].info.get("opcode_index")
            opcode_type = split_graph.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "ADD":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "SUB":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "MUL":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "LOGISTIC":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "MEAN":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "RSQRT":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "SQUARED_DIFFERENCE":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "CONV_2D":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "DEPTHWISE_CONV_2D":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "FULLY_CONNECTED":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "SOFTMAX":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "TRANSPOSE_CONV":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "GELU":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "LEAKY_RELU":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "MAX_POOL_2D":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "BATCH_MATMUL":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "RESHAPE":
                split_graph.ops[opid].is_mem_main_op = True
            else:
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = False

    def re_schedule(split_graph: Graph):
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
        # Store the new schedule order into a list, it contain some op have the same schedule order
        split_graph.operators = []
        split_graph.ordered_opid = []
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

    # Set active engine
    set_active_engine(split_graph)

    # Start to piepline schedule
    re_schedule(split_graph)
    
    #return pipeline_split_graph
    split_graph.pipeline_schedule = True
    return split_graph