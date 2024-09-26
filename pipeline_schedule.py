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
            elif opcode_type == "MUL":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "LOGISTIC":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
            elif opcode_type == "CONV_2D":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "DEPTHWISE_CONV_2D":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            else:
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = False

    def re_schedule(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for opid in new_operators:
            # Before re-schedule this op, update its hoidt_min_schedule_order, since their parents may have been re-scheduled
            split_graph.ops[opid].hoist_min_schedule_order = -1
            parent_status = []
            for parent in split_graph.ops[opid].parents:
                parent_status.append(split_graph.ops[parent].schedule_order)
                split_graph.ops[opid].hoist_min_schedule_order = max(split_graph.ops[opid].hoist_min_schedule_order, round(split_graph.ops[parent].schedule_order))

            # Check whether this op can be hoisted
            if split_graph.ops[opid].is_mac_main_op:
                # Find the insert position
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent in split_graph.ops[opid].parents:
                        if split_graph.ops[parent].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_elem_wise_main_op and split_graph.ops[insert_pos_opid].have_matched is False:
                        split_graph.ops[insert_pos_opid].have_matched = True
                        split_graph.ops[opid].have_matched = True
                        # To differnetiate the schedule order between original op and its next op
                        split_graph.ops[opid].schedule_order = insert_pos + 0.6
                        split_graph.matched_ops.append((insert_pos_opid, opid))
                        break
            elif split_graph.ops[opid].is_elem_wise_main_op:
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    # We don't plus one on op's hoist_min_schedule_order, since it will loss some optimize opportunity
                    # But we need to check that op won't run concurrently with its parents
                    illegal = False
                    for parent in split_graph.ops[opid].parents:
                        if split_graph.ops[parent].schedule_order == insert_pos:
                            illegal = True
                    if illegal:
                        continue

                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_mac_main_op and split_graph.ops[insert_pos_opid].have_matched is False:
                        split_graph.ops[insert_pos_opid].have_matched = True
                        split_graph.ops[opid].have_matched = True
                        # To differnetiate the schedule order between original op and its next op
                        split_graph.ops[opid].schedule_order = insert_pos + 0.6
                        split_graph.matched_ops.append((insert_pos_opid, opid))
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
    return split_graph