from MyGraph import Graph
from performance import estimate_add_cycles
import queue

def pipeline_schedule(split_graph: Graph):
    # TODO: Implement pipeline scheduling
    def set_hoist_min_schedule_order(split_graph: Graph, current_id):
        q = queue.Queue()
        q.put(current_id)
        # Use BFS to traverse the graph & set hoist_min_schedule_order
        visited_node = []
        while(not q.empty()):
            current_id = q.get()
            visited_node.append(current_id)
            for parent in split_graph.ops[current_id].parents:
                split_graph.ops[current_id].hoist_min_schedule_order = max(split_graph.ops[current_id].hoist_min_schedule_order, split_graph.ops[parent].schedule_order)
            for child in split_graph.ops[current_id].children:
                if child not in visited_node:
                    q.put(child)
        # for op in split_graph.ordered_opid:
        #     print(f"opid: {op}, hoist_min_schedule_order: {split_graph.ops[op].hoist_min_schedule_order}")

    def set_active_engine(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        for opid in new_operators:
            opcode_type = split_graph.ops[opid].info.get("builtin_options_type")
            if opcode_type == "AddOptions":
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = True
                estimate_add_cycles(split_graph, opid)
            elif opcode_type == "Conv2DOptions":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            elif opcode_type == "DepthwiseConv2DOptions":
                split_graph.ops[opid].is_mac_main_op = True
                split_graph.ops[opid].is_elem_wise_main_op = False
            else:
                split_graph.ops[opid].is_mac_main_op = False
                split_graph.ops[opid].is_elem_wise_main_op = False

    def re_schedule(split_graph: Graph):
        new_operators = split_graph.ordered_opid
        # Check each op whether can hoist and modify its order in DF-1 schedule
        for opid in new_operators:
            #print(f"opid: {opid}, is_mac_main_op: {split_graph.ops[opid].is_mac_main_op}, is_ele_main_op: {split_graph.ops[opid].is_elem_wise_main_op}")
            if split_graph.ops[opid].is_mac_main_op:
                # Find the insert position
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_elem_wise_main_op and split_graph.ops[insert_pos_opid].have_matched is False:
                        split_graph.ops[insert_pos_opid].have_matched = True
                        split_graph.ops[opid].have_matched = True
                        split_graph.ops[opid].schedule_order = insert_pos
                        break
            elif split_graph.ops[opid].is_elem_wise_main_op:
                for insert_pos in range(split_graph.ops[opid].hoist_min_schedule_order, split_graph.ops[opid].schedule_order):
                    insert_pos_opid = split_graph.ordered_opid[insert_pos]
                    if split_graph.ops[insert_pos_opid].is_mac_main_op and split_graph.ops[insert_pos_opid].have_matched is False:
                        split_graph.ops[insert_pos_opid].have_matched = True
                        split_graph.ops[opid].have_matched = True
                        split_graph.ops[opid].schedule_order = insert_pos
                        break
        # Store the new schedule order into a list
        split_graph.operators = []
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
    
    # Assume root id is 0
    set_hoist_min_schedule_order(split_graph, 0)

    # Set active engine
    set_active_engine(split_graph)

    # Start to piepline schedule
    re_schedule(split_graph)
    
    #return pipeline_split_graph
    return split_graph