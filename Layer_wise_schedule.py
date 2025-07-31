# Baseline implementation of OEMC experiment

from Architecture_feature import ArchitectureFeatures
from OpClassify import Op_Classify

op_classify = Op_Classify()
elem_wise_ops = op_classify.elementwise_ops
mac_ops = op_classify.mac_ops
fall_back_cpu_ops = op_classify.fall_back_cpu_ops
unary_ops = op_classify.unary_ops
binary_ops = op_classify.binary_ops
trinary_ops = op_classify.trinary_ops
use_lut_ops = op_classify.use_lut_ops

# Based on the DF order, but DF's depth is 1
class Layer_wise_scheduler:
    def __init__(self, graph, need_allocate_tensors, same_layer_next_opids):
        self.graph = graph
        self.tensor_info = need_allocate_tensors
        # same_layer_next_opids[x] = (x's head_opid, x's next_opid, #ops in layer)
        self.same_layer_next_opids = same_layer_next_opids
        self.visited = [False for _ in range(len(self.graph.ops))]
        self.scheduled = [False for _ in range(len(self.graph.ops))]
        self.tensor_in_SRAM = set()
        self.opids_in_block = set()
        self.new_order = 0
        self.layer_wise_mapping = {}

    # Inter-layer's intermediate tensors are not reuse
    def layer_wise_schedule(self):
        # Traverse the DF order to reschedule
        ordered_ops = self.graph.ordered_ops
        for op in ordered_ops:
            if self.visited[op.opid]:
                continue
            self.tensor_in_SRAM = set()
            # Each layer is a block
            self.opids_in_block = set()
            # Steps 1: Fetch opids in the same layer
            same_layer_opids = self.collect_same_layer_opids_in_same_block(op.opid)
            self.opids_in_block.update(same_layer_opids)

            # Steps 2: Update the schedule order
            re_order_opids = sorted(self.opids_in_block, key = lambda x: self.graph.ops[x].schedule_order)
            for opid in re_order_opids:
                self.layer_wise_mapping[opid] = self.new_order
                self.new_order += 1
                self.scheduled[opid] = True

            # Steps 3: Update the tensor metadata (tensor's storage area)
            opcode_index = op.info.get('opcode_index')
            opcode_type = self.graph.opcodes[opcode_index].get('builtin_code')
            # Operation that will execute in CPU
            if opcode_type in fall_back_cpu_ops:
                for opid in same_layer_opids:
                    # Update each opid's tensor metadata
                    op_info = self.graph.ops[opid].info
                    # Input tensor
                    for tensor_id in op_info['inputs']:
                        # Skip the tensor that related to the split_axis or permutation
                        buffer = self.graph.buffers[self.graph.tensors[tensor_id]['buffer']]
                        if len(buffer) != 0:
                            continue
                        # Update the tensor metadata
                        for tensor in self.tensor_info[tensor_id].tensors:
                            if tensor.cid == opid:
                                tensor.in_DRAM = True
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        # Update the tensor metadata
                        for tensor in self.tensor_info[tensor_id].tensors:
                            # Store the output tensor in the DRAM
                            if tensor.pid == opid:
                                tensor.in_DRAM = True
            # Operation that will execute in NPU's element-wise unit
            elif opcode_type in elem_wise_ops:
                for opid in same_layer_opids:
                    op_info = self.graph.ops[opid].info
                    # Input tensor
                    unary, binary = 1, 2
                    input_nums = [unary, binary]
                    if opcode_type in unary_ops:
                        input_idx = 0
                        if opcode_type in use_lut_ops:
                            input_idx = 1
                    elif opcode_type in binary_ops:
                        input_idx = 1
                    for tensor_id in op_info['inputs'][0:input_nums[input_idx]]:
                        if tensor_id == -1:
                            continue
                        # Set the tensor metadata, if the tensor had allocated in the SRAM
                        if tensor_id in self.tensor_in_SRAM:
                            for tensor_metadata in self.tensor_info[tensor_id].tensors:
                                if tensor_metadata.cid == opid:
                                    tensor_metadata.in_DRAM = False
                        # The tensor may cross the block (in previous block in_DRAM := False), so need to set the tensor's in_DRAM
                        else:
                            for tensor_metadata in self.tensor_info[tensor_id].tensors:
                                if tensor_metadata.cid == opid:
                                    tensor_metadata.in_DRAM = True
                            self.tensor_in_SRAM.add(tensor_id)
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        # Update the tensor metadata
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.pid == opid:
                                tensor_metadata.in_DRAM = False
                        self.tensor_in_SRAM.add(tensor_id)
            # Operation that will execute in NPU's MAC unit
            elif opcode_type in mac_ops:
                for opid in same_layer_opids:
                    op_info = self.graph.ops[opid].info
                    # Input tensor
                    unary, binary, trinary = 1, 2, 3
                    input_nums = [unary, binary, trinary]
                    if opcode_type in unary_ops:
                        input_idx = 0
                    elif opcode_type in binary_ops:
                        input_idx = 1
                    elif opcode_type in trinary_ops:
                        input_idx = 2
                    for tensor_id in op_info['inputs'][0:input_nums[input_idx]]:
                        # Sometimes the tensor maybe empty, ex: no bias
                        if tensor_id == -1:
                            continue
                        # Set the tensor metadata, if the tensor had allocated in the SRAM
                        if tensor_id in self.tensor_in_SRAM:
                            for tensor_metadata in self.tensor_info[tensor_id].tensors:
                                if tensor_metadata.cid == opid:
                                    tensor_metadata.in_DRAM = False
                        # The tensor may cross the block (in previous block in_DRAM := False), so need to set the tensor's in_DRAM
                        else:
                            for tensor_metadata in self.tensor_info[tensor_id].tensors:
                                if tensor_metadata.cid == opid:
                                    tensor_metadata.in_DRAM = True
                            self.tensor_in_SRAM.add(tensor_id)
                    # Output tensor
                    for tensor_id in op_info['outputs']:
                        # Update the tensor metadata
                        for tensor_metadata in self.tensor_info[tensor_id].tensors:
                            if tensor_metadata.pid == opid:
                                tensor_metadata.in_DRAM = False
                        self.tensor_in_SRAM.add(tensor_id)
        # Step 4: Update the order of the operations
        for op in self.graph.ops:
            op.schedule_order = self.layer_wise_mapping[op.opid]
        # Update the schedule order in the ordered_ops
        self.graph.ordered_ops = sorted(self.graph.ops, key=lambda x: x.schedule_order)
        # Update the operators in the split_graph
        self.graph.operators = []
        for op in self.graph.ordered_ops:
            self.graph.operators.append(op.info)
        return self.graph

    def collect_same_layer_opids_in_same_block(self, opid):
        collect_opids = []
        head_opid = self.same_layer_next_opids.get(opid, [-1, -1, 1])[0]

        # Non splitted op's head_opid is -1
        if head_opid == -1 and not self.scheduled[opid]:
            collect_opids.append(opid)
            self.visited[opid] = True
            return collect_opids
        
        next_opid = head_opid
        while next_opid != -1:
            if not self.scheduled[next_opid]:
                collect_opids.append(next_opid)
                self.visited[next_opid] = True
            next_opid = self.same_layer_next_opids.get(next_opid, [-1, -1, 1])[1]
        return collect_opids