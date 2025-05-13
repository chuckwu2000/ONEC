import numpy as np
import math
from collections import defaultdict
from MyGraph import Graph, Node
from Distributed_SRAM_allocator import Distributed_SRAM_tensor
from OpClassify import Op_Classify

op_classify = Op_Classify()
mac_ops = op_classify.mac_ops
unary_ops = op_classify.unary_ops
binary_ops = op_classify.binary_ops

op_mapping = {"IDLE": '0000', "CONV_2D": '0001', "FULLY_CONNECTED": '0010', "EXP": '0011', "RECIPROCAL": '0100', \
              "ADD": '0101', "SUB": '0110', "MUL": '0111'}

class OPGen:
    def __init__(self, graph : Graph, allocated_tensors : dict[int, Distributed_SRAM_tensor]):
        self.opcodes = graph.opcodes
        self.buffers = graph.buffers
        self.tensors = graph.tensors
        # Contain the tensor's storage info
        self.allocated_tensors = allocated_tensors
        self.op_code = ""
        self.op_gen_id = 0
        # Format: [4 ops, weight_num, sram_ids, 4 op_broadcast, 4 op_output_elements, op_0_input_elements + op_0_weight_elements + 3 op_weight_elements]
        self.op_launch_related = defaultdict(dict)
        self.op_metadata = defaultdict(str)

    def op_code_gen(self, ops : list[Node]):
        self.op_code = ""
        # Some op may be lowered to other ops
        total_ops = 0
        for op in ops:
            opcode_index = op.info.get("opcode_index")
            opcode_type = self.opcodes[opcode_index].get("builtin_code")
            if opcode_type == "LOGISTIC":
                total_ops += 3
            else:
                total_ops += 1
        
        # Our hardware now only support most 4 ops in one time
        if total_ops < 5:
            # Reset metadata index & op metadata
            self.op_gen_id = 0
            self.op_launch_related = defaultdict(dict)
            self.op_metadata = defaultdict(str)
            # First op's classify (influence number of sram used)
            first_op_index = ops[0].info.get("opcode_index")
            first_op_opcode_type = self.opcodes[first_op_index].get("builtin_code")
            first_op_is_mac = False
            first_op_is_unary_elementwise = False
            first_op_is_binary_elementwise = False
            if  first_op_opcode_type in mac_ops:
                first_op_is_mac = True
            elif first_op_opcode_type in unary_ops:
                first_op_is_unary_elementwise = True
            elif first_op_opcode_type in binary_ops:
                first_op_is_binary_elementwise = True
            # Codegen for each op
            for op in ops:
                opcode_index = op.info.get("opcode_index")
                opcode_type = self.opcodes[opcode_index].get("builtin_code")
                if opcode_type == 'FULLY_CONNECTED':
                    self.fully_connected_codegen(op)
                elif opcode_type == 'CONV_2D':
                    self.conv_codegen(op)
                elif opcode_type == 'ADD':
                    self.add_codegen(op)
                elif opcode_type == 'SUB':
                    self.sub_codegen(op)
                elif opcode_type == 'MUL':
                    self.mul_codegen(op)
                elif opcode_type == 'EXP':
                    self.exp_codegen(op)
                elif opcode_type == 'RECIPROCAL':
                    self.reciprocal_codegen(op)
                elif opcode_type == 'CONCATENATION':
                    self.concat_codegen(op)
                    return self.op_code
                elif opcode_type == 'SPLIT':
                    self.split_codegen(op)
                    return self.op_code
                elif opcode_type == 'RESHAPE':
                    return self.op_code
                else:
                    print(f"[CODE_GEN] Unknown op: {opcode_type}")
                    return self.op_code
        else:
            raise BaseException(f"[CODE_GEN] Too many ops, first op: {ops[0]}, total: {total_ops}")
        
        # Final codegen
        # Format: [4 ops, weight_num, sram_ids, 4 op_broadcast, 4 op_output_elements, op_0_input_elements + op_0_weight_elements + 3 op_weight_elements
        #          + each op_metadata]
        # ops (reversed order)
        for i in range(total_ops - 1, -1, -1):
            self.op_code += self.op_launch_related[i]["op_name"]
        self.op_code += " "
        # weight_num
        weights_num = 0
        for i in range(total_ops):
            weights_num += self.op_launch_related[i]["weight_num"]
        self.op_code += str(weights_num) + " "
        # sram_ids (store_sram_idx, op0_weight_idx0, op0_weight_idx1, op1_weight_idx0, op2_weight_idx0, op3_weight_idx0)
        # If not use: set to 0
        self.op_code += str(self.op_launch_related[total_ops - 1]["output_sram_id"]) + " "
        if first_op_is_mac or first_op_is_binary_elementwise:
            self.op_code += str(self.op_launch_related[0]["input0_sram_id"]) + " "
            self.op_code += str(self.op_launch_related[0]["input1_sram_id"]) + " "
        elif first_op_is_unary_elementwise:
            self.op_code += str(self.op_launch_related[0]["input0_sram_id"]) + " "
            self.op_code += str(0) + " "
        else:
            raise(f"[CODE_GEN] Unknown first op: {first_op_opcode_type}")
        for i in range(1, 4):
            if i < total_ops:
                # OEM's NPU default to put last op's output tensor to buffer and consume as second operand
                self.op_code += str(self.op_launch_related[i].get("input0_sram_id", 0)) + " "
            else:
                self.op_code += str(0) + " "
        # broadcast
        for i in range(total_ops):
            self.op_code += str(self.op_launch_related[i]["broadcast"]) + " "
        # output_elements
        for i in range(total_ops):
            self.op_code += str(self.op_launch_related[i]["output_elements"]) + " "
        # input_elements + weight_elements
        self.op_code += str(self.op_launch_related[0]["input_elements"]) + " "
        for i in range(0, total_ops):
            self.op_code += str(self.op_launch_related[i]["weight_elements"]) + " "
        # metadata
        for i in range(total_ops):
            self.op_code += self.op_metadata[i]
        return self.op_code
        
    def fully_connected_codegen(self, op):
        pass

    # Not consider bias tensor for now
    def conv_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["CONV_2D"]
        # Conv's output tensor may depend on other input tensors
        input_tensor_ids = []
        input_tensors = []
        pad_config = None
        
        if len(op.info['inputs']) < 4:
            # No need to consume other ops' output tensor (normally because the filter is 1x1)
            input_tensor_ids.append(op.info['inputs'][0])
        else:
            pad_config = self.buffers[self.tensors[op.info['inputs'][3]]['buffer']]['data']
            # Need to consume other ops' output tensor
            for i in range(4, len(op.info['inputs'])):
                input_tensor_ids.append(op.info['inputs'][i])
        filter_tensor_id = op.info['inputs'][1]
        output_tensor_id = op.info['outputs'][0]

        # Check whether have constant tensors
        conv_input_tensor_count = 0
        if len(self.buffers[self.tensors[filter_tensor_id]['buffer']]) != 0:
            conv_input_tensor_count += 1
        # Plus 1: input tensor
        self.op_launch_related[self.op_gen_id]['weight_num'] = conv_input_tensor_count + 1

        for input_tensor_id in input_tensor_ids:
            input_tensors.append(self.tensors[input_tensor_id])
        # This is temperal handling, actually need to perform data concatenation in the later stage
        input_tensor_id = input_tensor_ids[math.floor(len(input_tensor_ids) / 2)]
        filter_tensor = self.tensors[filter_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input_tensor = self.allocated_tensors[input_tensor_id]
        for tensor in allocated_input_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_filter_tensor = self.allocated_tensors[filter_tensor_id]
        for tensor in allocated_filter_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input1_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # Conv usually no need to consider broadcast
        # Need to broadcast, 1: weight 2: data (since OEM's NPU will flow the previous op's output to the second operand)
        self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        
        stride_h = op.info['builtin_options']['stride_h']
        stride_w = op.info['builtin_options']['stride_w']
        pad_h = pad_config[0]

        tokens = output_tensor['name'].split('_split_')
        output_id = -1
        if len(tokens) == 2:
            if tokens[1].isnumeric():
                output_id = int(tokens[1])
        elif len(tokens) > 2:
                raise(f"Invalid output tensor name: {output_tensor['name']}")
        
        # Initial the input tensor's range required
        split_size = self.tensors[op.info['inputs'][0]]['shape'][1]
        input_range_required = defaultdict(list)
        input_id_list = []
        for input_tensor_id in input_tensor_ids:
            input_tensor = self.tensors[input_tensor_id]
            tokens = input_tensor['name'].split('_split_')
            if tokens[1].isnumeric():
                input_id = int(tokens[1])
                # Put the input tensor's range reversed, for the following comparison to find the real boundry
                input_range_required[input_id] = [(input_id + 1) * split_size - 1, input_id * split_size]
                input_id_list.append(input_id)
        
        # Compute the start of the boundry of the input tensor
        for out_h in range(split_size * output_id, split_size * (output_id + 1)):
            in_h_origin = out_h * stride_h - pad_h
            for kernel_h in range(filter_tensor['shape'][1]):
                in_h = in_h_origin + kernel_h
                # OverBound case, no need to consider over the maximum boundry, since it has considered in the model_gen
                source_sid = in_h // split_size
                if in_h < 0 or source_sid not in input_range_required:
                    continue
                else:
                    # print(f"out_h: {out_h}, in_h: {in_h}, source_sid: {source_sid}")
                    input_range_required[source_sid][0] = min(input_range_required[source_sid][0], in_h)
                    input_range_required[source_sid][1] = max(input_range_required[source_sid][1], in_h)
        total_input_elements = 0
        input_tensor_idxs = []
        if len(op.info['inputs']) <= 5:
            input_tensor_idxs.append(0)
        else:
            for i in range(4, len(op.info['inputs'])):
                input_tensor_idxs.append(i)
        for input_tensor_idx in input_tensor_idxs:
            batch = input_tensors[input_tensor_idx]['shape'][0]
            # Since input_id_list start from 0, but the input_tensor_idx may start from 4 (because of our tiling)
            range_idx = input_tensor_idx - input_tensor_idxs[0]
            height = input_range_required[input_id_list[range_idx]][1] - input_range_required[input_id_list[range_idx]][0] + 1
            width = input_tensors[input_tensor_idx]['shape'][2]
            ic = input_tensors[input_tensor_idx]['shape'][3]
            total_input_elements += batch * height * width * ic
        self.op_launch_related[self.op_gen_id]['input_elements'] = total_input_elements
        self.op_launch_related[self.op_gen_id]['weight_elements'] = self.Compute_tensor_size(filter_tensor)
        self.op_launch_related[self.op_gen_id]['output_elements'] = self.Compute_tensor_size(output_tensor)

        # TODO: May try to concatenate input_range_required first
        for input_tensor_id, source_sid in zip(input_tensor_ids, input_id_list):
            pass

        input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
        input_zero_point = input_tensor['quantization']['zero_point'][0]
        weight_quant_scale = np.int32(filter_tensor['quantization']['scale'][0]).view('float32')
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        scale = input_quant_scale * weight_quant_scale / output_quant_scale
        multiplier, shift = self.QuantizeMultiplier(scale)

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(stride_h) + " "
        self.op_metadata[self.op_gen_id] += str(stride_w) + " "
        self.op_metadata[self.op_gen_id] += str(pad_h) + " "
        self.op_metadata[self.op_gen_id] += str(input_tensor['shape'][0]) + " "
        self.op_metadata[self.op_gen_id] += str(input_tensor['shape'][1]) + " "
        self.op_metadata[self.op_gen_id] += str(input_tensor['shape'][2]) + " "
        self.op_metadata[self.op_gen_id] += str(input_tensor['shape'][3]) + " "
        self.op_metadata[self.op_gen_id] += str(filter_tensor['shape'][0]) + " "
        self.op_metadata[self.op_gen_id] += str(filter_tensor['shape'][1]) + " "
        self.op_metadata[self.op_gen_id] += str(filter_tensor['shape'][2]) + " "
        self.op_metadata[self.op_gen_id] += str(multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point)

        # Update the op_gen_id
        self.op_gen_id += 1

    def add_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["ADD"]
        input1_tensor_id = op.info['inputs'][0]
        input2_tensor_id = op.info['inputs'][1]
        output_tensor_id = op.info['outputs'][0]

        # Check whether have constant tensors
        constant_tensor_count = 0
        if len(self.buffers[self.tensors[input1_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        if len(self.buffers[self.tensors[input2_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        self.op_launch_related[self.op_gen_id]['weight_num'] = constant_tensor_count

        input1_tensor = self.tensors[input1_tensor_id]
        input2_tensor = self.tensors[input2_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input1_tensor = self.allocated_tensors[input1_tensor_id]
        for tensor in allocated_input1_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_input2_tensor = self.allocated_tensors[input2_tensor_id]
        for tensor in allocated_input2_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input1_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # Check which tensor needs broadcast
        input1_size = self.Compute_tensor_size(input1_tensor)
        input2_size = self.Compute_tensor_size(input2_tensor)
        # Need to broadcast, 1: weight 2: data (since OEM's NPU will flow the previous op's output to the second operand)
        if input1_size > input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 2
        elif input1_size < input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 1
        else:
            # No need to broadcast
            self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        
        self.op_launch_related[self.op_gen_id]["input_elements"] = input2_size
        self.op_launch_related[self.op_gen_id]["weight_elements"] = input1_size
        self.op_launch_related[self.op_gen_id]["output_elements"] = self.Compute_tensor_size(output_tensor)
        
        input1_quant_scale = np.int32(input1_tensor['quantization']['scale'][0]).view('float32')
        input1_zero_point = input1_tensor['quantization']['zero_point'][0]
        input2_quant_scale = np.int32(input2_tensor['quantization']['scale'][0]).view('float32')
        input2_zero_point = input2_tensor['quantization']['zero_point'][0]
        twice_max_input_scale = 2 * max(input1_quant_scale, input2_quant_scale)
        real_input1_scale = input1_quant_scale / twice_max_input_scale
        real_input1_multiplier, real_input1_shift = self.QuantizeMultiplier(real_input1_scale)
        real_input2_scale = input2_quant_scale / twice_max_input_scale
        real_input2_multiplier, real_input2_shift = self.QuantizeMultiplier(real_input2_scale)
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        real_output_scale = twice_max_input_scale / ((1 << 20) * output_quant_scale)
        real_output_multiplier, real_output_shift = self.QuantizeMultiplier(real_output_scale)
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(input1_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(input2_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(20) + " "
        self.op_metadata[self.op_gen_id] += str(real_input1_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_input2_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_input1_shift) + " "
        self.op_metadata[self.op_gen_id] += str(real_input2_shift) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(-128) + " "
        self.op_metadata[self.op_gen_id] += str(127) + " "

        # Update the op_gen_id
        self.op_gen_id += 1

    def sub_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["SUB"]
        input1_tensor_id = op.info['inputs'][0]
        input2_tensor_id = op.info['inputs'][1]
        output_tensor_id = op.info['outputs'][0]

        # Check whether have constant tensors
        constant_tensor_count = 0
        if len(self.buffers[self.tensors[input1_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        if len(self.buffers[self.tensors[input2_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        self.op_launch_related[self.op_gen_id]['weight_num'] = constant_tensor_count

        input1_tensor = self.tensors[input1_tensor_id]
        input2_tensor = self.tensors[input2_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input1_tensor = self.allocated_tensors[input1_tensor_id]
        for tensor in allocated_input1_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_input2_tensor = self.allocated_tensors[input2_tensor_id]
        for tensor in allocated_input2_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input1_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # Check which tensor needs broadcast
        input1_size = self.Compute_tensor_size(input1_tensor)
        input2_size = self.Compute_tensor_size(input2_tensor)
        # Need to broadcast, 1: weight 2: data (since OEM's NPU will flow the previous op's output to the second operand)
        if input1_size > input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 2
        elif input1_size < input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 1
        else:
            # No need to broadcast
            self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        
        self.op_launch_related[self.op_gen_id]["input_elements"] = input2_size
        self.op_launch_related[self.op_gen_id]["weight_elements"] = input1_size
        self.op_launch_related[self.op_gen_id]["output_elements"] = self.Compute_tensor_size(output_tensor)

        input1_quant_scale = np.int32(input1_tensor['quantization']['scale'][0]).view('float32')
        input1_zero_point = input1_tensor['quantization']['zero_point'][0]
        input2_quant_scale = np.int32(input2_tensor['quantization']['scale'][0]).view('float32')
        input2_zero_point = input2_tensor['quantization']['zero_point'][0]
        twice_max_input_scale = 2 * max(input1_quant_scale, input2_quant_scale)
        real_input1_scale = input1_quant_scale / twice_max_input_scale
        real_input1_multiplier, real_input1_shift = self.QuantizeMultiplier(real_input1_scale)
        real_input2_scale = input2_quant_scale / twice_max_input_scale
        real_input2_multiplier, real_input2_shift = self.QuantizeMultiplier(real_input2_scale)
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        real_output_scale = twice_max_input_scale / ((1 << 20) * output_quant_scale)
        real_output_multiplier, real_output_shift = self.QuantizeMultiplier(real_output_scale)
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(input1_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(input2_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(20) + " "
        self.op_metadata[self.op_gen_id] += str(real_input1_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_input2_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_input1_shift) + " "
        self.op_metadata[self.op_gen_id] += str(real_input2_shift) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(-128) + " "
        self.op_metadata[self.op_gen_id] += str(127) + " "

        # Update the op_gen_id
        self.op_gen_id += 1

    def mul_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["MUL"]
        
        input1_tensor_id = op.info['inputs'][0]
        input2_tensor_id = op.info['inputs'][1]
        output_tensor_id = op.info['outputs'][0]

        # Check whether have constant tensors
        constant_tensor_count = 0
        if len(self.buffers[self.tensors[input1_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        if len(self.buffers[self.tensors[input2_tensor_id]['buffer']]) != 0:
            constant_tensor_count += 1
        self.op_launch_related[self.op_gen_id]['weight_num'] = constant_tensor_count

        input1_tensor = self.tensors[input1_tensor_id]
        input2_tensor = self.tensors[input2_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input1_tensor = self.allocated_tensors[input1_tensor_id]
        for tensor in allocated_input1_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_input2_tensor = self.allocated_tensors[input2_tensor_id]
        for tensor in allocated_input2_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input1_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # Check which tensor needs broadcast
        input1_size = self.Compute_tensor_size(input1_tensor)
        input2_size = self.Compute_tensor_size(input2_tensor)
        # Need to broadcast, 1: weight 2: data (since OEM's NPU will flow the previous op's output to the second operand)
        if input1_size > input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 2
        elif input1_size < input2_size:
            self.op_launch_related[self.op_gen_id]["broadcast"] = 1
        else:
            # No need to broadcast
            self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        self.op_launch_related[self.op_gen_id]["input_elements"] = input2_size
        self.op_launch_related[self.op_gen_id]["weight_elements"] = input1_size
        self.op_launch_related[self.op_gen_id]["output_elements"] = self.Compute_tensor_size(output_tensor)

        input1_quant_scale = np.int32(input1_tensor['quantization']['scale'][0]).view('float32')
        input1_zero_point = input1_tensor['quantization']['zero_point'][0]
        input2_quant_scale = np.int32(input2_tensor['quantization']['scale'][0]).view('float32')
        input2_zero_point = input2_tensor['quantization']['zero_point'][0]
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        real_output_scale = input1_quant_scale * input2_quant_scale / output_quant_scale
        real_output_multiplier, real_output_shift = self.QuantizeMultiplier(real_output_scale)
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(input1_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(input2_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(real_output_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(-128) + " "
        self.op_metadata[self.op_gen_id] += str(127)

        # Update the op_gen_id
        self.op_gen_id += 1

    def exp_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["EXP"]
        input_tensor_id = op.info['inputs'][0]
        output_tensor_id = op.info['outputs'][0]

        # No way has constant tensors
        self.op_launch_related[self.op_gen_id]['weight_num'] = 0

        input_tensor = self.tensors[input_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input_tensor = self.allocated_tensors[input_tensor_id]
        for tensor in allocated_input_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # No way needs broadcast
        self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        self.op_launch_related[self.op_gen_id]["input_elements"] = self.Compute_tensor_size(input_tensor)
        self.op_launch_related[self.op_gen_id]["weight_elements"] = 0
        self.op_launch_related[self.op_gen_id]["output_elements"] = self.Compute_tensor_size(output_tensor)

        input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
        input_multiplier, input_shift = self.QuantizeMultiplier(input_quant_scale)
        input_zero_point = input_tensor['quantization']['zero_point'][0]
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        output_multiplier, output_shift = self.QuantizeMultiplier(output_quant_scale)
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(480) + " "
        self.op_metadata[self.op_gen_id] += str(input_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(input_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(input_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(output_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point) + " "

        # Update the op_gen_id
        self.op_gen_id += 1

    def reciprocal_codegen(self, op : Node):
        self.op_launch_related[self.op_gen_id]['op_name'] = op_mapping["RECIPROCAL"]
        input_tensor_id = op.info['inputs'][0]
        output_tensor_id = op.info['outputs'][0]

        # No way has constant tensors
        self.op_launch_related[self.op_gen_id]['weight_num'] = 0

        input_tensor = self.tensors[input_tensor_id]
        output_tensor = self.tensors[output_tensor_id]

        # Set distributed SRAM
        allocated_input_tensor = self.allocated_tensors[input_tensor_id]
        for tensor in allocated_input_tensor.tensors:
            if tensor.cid == op.opid:
                self.op_launch_related[self.op_gen_id]['input0_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break
        allocated_output_tensor = self.allocated_tensors[output_tensor_id]
        for tensor in allocated_output_tensor.tensors:
            if tensor.pid == op.opid:
                self.op_launch_related[self.op_gen_id]['output_sram_id'] = tensor.sram_id if tensor.sram_id != -1 else 0
                break

        # No way needs broadcast
        self.op_launch_related[self.op_gen_id]["broadcast"] = 0
        self.op_launch_related[self.op_gen_id]["input_elements"] = self.Compute_tensor_size(input_tensor)
        self.op_launch_related[self.op_gen_id]["weight_elements"] = 0
        self.op_launch_related[self.op_gen_id]["output_elements"] = self.Compute_tensor_size(output_tensor)

        input_quant_scale = np.int32(input_tensor['quantization']['scale'][0]).view('float32')
        input_multiplier, input_shift = self.QuantizeMultiplier(input_quant_scale)
        input_zero_point = input_tensor['quantization']['zero_point'][0]
        output_quant_scale = np.int32(output_tensor['quantization']['scale'][0]).view('float32')
        output_multiplier, output_shift = self.QuantizeMultiplier(output_quant_scale)
        output_zero_point = output_tensor['quantization']['zero_point'][0]

        # Set op metadata
        self.op_metadata[self.op_gen_id] += str(480) + " "
        self.op_metadata[self.op_gen_id] += str(input_zero_point) + " "
        self.op_metadata[self.op_gen_id] += str(input_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(input_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_multiplier) + " "
        self.op_metadata[self.op_gen_id] += str(output_shift) + " "
        self.op_metadata[self.op_gen_id] += str(output_zero_point) + " "

        # Update the op_gen_id
        self.op_gen_id += 1

    # Need to codegen RISCV code to move data in external memory
    def concat_codegen(self, op : Node):
        # print(f"CONCAT: {op.info['inputs']}, {op.info['outputs']}")
        pass

    def split_codegen(self, op : Node):
        # print(f"SPLIT: {op.info['inputs']}, {op.info['outputs']}")
        pass

    def QuantizeMultiplier(self, scale):
        if scale == 0.0:
            return 0, 0
        # q: [0.5, 1]
        q, shift = math.frexp(scale)
        q_fixed = np.int32(round(q * (1 << 31)))

        if q_fixed == (1 << 31):
            q_fixed /= 2
            shift += 1

        # If the shift is smaller than -31, all bits are shifted out, so the quantized value is 0
        if shift < -31:
            shift = 0
            q_fixed = 0
        return q_fixed, shift

    def Compute_tensor_size(self, tensor):
        shape = tensor['shape']
        size = 1
        for dim in shape:
            size *= dim
        return size