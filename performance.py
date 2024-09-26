from MyGraph import Graph
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures

# Estimate the number of cycles for a given add operation
def estimate_add_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "Add operation should have 2 inputs and 1 output"
    ifm1 = tensors[inputs[0]]
    ifm2 = tensors[inputs[1]]
    ofm = tensors[outputs[0]]
    ifm1_shape = ifm1.get("shape")
    ifm2_shape = ifm2.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = ifm1_shape[0] * ifm1_shape[1] * ifm1_shape[2] * ifm1_shape[3] * (ifm1_elem_size / 8)
    ifm2_storge_size = ifm2_shape[0] * ifm2_shape[1] * ifm2_shape[2] * ifm2_shape[3] * (ifm2_elem_size / 8)

    # DMA transfer cycles
    dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
    ofm_elems = ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3]
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given mul operation
def estimate_mul_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "Mul operation should have 2 inputs and 1 output"
    ifm1 = tensors[inputs[0]]
    ifm2 = tensors[inputs[1]]
    ofm = tensors[outputs[0]]
    ifm1_shape = ifm1.get("shape")
    ifm2_shape = ifm2.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = ifm1_shape[0] * ifm1_shape[1] * ifm1_shape[2] * ifm1_shape[3] * (ifm1_elem_size / 8)
    ifm2_storge_size = ifm2_shape[0] * ifm2_shape[1] * ifm2_shape[2] * ifm2_shape[3] * (ifm2_elem_size / 8)

    # DMA transfer cycles
    dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
    ofm_elems = ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3]
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given logistic operation
def estimate_logistic_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Logistic operation should have 1 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8":
        ifm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = ifm_shape[0] * ifm_shape[1] * ifm_shape[2] * ifm_shape[3] * (ifm_elem_size / 8)

    # DMA transfer cycles
    dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ifm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"]
    ofm_elems = ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3]
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given convolution operation
def estimate_conv_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    # Not constraint the number of inputs, since in TS model, the input tensor is represent the dependence relationship, so it may have more than 1 inputs
    if len(outputs) != 1:
        raise "Conv2D operation should have at least 1 output"
    
    ifm_list = []
    for input_id in range (4, len(inputs)):
        ifm_list.append(tensors[inputs[input_id]])
    ifm = tensors[inputs[4]]
    ofm = tensors[outputs[0]]

    # Only fetch first ifm in the list, since it has checked that all ifm have the same shape in the TS model
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8":
        ifm_elem_size = 8
    else:
        raise "IFM only support INT8 data type"
    # ifm's size (bytes)
    ifm_storge_size = ifm_shape[0] * ifm_shape[1] * ifm_shape[2] * ifm_shape[3] * (ifm_elem_size / 8)
    
    # Filter tensor
    filter = tensors[inputs[1]]
    filter_shape = filter.get("shape")
    if filter.get("type") == "INT8":
        filter_elem_size = 8
    else:
        raise "Filter only support INT8 data type"
    # filter's size (bytes)
    filter_storge_size = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * (filter_elem_size / 8)
    
    # Bias tensor
    bias = tensors[inputs[2]]
    bias_shape = bias.get("shape")
    if bias.get("type") == "INT32":
        bias_elem_size = 32
    else:
        raise "Bias only support INT32 data type"
    # bias's size (bytes)
    bias_storge_size = bias_shape[0] * (bias_elem_size / 8)

    # DMA transfer cycles
    total_ifm_storge_size = ifm_storge_size * len(ifm_list)
    dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, total_ifm_storge_size)
    dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.OffChipFlash, Mem_area.SRAM, filter_storge_size)
    dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.OffChipFlash, Mem_area.SRAM, bias_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need ifm's channel * filtersize * filtersize MACs => filter_shape * ofm_height * ofm_width
    MACs = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * ofm_shape[1] * ofm_shape[2]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for a given depthwise convolution operation
def estimate_depthwise_conv_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    # Not constraint the number of inputs, since in TS model, the input tensor is represent the dependence relationship, so it may have more than 1 inputs
    if len(outputs) != 1:
        raise "DepthwiseConv2D operation should have at least 1 output"
    
    ifm_list = []
    for input_id in range (4, len(inputs)):
        ifm_list.append(tensors[inputs[input_id]])
    ifm = tensors[inputs[4]]
    ofm = tensors[outputs[0]]

    # Only fetch first ifm in the list, since it has checked that all ifm have the same shape in the TS model
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8":
        ifm_elem_size = 8
    else:
        raise "IFM only support INT8 data type"
    # ifm's size (bytes)
    ifm_storge_size = ifm_shape[0] * ifm_shape[1] * ifm_shape[2] * ifm_shape[3] * (ifm_elem_size / 8)
    
    # Weight tensor
    weight = tensors[inputs[1]]
    weight_shape = weight.get("shape")
    if weight.get("type") == "INT8":
        weight_elem_size = 8
    else:
        raise "Weights only support INT8 data type"
    # weight's size (bytes)
    weight_storge_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] * (weight_elem_size / 8)
    
    # Bias tensor
    bias = tensors[inputs[2]]
    bias_shape = bias.get("shape")
    if bias.get("type") == "INT32":
        bias_elem_size = 32
    else:
        raise "Bias only support INT32 data type"
    # bias's size (bytes)
    bias_storge_size = bias_shape[0] * (bias_elem_size / 8)

    # DMA transfer cycles
    total_ifm_storge_size = ifm_storge_size * len(ifm_list)
    dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, total_ifm_storge_size)
    dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.OffChipFlash, Mem_area.SRAM, weight_storge_size)
    dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.OffChipFlash, Mem_area.SRAM, bias_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need 1 * weight_size * weight_size MACs => weight_shape * ofm_height * ofm_width
    MACs = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] * ofm_shape[1] * ofm_shape[2]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for memory to memory transfer
def estimate_mem2mem_cycles(src_tensor_mem_area, dst_tensor_mem_area, transfer_size) -> int:
    if src_tensor_mem_area == Mem_area.OffChipFlash and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.OffChipFlash_clock_scale
        transfer_cycles = transfer_size / bws_per_cycle
    elif src_tensor_mem_area == Mem_area.DRAM and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
        transfer_cycles = transfer_size / bws_per_cycle
    return transfer_cycles

# Estimate the number of cycles for a given operation
def estimate_op_cycles(model: Graph, opid: int) -> int:
    op = model.ops[opid]
    opcode_index = op.info.get("opcode_index")
    opcode_type = model.opcodes[opcode_index].get("builtin_code")
    if opcode_type == "ADD":
        dma_cycles, op_cycles, total_cycles = estimate_add_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "MUL":
        dma_cycles, op_cycles, total_cycles = estimate_mul_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "LOGISTIC":
        dma_cycles, op_cycles, total_cycles = estimate_logistic_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "CONV_2D":
        dma_cycles, op_cycles, total_cycles = estimate_conv_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "DEPTHWISE_CONV_2D":
        dma_cycles, op_cycles, total_cycles = estimate_depthwise_conv_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    else:
        dma_cycles = 0
        op_cycles = 0
        total_cycles = 0
        print(f"Not yet supported {opcode_type}'s cycle estimation, its opcode_index is {opcode_index}")
    return dma_cycles, op_cycles, total_cycles

def estimate_model(model: Graph, pipeline: bool) -> int:
    total_dma_cycles = 0
    total_op_cycles = 0
    total_cycles = 0
    # Assume pipeline always happen after the model is splitted
    if pipeline:
        matched_ops = model.matched_ops
        for opid in model.ordered_opid:
            total_dma_cycles += model.ops[opid].estimated_DMA_cycles
            total_op_cycles += model.ops[opid].estimated_op_cycles
            total_cycles += model.ops[opid].estimated_total_cycles
        # For now, assume the two op parallel execution will take the longer one
        for matched_op in matched_ops:
            op1 = model.ops[matched_op[0]]
            op2 = model.ops[matched_op[1]]
            if op1.estimated_total_cycles > op2.estimated_total_cycles:
                total_dma_cycles -= op2.estimated_DMA_cycles
                total_op_cycles -= op2.estimated_op_cycles
                total_cycles -= op2.estimated_total_cycles
            else:
                total_dma_cycles -= op1.estimated_DMA_cycles
                total_op_cycles -= op1.estimated_op_cycles
                total_cycles -= op1.estimated_total_cycles
    else:
        for opid in model.ordered_opid:
            dma_cycles, op_cycles, op_total_cycles = estimate_op_cycles(model, opid)
            total_dma_cycles += dma_cycles
            total_op_cycles += op_cycles
            total_cycles += op_total_cycles
    return total_dma_cycles, total_op_cycles, total_cycles

def print_performance(model: Graph):
    for order, opid in enumerate(model.ordered_opid):
        op = model.ops[opid]
        opcode_index = op.info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        dma_cycles = op.estimated_DMA_cycles
        op_cycles = op.estimated_op_cycles
        op_total_cycles = op.estimated_total_cycles
        print(f"opcode_index: {opid}, order: {order}, opcode_type: {opcode_type}, DMA cycles: {dma_cycles}, OP cycles: {op_cycles}, Total cycles: {op_total_cycles}")