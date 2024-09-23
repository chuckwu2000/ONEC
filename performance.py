from MyGraph import Graph
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures

# Estimate the number of cycles for a given add operation (reference to _estimate_add_cycles)
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
    return total_cycles

# Estimate the number of cycles for a given convolution operation (reference to _estimate_conv_cycles)
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
    return total_cycles

# Estimate the number of cycles for a given depthwise convolution operation (reference to _estimate_depthwise_conv_cycles)
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
    return total_cycles

# Estimate the number of cycles for memory to memory transfer (reference to measure_mem2mem_cycles)
def estimate_mem2mem_cycles(src_tensor_mem_area, dst_tensor_mem_area, transfer_size) -> int:
    if src_tensor_mem_area == Mem_area.OffChipFlash and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.OffChipFlash_clock_scale
        transfer_cycles = transfer_size / bws_per_cycle
    elif src_tensor_mem_area == Mem_area.DRAM and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
        transfer_cycles = transfer_size / bws_per_cycle
    return transfer_cycles

# Estimate the number of cycles for a given operation (reference to estimate_full_op_performance)
def estimate_op_cycles(model: Graph, opid: int) -> int:
    op = model.ops[opid]
    opcode_type = op.info.get("builtin_options_type")
    if opcode_type == "AddOptions":
        op_cycles = estimate_add_cycles(model, opid)
        op.estimated_cycles = op_cycles
    elif opcode_type == "Conv2DOptions":
        op_cycles = estimate_conv_cycles(model, opid)
        op.estimated_cycles = op_cycles
    elif opcode_type == "DepthwiseConv2DOptions":
        op_cycles = estimate_depthwise_conv_cycles(model, opid)
        op.estimated_cycles = op_cycles
    else:
        op_cycles = 0
        op.estimated_cycles = 0
        print(f"Not yet supported {opcode_type}'s cycle estimation")
    return op_cycles

def estimate_model(model: Graph) -> int:
    total_cycles = 0
    for opid in model.ordered_opid:
        total_cycles += estimate_op_cycles(model, opid)
    print_performance(model)
    return total_cycles

def print_performance(model: Graph):
    for opid in model.ordered_opid:
        op = model.ops[opid]
        opcode_type = op.info.get("builtin_options_type")
        op_cycles = op.estimated_cycles
        print(f"opcode_type: {opcode_type}, cycles: {op_cycles}")