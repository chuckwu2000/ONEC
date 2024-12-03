from MyGraph import Graph
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures
import math

data_layout_op = ["CONCATENATION", "SPLIT", "RESHAPE", "SPLIT_V", "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR"]

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
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = 1
    if ifm1_shape != []:
        for dim in ifm1_shape:
            ifm1_storge_size *= dim
        ifm1_storge_size *= (ifm1_elem_size / 8)
    ifm2_storge_size = 1
    if ifm2_shape != []:
        for dim in ifm2_shape:
            ifm2_storge_size *= dim
        ifm2_storge_size *= (ifm2_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given sub operation
def estimate_sub_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "Sub operation should have 2 inputs and 1 output"
    ifm1 = tensors[inputs[0]]
    ifm2 = tensors[inputs[1]]
    ofm = tensors[outputs[0]]
    ifm1_shape = ifm1.get("shape")
    ifm2_shape = ifm2.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = 1
    if ifm1_shape != []:
        for dim in ifm1_shape:
            ifm1_storge_size *= dim
        ifm1_storge_size *= (ifm1_elem_size / 8)
    ifm2_storge_size = 1
    if ifm2_shape != []:
        for dim in ifm2_shape:
            ifm2_storge_size *= dim
        ifm2_storge_size *= (ifm2_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
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
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = 1
    if ifm1_shape != []:
        for dim in ifm1_shape:
            ifm1_storge_size *= dim
        ifm1_storge_size *= (ifm1_elem_size / 8)
    ifm2_storge_size = 1
    if ifm2_shape != []:
        for dim in ifm2_shape:
            ifm2_storge_size *= dim
        ifm2_storge_size *= (ifm2_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
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
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)
    
    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles (Dequantize + Logistic + Quantize)
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"] * 2 + ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
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
    
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "IFM only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # ofm elements
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    
    # Filter tensor
    filter = tensors[inputs[1]]
    filter_shape = filter.get("shape")
    if filter.get("type") == "INT8":
        filter_elem_size = 8
    else:
        raise "Filter only support INT8 data type"
    # filter's size (bytes)
    filter_storge_size = 1
    for dim in filter_shape:
        filter_storge_size *= dim
    filter_storge_size *= (filter_elem_size / 8)
    
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
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        # Filter and bias are fetched from DRAM
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, filter_storge_size + bias_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, bias_storge_size + ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        sub_mac_size = math.floor(4 / filter_shape[1])
        # There may have multiple sub-mac engines (here, take 4x4 mac as smallest unit)
        output_per_MAC = ArchitectureFeatures.MAC_PE / 16 * (sub_mac_size * sub_mac_size)
        inner_product_size = filter_shape[1] * filter_shape[2]
        ifm_transfer_cycles_per_MAC = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size * output_per_MAC)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles +=  math.ceil(ofm_elems / output_per_MAC) * ifm_transfer_cycles_per_MAC
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + filter_storge_size + bias_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, bias_storge_size + ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        sub_mac_size = math.floor(4 / filter_shape[1])
        # There may have multiple sub-mac engines (here, take 4x4 mac as smallest unit)
        output_per_MAC = ArchitectureFeatures.MAC_PE / 16 * (sub_mac_size * sub_mac_size)
        inner_product_size = filter_shape[1] * filter_shape[2]
        ifm_transfer_cycles_per_MAC = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size * output_per_MAC)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles +=  math.ceil(ofm_elems / output_per_MAC) * ifm_transfer_cycles_per_MAC

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need ifm's channel * filtersize * filtersize MACs => filter_shape * ofm_height * ofm_width
    MACs = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * ofm_shape[1] * ofm_shape[2]
    op_cycles = MACs * cycle_per_elem

    # Requantize cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
    op_cycles += ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for a given depthwise convolution operation
# TODO: this estimation is not accurate
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
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)
    
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
    if model.pipeline_schedule:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, total_ifm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, weight_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, bias_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, total_ifm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, weight_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, bias_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need 1 * weight_size * weight_size MACs => weight_shape * ofm_height * ofm_width
    MACs = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] * ofm_shape[1] * ofm_shape[2]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for a given mean operation
def estimate_mean_cycles(model: Graph, opid: int) -> int:
    # Mean op will be converted to depthwise convolution + mul
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "Mean operation should have 2 inputs and 1 output"
    
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]

    # Only fetch first ifm in the list, since it has checked that all ifm have the same shape in the TS model
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "IFM only support INT8 data type"
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = 20
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given transpose convolution operation
# TODO: this estimation is not accurate
def estimate_trconv_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    # Not constraint the number of inputs, since in TS model, the input tensor is represent the dependence relationship, so it may have more than 1 inputs
    if len(outputs) != 1:
        raise "Transpose Conv2D operation should have at least 1 output"
    
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
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)
    
    # Filter tensor
    filter = tensors[inputs[1]]
    filter_shape = filter.get("shape")
    if filter.get("type") == "INT8":
        filter_elem_size = 8
    else:
        raise "Filter only support INT8 data type"
    # filter's size (bytes)
    filter_storge_size = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * (filter_elem_size / 8)

    # DMA transfer cycles
    total_ifm_storge_size = ifm_storge_size * len(ifm_list)
    if model.pipeline_schedule:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, total_ifm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, filter_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, total_ifm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, filter_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need ifm's channel * filtersize * filtersize MACs => filter_shape * ofm_height * ofm_width
    MACs = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * ofm_shape[1] * ofm_shape[2]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for a given maxpool operation
def estimate_maxpool_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    # Not constraint the number of inputs, since in TS model, the input tensor is represent the dependence relationship, so it may have more than 1 inputs
    if len(outputs) != 1:
        raise "MaxPool operation should have at least 1 output"
    
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # ofm elements
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim

    # kernel shape
    ker_shape = (info['builtin_options']['filter_height'], info['builtin_options']['filter_width'])

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        sub_mac_size = math.floor(8 / ker_shape[0])
        # There may have multiple sub-mac engines (here, take 4x4 mac as smallest unit)
        output_per_MAC = ArchitectureFeatures.MAC_PE / 64 * (sub_mac_size * sub_mac_size)
        inner_product_size = ker_shape[0] * ker_shape[1]
        ifm_transfer_cycles_per_MAC = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size * output_per_MAC)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles +=  math.ceil(ofm_elems / output_per_MAC) * ifm_transfer_cycles_per_MAC
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        sub_mac_size = math.floor(8 / ker_shape[0])
        # There may have multiple sub-mac engines (here, take 4x4 mac as smallest unit)
        output_per_MAC = ArchitectureFeatures.MAC_PE / 64 * (sub_mac_size * sub_mac_size)
        inner_product_size = ker_shape[0] * ker_shape[1]
        ifm_transfer_cycles_per_MAC = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size * output_per_MAC)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles +=  math.ceil(ofm_elems / output_per_MAC) * ifm_transfer_cycles_per_MAC

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need ifm's filtersize * filtersize MACs
    MACs = ker_shape[0] * ker_shape[1] * ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return (dma_transfer_cycles, op_cycles, total_cycles)

# Estimate the number of cycles for a given rsqrt operation
def estimate_rsqrt_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Rsqrt operation should have 1 inputs and 1 output"
    
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")

    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "IFM only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["RSQRT"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = op_cycles + dma_transfer_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given squared difference operation
def estimate_squared_difference_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "SquaredDifference operation should have 2 inputs and 1 output"
    ifm1 = tensors[inputs[0]]
    ifm2 = tensors[inputs[1]]
    ofm = tensors[outputs[0]]
    ifm1_shape = ifm1.get("shape")
    ifm2_shape = ifm2.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = 1
    if ifm1_shape != []:
        for dim in ifm1_shape:
            ifm1_storge_size *= dim
        ifm1_storge_size *= (ifm1_elem_size / 8)
    ifm2_storge_size = 1
    if ifm2_shape != []:
        for dim in ifm2_shape:
            ifm2_storge_size *= dim
        ifm2_storge_size *= (ifm2_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)
    
    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)

    # Computations cycles (x - y)(x - y)
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
    op_cycles += ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given gelu operation
def estimate_gelu_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Gelu operation should have 1 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles gelu(x)  = x * logistic(1.702 * x)
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LOGISTIC"]
    op_cycles += ofm_elems * cycle_per_elem
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MUL"]
    op_cycles += ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given leaky relu operation
def estimate_leaky_relu_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "LeakyRelu operation should have 1 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm1_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["LEAKY_RELU"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given fully connected operation
def estimate_fully_connected_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 3 or len(outputs) != 1:
        raise "FullyConnected operation should have 3 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # ofm elements
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim

    # weight tensor
    weight = tensors[inputs[1]]
    weight_shape = weight.get("shape")
    if weight.get("type") == "INT8":
        weight_elem_size = 8
    else:
        raise "Weight only support INT8 data type"
    # weight's size (bytes)
    weight_storge_size = weight_shape[0] * weight_shape[1] * (weight_elem_size / 8)

    # bias tensor
    # Assume bias tensor is None

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, weight_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        inner_product_size = ifm_shape[-1]
        one_ifm_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles += one_ifm_transfer_cycles * ofm_elems
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + weight_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        inner_product_size = ifm_shape[-1]
        one_ifm_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles += one_ifm_transfer_cycles * ofm_elems

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce #token * #feature elements, each element need weight's #feature MACs
    MACs = ofm_elems * weight_shape[1]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given softmax operation
def estimate_softmax_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Softmax operation should have 1 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["SOFTMAX"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given batch matmul operation
def estimate_batch_matmul_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "BatchMatmul operation should have 2 inputs and 1 output"
    ifm1 = tensors[inputs[0]]
    ifm2 = tensors[inputs[1]]
    ofm = tensors[outputs[0]]
    ifm1_shape = ifm1.get("shape")
    ifm2_shape = ifm2.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm1.get("type") == "INT8" and ifm2.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm1_elem_size = 8
        ifm2_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    # ifm's size (bytes)
    ifm1_storge_size = 1
    if ifm1_shape != []:
        for dim in ifm1_shape:
            ifm1_storge_size *= dim
        ifm1_storge_size *= (ifm1_elem_size / 8)
    ifm2_storge_size = 1
    if ifm2_shape != []:
        for dim in ifm2_shape:
            ifm2_storge_size *= dim
        ifm2_storge_size *= (ifm2_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # ofm's elements
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        # The weight tensor is stored in DRAM
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm2_storge_size) 
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        inner_product_size = ifm1_shape[3]
        one_ifm_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles += (inner_product_size / ArchitectureFeatures.MAC_PE) * one_ifm_transfer_cycles * ofm_elems
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm1_storge_size + ifm2_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ofm_storge_size)
        # Assume input tensors can be accessed in parallel, one output element needs inner_product_size MACs
        inner_product_size = ifm1_shape[3]
        one_ifm_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, inner_product_size)
        # Our MAC engine have 64 PEs
        dma_transfer_cycles += (inner_product_size / ArchitectureFeatures.MAC_PE) * one_ifm_transfer_cycles * ofm_elems

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["MAC"]
    # Total produce height * width * channel elements, each element need ifm1's channel MACs
    MACs = ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3] * ifm1_shape[3]
    op_cycles = MACs * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given reduce_max operation
def estimate_reduce_max_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 2 or len(outputs) != 1:
        raise "ReduceMax operation should have 2 inputs and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8" and ofm.get("type") == "INT8":
        ifm_elem_size = 8
        ofm_elem_size = 8
    else:
        raise "Only support INT8 data type"
    
    axis_buffer = model.buffers[model.tensors[info['inputs'][1]]['buffer']]
    axis = axis_buffer['data'][0]
    reduce_size = ifm_shape[axis]
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    # Assume each element need reduce_size - 1 cycles to compare
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["REDUCE_MAX"] * (reduce_size - 1)
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given quantize operation
def estimate_quantize_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Quantize operation should have 1 input and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8":
        ifm_elem_size = 8
    else:
        # We only support INT8 and float32 data type
        ifm_elem_size = 32
    if ofm.get("type") == "INT8":
        ofm_elem_size = 8
    else:
        # We only support INT8 and float32 data type
        ofm_elem_size = 32
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for a given dequantize operation
def estimate_dequantize_cycles(model: Graph, opid: int) -> int:
    tensors = model.tensors
    info = model.ops[opid].info
    inputs = info.get("inputs")
    outputs = info.get("outputs")

    if len(inputs) != 1 or len(outputs) != 1:
        raise "Quantize operation should have 1 input and 1 output"
    ifm = tensors[inputs[0]]
    ofm = tensors[outputs[0]]
    ifm_shape = ifm.get("shape")
    ofm_shape = ofm.get("shape")
    if ifm.get("type") == "INT8":
        ifm_elem_size = 8
    else:
        # We only support INT8 and float32 data type
        ifm_elem_size = 32
    if ofm.get("type") == "INT8":
        ofm_elem_size = 8
    else:
        # We only support INT8 and float32 data type
        ofm_elem_size = 32
    
    # ifm's size (bytes)
    ifm_storge_size = 1
    if ifm_shape != []:
        for dim in ifm_shape:
            ifm_storge_size *= dim
        ifm_storge_size *= (ifm_elem_size / 8)

    # ofm's size (bytes)
    ofm_storge_size = 1
    for dim in ofm_shape:
        ofm_storge_size *= dim
    ofm_storge_size *= (ofm_elem_size / 8)

    # DMA transfer cycles + IFM, OFM read/write cycles
    have_data_layout_parent = False
    for parent_op in model.ops[opid].parents:
        opcode_index = model.ops[parent_op].info.get("opcode_index")
        opcode_type = model.opcodes[opcode_index].get("builtin_code")
        if opcode_type in data_layout_op:
            have_data_layout_parent = True
            break
    if model.pipeline_schedule and not have_data_layout_parent:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)
    else:
        dma_transfer_cycles = estimate_mem2mem_cycles(Mem_area.DRAM, Mem_area.SRAM, ifm_storge_size + ofm_storge_size)
        dma_transfer_cycles += estimate_mem2mem_cycles(Mem_area.SRAM, Mem_area.PE, ifm_storge_size + ofm_storge_size)

    # Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["DE/QUANTIZE"]
    ofm_elems = 1
    for dim in ofm_shape:
        ofm_elems *= dim
    op_cycles = ofm_elems * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return dma_transfer_cycles, op_cycles, total_cycles

# Estimate the number of cycles for memory to memory transfer
def estimate_mem2mem_cycles(src_tensor_mem_area, dst_tensor_mem_area, transfer_size) -> int:
    if src_tensor_mem_area == Mem_area.DRAM and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
        transfer_cycles = math.ceil(transfer_size / bws_per_cycle)
    elif src_tensor_mem_area == Mem_area.SRAM and dst_tensor_mem_area == Mem_area.PE:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Sram_clock_scale
        transfer_cycles = math.ceil(transfer_size / bws_per_cycle)
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
    elif opcode_type == "SUB":
        dma_cycles, op_cycles, total_cycles = estimate_sub_cycles(model, opid)
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
    elif opcode_type == "MEAN":
        dma_cycles, op_cycles, total_cycles = estimate_mean_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "FULLY_CONNECTED":
        dma_cycles, op_cycles, total_cycles = estimate_fully_connected_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "SOFTMAX":
        dma_cycles, op_cycles, total_cycles = estimate_softmax_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "TRANSPOSE_CONV":
        dma_cycles, op_cycles, total_cycles = estimate_trconv_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "MAX_POOL_2D":
        dma_cycles, op_cycles, total_cycles = estimate_maxpool_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "RSQRT":
        dma_cycles, op_cycles, total_cycles = estimate_rsqrt_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "SQUARED_DIFFERENCE":
        dma_cycles, op_cycles, total_cycles = estimate_squared_difference_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "GELU":
        dma_cycles, op_cycles, total_cycles = estimate_gelu_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "LEAKY_RELU":
        dma_cycles, op_cycles, total_cycles = estimate_leaky_relu_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "BATCH_MATMUL":
        dma_cycles, op_cycles, total_cycles = estimate_batch_matmul_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "REDUCE_MAX":
        dma_cycles, op_cycles, total_cycles = estimate_reduce_max_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "QUANTIZE":
        dma_cycles, op_cycles, total_cycles = estimate_quantize_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "DEQUANTIZE":
        dma_cycles, op_cycles, total_cycles = estimate_dequantize_cycles(model, opid)
        op.estimated_DMA_cycles = dma_cycles
        op.estimated_op_cycles = op_cycles
        op.estimated_total_cycles = total_cycles
    elif opcode_type == "CONCATENATION" or opcode_type == "SPLIT" or opcode_type == "RESHAPE" or \
         opcode_type == "SPLIT_V" or opcode_type == "TRANSPOSE" or opcode_type == "RESIZE_NEAREST_NEIGHBOR" or \
         opcode_type == "PACK":
        # NPU won't do concatenation, so just set the cycles to 0
        dma_cycles, op_cycles, total_cycles = 0, 0, 0
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

    for opid in model.ordered_opid:
        # w/wo pipeline schedule, the estimated total cycles will be different, since we consider the memory footprint between DRAM and SRAM
        dma_cycles, op_cycles, op_total_cycles = estimate_op_cycles(model, opid)
        # For later pipeline schedule, op_total_cycles will be used to determine the overlap range
        if not pipeline:
            model.ops[opid].non_overlap_cycles = op_total_cycles
        total_dma_cycles += dma_cycles
        total_op_cycles += op_cycles
        total_cycles += op_total_cycles
    if pipeline:
        cascade_matched_ops = model.cascade_matched_ops
        # For now, assume the two op parallel execution will take the longer one
        for cascade_matched_op in cascade_matched_ops:
            op1 = model.ops[cascade_matched_op[0]]
            op2_estimated_DMA_cycles = 0
            op2_estimated_op_cycles = 0
            op2_estimated_total_cycles = 0
            for op_id in range(1, len(cascade_matched_op)):
                op2 = model.ops[cascade_matched_op[op_id]]
                op2_estimated_DMA_cycles += op2.estimated_DMA_cycles
                op2_estimated_op_cycles += op2.estimated_op_cycles
                op2_estimated_total_cycles += op2.estimated_total_cycles
            if op1.estimated_total_cycles > op2_estimated_total_cycles:
                total_dma_cycles -= op2_estimated_DMA_cycles
                total_op_cycles -= op2_estimated_op_cycles
                total_cycles -= op2_estimated_total_cycles
            else:
                total_dma_cycles -= op1.estimated_DMA_cycles
                total_op_cycles -= op1.estimated_op_cycles
                total_cycles -= op1.estimated_total_cycles
        matched_ops = model.matched_ops
        # For now, assume the two op parallel execution will take the longer one
        for matched_op in matched_ops:
            op1 = model.ops[matched_op[0]]
            op2_estimated_DMA_cycles = 0
            op2_estimated_op_cycles = 0
            op2_estimated_total_cycles = 0
            for op_id in range(1, len(matched_op)):
                op2 = model.ops[matched_op[op_id]]
                op2_estimated_DMA_cycles += op2.estimated_DMA_cycles
                op2_estimated_op_cycles += op2.estimated_op_cycles
                op2_estimated_total_cycles += op2.estimated_total_cycles
            if op1.estimated_total_cycles > op2_estimated_total_cycles:
                total_dma_cycles -= op2_estimated_DMA_cycles
                total_op_cycles -= op2_estimated_op_cycles
                total_cycles -= op2_estimated_total_cycles
            else:
                total_dma_cycles -= op1.estimated_DMA_cycles
                total_op_cycles -= op1.estimated_op_cycles
                total_cycles -= op1.estimated_total_cycles
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