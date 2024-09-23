from MyGraph import Graph
from Architecture_feature import Mem_area
from Architecture_feature import ArchitectureFeatures

# Estimate the number of cycles for generating an element (reference to _estimate_output_cycles_per_element)
def estimte_output_cycles_per_element():
    pass

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

    # TODO Computations cycles
    cycle_per_elem = ArchitectureFeatures.output_cycles_per_elem["ADD/SUB"]
    op_cycles = ofm_shape[0] * ofm_shape[1] * ofm_shape[2] * ofm_shape[3] * cycle_per_elem

    total_cycles = dma_transfer_cycles + op_cycles
    return total_cycles

# Estimate the number of cycles for a given convolution operation (reference to _estimate_conv_cycles)
def estimate_conv_cycles():
    pass

# Estimate the number of cycles for a given depthwise convolution operation (reference to _estimate_depthwise_conv_cycles)
def estimate_depthwise_conv_cycles():
    pass

# Estimate the number of cycles for memory to memory transfer (reference to measure_mem2mem_cycles)
def estimate_mem2mem_cycles(src_tensor_mem_area, dst_tensor_mem_area, transfer_size) -> int:
    if src_tensor_mem_area == Mem_area.DRAM and dst_tensor_mem_area == Mem_area.SRAM:
        bws_per_cycle = (ArchitectureFeatures.axi_bit_width / 8) * ArchitectureFeatures.Dram_clock_scale
        transfer_cycles = transfer_size / bws_per_cycle
    return transfer_cycles

# Estimate the number of cycles for a given operation (reference to estimate_full_op_performance)
def estimate_op_cycles(model: Graph, opid: int) -> int:
    opcode_type = model.ops[opid].info.get("builtin_options_type")
    if opcode_type == "AddOptions":
        op_cycles = estimate_add_cycles(model, opid)
    elif opcode_type == "Conv2DOptions":
        op_cycles = estimate_conv_cycles(model, opid)
    elif opcode_type == "DepthwiseConv2DOptions":
        op_cycles = estimate_depthwise_conv_cycles(model, opid)
    else:
        op_cycles = 0
        print("Not yet supported {opcode_type}'s cycle estimation")
    return op_cycles

def print_performance():
    pass