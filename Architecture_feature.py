from enum import Enum

class Mem_area(Enum):
    PE = 0
    SRAM = 1
    DRAM = 2
    OffChipFlash = 3

class ArchitectureFeatures:
    # Assume bandwidth in SRAM (16 GB/s) and DRAM (1.6 GB/s)
    axi_bit_width = 128
    core_clock = 100e6
    Sram_clock_scale = 10
    # Sram_burst_length = 32
    Dram_clock_scale = 1
    # Dram_burst_length = 128
    MAC_PE = 256

    # Output cycles per element (MAC main op, mul, add/sub)
    # Now assune we have 64 MACs
    output_cycles_per_elem = {
        "MAC": 1 / MAC_PE,
        "MUL": 3,
        "ADD/SUB": 3,
        "LOGISTIC": 3,
        "LEAKY_RELU": 3,
        "SOFTMAX": 10,
        "RSQRT": 9,
        "REDUCE_MAX": 1,
        "DE/QUANTIZE": 4,
        "LUT": 1
    }