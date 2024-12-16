from enum import Enum

class Mem_area(Enum):
    PE = 0
    SRAM = 1
    DRAM = 2
    OffChipFlash = 3

class ArchitectureFeatures:
    # Assume bandwidth in SRAM (assume 10 times faster) and DRAM (16 GB/s)
    # DDR4 DRAM always 64-bit wide
    axi_bit_width = 64
    core_clock = 2e9
    Sram_clock_scale = 10
    # Sram_burst_length = 32
    Dram_clock_scale = 1
    # Dram_burst_length = 128
    MAC_PE = 64

    # Output cycles per element (MAC main op, mul, add/sub)
    # Now assune we have 64 MACs
    output_cycles_per_elem = {
        "MAC": 1 / MAC_PE,
        "MUL": 3,
        "ADD/SUB": 3,
        "POW": 3,
        "TANH": 3,
        "LOGISTIC": 3,
        "LEAKY_RELU": 3,
        "SOFTMAX": 10,
        "RSQRT": 9,
        "REDUCE_MAX": 1,
        "DE/QUANTIZE": 4,
        "LUT": 1
    }