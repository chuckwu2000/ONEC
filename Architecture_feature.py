from enum import Enum

class Mem_area(Enum):
    SRAM = 0
    DRAM = 1
    OffChipFlash = 2

class ArchitectureFeatures:
    # Assume bandwidth in SRAM (4 GB/s) and DRAM (4 GB/s) and Flash (0.5 GB/s)
    # Assume bandwidth in DRAM (200 MB/s)
    axi_bit_width = 16
    core_clock = 100e6
    Sram_clock_scale = 10
    Sram_burst_length = 32
    Dram_clock_scale = 1
    Dram_burst_length = 128

    # Output cycles per element (MAC main op, mul, add/sub)
    # Now assune we have 64 MACs
    output_cycles_per_elem = {
        "MAC": 1 / 64,
        "MUL": 3,
        "ADD/SUB": 3,
        "LOGISTIC": 3,
        "LEAKY_RELU": 3,
        "SOFTMAX": 10,
        "RSQRT": 9,
        "LUT": 1
    }