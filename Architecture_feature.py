from enum import Enum

class Mem_area(Enum):
    PE = 0
    SRAM = 1
    DRAM = 2
    OffChipFlash = 3

class ArchitectureFeatures:
    # Assume bandwidth in SRAM (assume 10 times faster) and DRAM (16 GB/s)
    # DDR4 DRAM's DIMM always 64-bit wide, and can bursts of 8 data words -> gives 64 bytes per burst 
    # DMA will fetch SRAM_burst_length(bytes) in one cycle
    axi_bit_width = 64
    core_clock = 2e9
    Sram_clock_scale = 10
    Sram_burst_length = 8
    Dram_clock_scale = 1
    Dram_burst_length = 8
    
    # MAC configuration is refer to Planaria
    MAC_height = 128
    MAC_width = 128

    # DRAM's max size is 4GB
    DRAM_MAX_SIZE = 1 << 32
    # SRAM's max size is 2MB
    SRAM_MAX_SIZE = 1 << 21

    # Output cycles per element (MAC main op, mul, add/sub)
    # Now assune we have 64 MACs
    # Exponential, Reciprocal, Dequantize, Quantize are perform pipeline in our design
    output_cycles_per_elem = {
        "MAC": 1,
        "MUL": 1,
        "ADD/SUB": 1,
        "EXP": 1,
        "RECIPROCAL": 1,
        "LOGISTIC": 3,
        "RSQRT": 4,
        "DE/QUANTIZE": 1,
        "LUT": 1
    }

    # Element-wise operation vectorization
    VECTOR_LEN = 128