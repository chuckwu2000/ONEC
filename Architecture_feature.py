from enum import Enum

class Mem_area(Enum):
    SRAM = 0
    DRAM = 1
    OffChipFlash = 2

class ArchitectureFeatures:
    # Assume bandwidth in SRAM (4 GB/s) and DRAM (4 GB/s) and Flash (0.5 GB/s)
    axi_bit_width = 64
    core_clock = 500e6
    Sram_clock_scale = 1
    Sram_burst_length = 32
    Dram_clock_scale = 1
    Dram_burst_length = 128
    OffChipFlash_clock_scale = 0.125
    OffChipFlash_burst_length = 32

    # Output cycles per element (MAC main op, mul, add/sub)
    # Now assune we have 128 MACs
    output_cycles_per_elem = {"MAC": 1 / 128, "MUL": 2, "ADD/SUB": 1}