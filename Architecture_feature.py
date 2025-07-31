# Related to the architecture features of the NPU

class ArchitectureFeatures:
    core_clock = 3e8
    core_period = 1 / core_clock * 1e9  # in ns
    # Assume bandwidth in SRAM is 10 times faster than DRAM
    Sram_clock_scale = 10
    Dram_clock_scale = 1
    
    # MAC total PEs (MAC_height * MAC_width)
    MAC_height = 128
    MAC_width = 128

    # DRAM's max size is 4GB
    DRAM_MAX_SIZE = 1 << 32
    # SRAM's max size is 4MB
    double_buffer = 1
    SRAM_MAX_SIZE = int(1024 * 1024 * 4 / double_buffer)
    SRAM_MAX_BANK = 64
    SRAM_BANK_SIZE = int(SRAM_MAX_SIZE / SRAM_MAX_BANK)

    # Output cycles per element (MAC main op, mul, add/sub)
    # Exponential, Reciprocal, Dequantize, Quantize can perform pipeline in our NPU hardware design
    output_cycles_per_elem = {
        "MAC": 1,
        "MUL": 1,
        "ADD/SUB": 1,
        "EXP": 1,
        "RECIPROCAL": 1,
        # Logistic(x) = 1 / (1 + exp(-x)) => sub + exp + reciprocal
        "LOGISTIC": 3,
        "DE/QUANTIZE": 1,
        "LUT": 1
    }

    # Element-wise operation vectorization
    VECTOR_LEN = 128

    # SRAM access energy (nJ/bit)
    sram_cost = 0.2 * 1e-3
    # MAC engine energy (nJ/cycle)
    mac_cost = 4.48
    # Vector engine energy (nJ/cycle)
    vector_cost = 0.1 * VECTOR_LEN * 1e-3

    # DRAM config yaml [related to OEM_wrappers/main.cc]
    config_path = "extern/ramulator2/OEM_DDR5_config.yaml"
    # config_path = "extern/ramulator2/OEM_DDR4_config.yaml"
    # config_path = "extern/ramulator2/OEM_HBM_config.yaml"