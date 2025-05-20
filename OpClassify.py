class Op_Classify:
    def __init__(self):
        # The operation that will be fall back to CPU
        self.data_layout_ops = ["CONCATENATION", "SPLIT", "SPLIT_V", "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR", "PACK", "RESHAPE", "SLICE"]
        self.cpu_ops = ["REDUCE_MAX", "SUM"]
        self.fall_back_cpu_ops = list(set(self.data_layout_ops) - set(["RESHAPE"]))

        # MAC main operation
        self.mac_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "MAX_POOL_2D", "BATCH_MATMUL"]

        # The operation that need weights
        self.need_weights_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL"]
        # The operation that need bias
        self.need_bias_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]

        # Element-wise main operation
        self.elementwise_ops = ["RSQRT", "POW", "TANH", "GELU", "QUANTIZE", "DEQUANTIZE", "EXP", \
                                "ADD", "SUB", "MUL", "RECIPROCAL", "SQUARED_DIFFERENCE"]

        # The operation that need reduce
        self.reduce_ops = ["REDUCE_MAX", "SUM", "SOFTMAX"]

        # The input of the operation
        self.unary_ops = ["RSQRT", "GELU", "QUANTIZE", "DEQUANTIZE", \
                          "TANH", "POW", "MAX_POOL_2D", "EXP", "RECIPROCAL"]
        self.binary_ops = ["ADD", "SUB", "MUL", "SQUARED_DIFFERENCE", "BATCH_MATMUL"]
        self.trinary_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]