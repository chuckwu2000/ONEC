class Op_Classify:
    def __init__(self):
        # The operation that will be fall back to CPU
        self.data_layout_ops = ["CONCATENATION", "SPLIT", "SPLIT_V", "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR", "PACK", "RESHAPE", "SLICE"]
        self.fall_back_cpu_ops = list(set(self.data_layout_ops) - set(["RESHAPE"]))

        # MAC main operation
        self.mac_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "MEAN", "MAX_POOL_2D", "BATCH_MATMUL", "REDUCE_MAX", "SUM"]

        # The operation that need weights
        self.need_weights_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL"]
        # The operation that need bias
        self.need_bias_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]

        # Element-wise main operation
        self.elementwise_ops = ["LOGISTIC", "SOFTMAX", "RSQRT", "POW", "TANH", "GELU", "QUANTIZE", "DEQUANTIZE", "EXP", \
                                "ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE"]

        # The operation that need requantization
        self.need_requant_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "MEAN", "BATCH_MATMUL", "SUM", "LOGISTIC", \
                            "SOFTMAX", "RSQRT", "TANH", "GELU", "EXP", "ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE"]

        # The operation that need dequantization
        self.need_dequant_ops = ["LOGISTIC", "SOFTMAX", "RSQRT", "TANH", "GELU"]

        # The operation that need reduce
        self.reduce_ops = ["MEAN", "REDUCE_MAX", "SOFTMAX", "SUM"]

        # The input of the operation
        self.unary_ops = ["LOGISTIC", "RSQRT", "SOFTMAX", "GELU", "LEAKY_RELU", "REDUCE_MAX", "QUANTIZE", "DEQUANTIZE", \
                          "TANH", "POW", "MEAN", "MAX_POOL_2D", "SUM", "EXP"]
        self.binary_ops = ["ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "BATCH_MATMUL"]
        self.trinary_ops = ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"]