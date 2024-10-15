def fully_connected_codegen(operator):
    print(f"FULLY_CONNECTED: {operator['inputs']}, {operator['outputs']}")

def batch_matmul_codegen(operator):
    print(f"BATCH_MATMUL: {operator['inputs']}, {operator['outputs']}")

def mul_codegen(operator):
    print(f"MUL: {operator['inputs']}, {operator['outputs']}")

def add_codegen(operator):
    print(f"ADD: {operator['inputs']}, {operator['outputs']}")

def softmax_codegen(operator):
    print(f"SOFTMAX: {operator['inputs']}, {operator['outputs']}")

def concat_codegen(operator):
    print(f"CONCAT: {operator['inputs']}, {operator['outputs']}")

def split_codegen(operator):
    print(f"SPLIT: {operator['inputs']}, {operator['outputs']}")

def op_codegen(operator):
    if operator['builtin_options_type'] == 'FullyConnectedOptions':
        fully_connected_codegen(operator)
    elif operator['builtin_options_type'] == 'BatchMatMulOptions':
        batch_matmul_codegen(operator)
    elif operator['builtin_options_type'] == 'MulOptions':
        mul_codegen(operator)
    elif operator['builtin_options_type'] == 'AddOptions':
        add_codegen(operator)
    elif operator['builtin_options_type'] == 'SoftmaxOptions':
        softmax_codegen(operator)
    elif operator['builtin_options_type'] == 'ConcatenationOptions':
        concat_codegen(operator)
    elif operator['builtin_options_type'] == 'SplitOptions':
        split_codegen(operator)
    else:
        print(f"[CODE_GEN] Unknown operator: {operator['builtin_options_type']}")