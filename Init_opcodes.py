class Init_opcodes:
    def __init__(self, opcodes, new_opcodes, codegen: bool):
        self.opcodes = opcodes
        self.new_opcodes = new_opcodes
        self.codegen = codegen

    def init_opcodes(self):
        # For tensor splitting
        has_split = False
        has_concat = False
        # For lowering the sotfmax op & logistic op
        has_max_pool = False
        has_sub = False
        has_exp = False
        has_conv = False
        has_reciprocal = False
        has_mul = False
        # For convert the mean op
        has_depthwise_conv2d = False
        # For convert the tanh op
        has_logistic = False
        # For convert the gelu op
        has_reshape = False

        for opcode in self.opcodes:
            if opcode.get('deprecated_builtin_code',0) == 2:
                has_concat = True
            elif opcode.get('deprecated_builtin_code',0) == 49:
                has_split = True
            elif opcode.get('deprecated_builtin_code',0) == 17:
                has_max_pool = True
            elif opcode.get('deprecated_builtin_code',0) == 41:
                has_sub = True
            elif opcode.get('deprecated_builtin_code',0) == 47:
                has_exp = True
            elif opcode.get('deprecated_builtin_code',0) == 3:
                has_conv = True
            elif opcode.get('deprecated_builtin_code',0) == 124:
                has_reciprocal = True
            elif opcode.get('deprecated_builtin_code',0) == 18:
                has_mul = True
            elif opcode.get('deprecated_builtin_code',0) == 4:
                has_depthwise_conv2d = True
            elif opcode.get('deprecated_builtin_code',0) == 14:
                has_logistic = True
            elif opcode.get('deprecated_builtin_code',0) == 22:
                has_reshape = True

        if has_concat == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 2,
                "version": 1,
                "builtin_code": "CONCATENATION"
            })
        if has_split == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 49,
                "version": 1,
                "builtin_code": "SPLIT"
            })
        if has_max_pool == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 17,
                "version": 1,
                "builtin_code": "MAX_POOL_2D"
            })
        if has_sub == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 41,
                "version": 1,
                "builtin_code": "SUB"
            })
        if has_exp == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 47,
                "version": 1,
                "builtin_code": "EXP"
            })
        if has_conv == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 3,
                "version": 1,
                "builtin_code": "CONV_2D"
            })
        if has_reciprocal == False:
            self.new_opcodes.append({
                # This op is not in the official schema
                "deprecated_builtin_code": 124,
                "version": 1,
                "builtin_code": "RECIPROCAL"
            })
        if has_mul == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 18,
                "version": 1,
                "builtin_code": "MUL"
            })
        if has_depthwise_conv2d == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 4,
                "version": 1,
                "builtin_code": "DEPTHWISE_CONV_2D"
            })
        if has_logistic == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 14,
                "version": 1,
                "builtin_code": "LOGISTIC"
            })
        if has_reshape == False:
            self.new_opcodes.append({
                "deprecated_builtin_code": 22,
                "version": 1,
                "builtin_code": "RESHAPE"
            })