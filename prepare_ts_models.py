# The project's entry point

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rewriter_path", nargs = '?', type = str, default = "Compiler.py")
parser.add_argument("--schema_path", nargs = '?', type = str, default = "utils/schema.fbs")
parser.add_argument("--models_dir", nargs = '?', type = str, default = "models/")
parser.add_argument("--split_size", nargs = '?', type = int, default = 50)
parser.add_argument("--token_size", nargs = '?', type = int, default = 50)
parser.add_argument("--model_type", nargs = '?', type = str, default = "bert")
parser.add_argument("--verbose_performance", action = 'store_true')
parser.add_argument("--pad_fusion", action = 'store_true')
parser.add_argument("--remove_data_layout_op", action = 'store_true')
parser.add_argument("--cancel_move_data_layout_op", action = 'store_true')
parser.add_argument("--cancel_lowering", action = 'store_true')
parser.add_argument("--unify_sram", action = 'store_true')
parser.add_argument("--codegen", action = 'store_true')
parser.add_argument("--tandem", action = 'store_true')
args = parser.parse_args()
rewriter_path = str(args.rewriter_path)
schema_path = str(args.schema_path)
models_dir = str(args.models_dir)
split_size = int(args.split_size)
token_size = int(args.token_size)
model_type = str(args.model_type)

tflite_path = models_dir
model_name = os.path.splitext(os.path.basename(tflite_path))[0]
folder_root = os.path.dirname(tflite_path).replace('tflite/', 'ts_model/')
for split_height in [split_size]:
    for exec_order in ['DF']:
        subfolder_dir = os.path.join(folder_root, f"{model_name}_fuse_split_{split_height}")
        os.makedirs(subfolder_dir, exist_ok = True)
        # out_path: optimized tflite model
        out_path = f"{subfolder_dir}/{model_name}_fuse_split_{split_height}.tflite"
        # code_path: generated code
        code_path = f"{subfolder_dir}/{model_name}_fuse_split_{split_height}_code.txt"
        print(out_path)
        cmd =   f"python {rewriter_path} {tflite_path} --schema_path {schema_path}" \
                f" --exec_order {exec_order} --split_height {split_height} --token_size {token_size}" \
                f" --model_type {model_type}" \
                f" --out_path {out_path}"
        if args.pad_fusion:
            cmd += " --pad_fusion"
        if args.verbose_performance:
            cmd += " --verbose_performance"
        if args.remove_data_layout_op:
            cmd += " --remove_data_layout_op"
        # Notice that the cancel_move_data_layout_op will interfere with the baseline performance
        if not args.cancel_move_data_layout_op and not args.tandem:
            cmd += " --move_data_layout_op"
        if not args.cancel_lowering:
            cmd += " --softmax_lowering"
            cmd += " --mean_convert"
            cmd += " --logistic_lowering"
        if args.unify_sram:
            cmd += " --unify_sram"
        if args.codegen:
            cmd += " --codegen"
            cmd += f" --code_path {code_path}"
        if args.tandem:
            cmd += " --tandem"
        subprocess.run(cmd.split(' '))
