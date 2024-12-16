import glob
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rewriter_path", nargs='?', type=str,
                    default="Compiler.py")
parser.add_argument("--schema_path", nargs='?', type=str,
                    default="utils/schema.fbs")
parser.add_argument("--models_dir", nargs='?', type=str,
                    default="models/")
parser.add_argument("--split_size", nargs='?', type=int, default=50)
parser.add_argument("--token_size", nargs='?', type=int, default=50)
parser.add_argument("--model_type", nargs='?', type=str, 
                    default="bert")
parser.add_argument("--verbose_performance", action='store_true')
parser.add_argument("--block_based", action='store_true')
args = parser.parse_args()
rewriter_path = str(args.rewriter_path)
schema_path = str(args.schema_path)
models_dir = str(args.models_dir)
split_size = int(args.split_size)
token_size = int(args.token_size)
model_type = str(args.model_type)

# tflite_list = [path for path in glob.glob(f'{models_dir}/*.tflite', recursive=True)]
# for tflite_path in sorted(tflite_list):
tflite_path = models_dir
model_name = os.path.splitext(os.path.basename(tflite_path))[0]
folder_root = os.path.dirname(tflite_path).replace('tflite/', 'ts_model/')
for split_height in [split_size]:
    for exec_order in ['DF']:
        subfolder_dir = os.path.join(folder_root, f"{model_name}_{exec_order}_{split_height}")
        os.makedirs(subfolder_dir, exist_ok=True)
        out_path = f"{subfolder_dir}/{model_name}_splitted_{exec_order}_{split_height}.tflite"
        print(out_path)
        cmd =   f"python {rewriter_path} {tflite_path} --schema_path {schema_path}" \
                f" --exec_order {exec_order} --split_height {split_height} --token_size {token_size}" \
                f" --model_type {model_type}" \
                f" --pad_fusion" \
                f" --out_path {out_path}"
        if args.verbose_performance:
            cmd += " --verbose_performance"
        if args.block_based:
            cmd += " --block_based"
        subprocess.run(cmd.split(' '))
