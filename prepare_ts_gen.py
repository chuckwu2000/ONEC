import glob
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--codegen_path", nargs='?', type=str,
                    default="CodeGen.py")
parser.add_argument("--schema_path", nargs='?', type=str,
                    default="utils/schema.fbs")
parser.add_argument("--models_dir", nargs='?', type=str,
                    default="models/")
args = parser.parse_args()
codegen_path = str(args.codegen_path)
schema_path = str(args.schema_path)
models_dir = str(args.models_dir)

# tflite_list = [path for path in glob.glob(f'{models_dir}/*.tflite', recursive=True)]
# for tflite_path in sorted(tflite_list):
tflite_path = models_dir
model_name = os.path.splitext(os.path.basename(tflite_path))[0]
out_path = f"{os.path.dirname(tflite_path)}/{model_name}_splitted_DP_1.txt"
cmd =   f"python {codegen_path} {tflite_path} --schema_path {schema_path}" \
        f" --out_path {out_path}"
subprocess.run(cmd.split(' '))