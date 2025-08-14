# ONEC: DNN Operator Fusion on Heterogeneous NPU Specialized Engines - Compiler

## Table of Contents

- [Third-party code](#third-party-code)
- [Installation](#installation)

# Third-party code
Some code is used for evaluation procedures and input format conversion.

## Benchmark Suite
| DNN Models               | Model Zoo    |
|--------------------------|--------------|
| DistilGPT2 [GPT2]        | HuggingFace  |
| MobileBERT [MBT]         | AI-Benchmark |
| MobileVit [MVT]          | Qualcomm     |
| WhisperTiny [WHT]        | Qualcomm     |
| EfficientNet-B0 [EFF]    | Qualcomm     |
| ResNet50 [RES]           | Qualcomm     |
| YoloX [YOX]              | Qualcomm     |
| GCN [GCN]                | [GCN](https://github.com/ZhimingZo/Modulated-GCN)          |

- Put [[Benchmark Download Link]](https://drive.google.com/drive/folders/1tT0Vy3KFhyeqRaR6JZ7GHrf6anLUZGJ3?usp=sharing) in new create dir benchmark/tflite

## TFlite flatbuffers
- This project will use `flatc` to convert .tflite to .json to modify the tflite model.
```sh
$ git clone https://github.com/google/flatbuffers.git
$ cd flatbuffers
$ cmake -G "Unix Makefiles"
$ make
$ ./flattests # this is quick, and should print "ALL TESTS PASSED"
$ sudo make install # install
$ sudo ldconfig # Configuring a dynamic link library
$ flatc --version # Check if FlatBuffers is installed successfully
```
- Corresponding directories and files:
    - ONEC/utils/schema.fbs

## Ramulator2
- This project will use ramulator2 to simulate DRAM access latency & energy
```sh
mkdir extern
cd extern
git clone https://github.com/CMU-SAFARI/ramulator2
cd ramulator2
mkdir build
cd build
cmake ..
make -j # Will generate libramulator.so
# Copy ONEC's DDR4, DDR5, HBM2 config yaml to ramulator2 dir
cd ..
cp ../../ONEC_ramulator2/*.yaml ./
# Copy ONEC's ramulator2 entry to resource dir
cp -r ../../ONEC_ramulator2/ONEC_wrappers ./resources
cd resources/ONEC_wrappers
# Note user needs to type their 
g++ -c main.cc ramulator2.cc -I /path/of/ext/yaml-cpp/include -I /path/of/ext/spdlog/include -I /path/of/src
g++ main.o ramulator2.o -L path/of/libramulator.so/dir -lramulator -o ramulator2
```

- Corresponding directories and files:
    - ONEC/ONEC_ramulator2

# Installation
```sh
git clone https://github.com/chuckwu2000/ONEC.git
cd ONEC
```

# Run
```sh
python run.sh
```
