# OEMC: Compiling Fused DNN Operators on Heterogeneous NPU Engine Machine - Compiler

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

- Corresponding directories and files:
    - OEMC/benchmark

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
    - OEMC/utils/schema.fbs

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
# Copy OEM's DDR4, DDR5, HBM2 config yaml to ramulator2 dir
cd ..
cp ../../OEM_ramulator2/*.yaml ./
# Copy OEM's ramulator2 entry to resource dir
cp -r ../../OEM_ramulator2/OEM_wrappers ./resources
cd resources/OEM_wrappers
# Note user needs to type their 
g++ -c main.cc ramulator.cc -I /path/of/ext/yaml-cpp/include -I /path/of/ext/spdlog/include -I /path/of/src
g++ main.o ramulator2.o -L path/of/libramulator.so -lramulator -o ramulator2
```

- Corresponding directories and files:
    - OEMC/OEM_ramulator2

# Installation
```sh
git clone https://github.com/chuckwu2000/OEMC.git
cd OEMC
```

# Run
```sh
python run.sh
```
