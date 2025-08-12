#!/bin/bash

python prepare_ts_models.py --models_dir ./benchmark/tflite/transformer/Mobilebert_quant_int8_simplify.tflite --verbose_performance --token_size 384 --split_size 128
echo "Mobilebert finish!!"
python prepare_ts_models.py --models_dir ./benchmark/tflite/CNN/EfficientNet-B0_quant_int8_pad_fusion.tflite --verbose_performance --split_size 14 --model_type CNN
echo "EfficientNet-B0 finish!!"
python prepare_ts_models.py --models_dir ./benchmark/tflite/GNN/GCN_batch_4_quant_int8.tflite --verbose_performance --token_size 17 --split_size 17
echo "GCN finish!!"
