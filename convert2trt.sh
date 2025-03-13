#!/bin/bash

# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本

# 输入参数
ONNX_PATH="triton_model/Secondary_Classify/1/efficientnet_110_with_softmax.onnx"  # 输入ONNX文件路径
ENGINE_PATH="efficientnet_fp16.engine"           # 输出engine文件路径
PRECISION="fp32"                                 # 精度 (fp32/fp16/int8)
MAX_BATCH=8                                      # 最大batch_size

# 动态形状配置（min/opt/max batch）
MIN_BATCH=1
OPT_BATCH=4

# 使用trtexec转换
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --minShapes=input:${MIN_BATCH}x3x224x224 \
  --optShapes=input:${OPT_BATCH}x3x224x224 \
  --maxShapes=input:${MAX_BATCH}x3x224x224 \
  --inputIOFormats=fp32:chw \
  --outputIOFormats=fp32:chw \
  --verbose