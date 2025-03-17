#!/bin/bash

# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本

# 输入参数
# detection配置
ONNX_PATH="triton_model/Primary_Detect/1/yolov11_ir_drones_p2_single_target_end2end.onnx"  # 输入ONNX文件路径
ENGINE_PATH="triton_model/Primary_Detect/1/yolov11_ir_drones_p2_single_target_end2end.engine"           # 输出engine文件路径
MAX_BATCH=4                                      # 最大batch_size
HEIGHT=640
WIDTH=640
INPUT_NAME="images"

# 分类配置
# ONNX_PATH="triton_model/Secondary_Classify/1/efficientnet_110_with_softmax.onnx"  # 输入ONNX文件路径
# ENGINE_PATH="triton_model/Secondary_Classify/1/efficientnet_110_with_softmax.engine"           # 输出engine文件路径
# MAX_BATCH=8                                      # 最大batch_size
# HEIGHT=224
# WIDTH=224
# INPUT_NAME="input"

# 动态形状配置（min/opt/max batch）
MIN_BATCH=1
OPT_BATCH=1

# 使用trtexec转换
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --minShapes=${INPUT_NAME}:${MIN_BATCH}x3x${HEIGHT}x${WIDTH} \
  --optShapes=${INPUT_NAME}:${OPT_BATCH}x3x${HEIGHT}x${WIDTH} \
  --maxShapes=${INPUT_NAME}:${MAX_BATCH}x3x${HEIGHT}x${WIDTH} \
  --fp16 \
  --verbose