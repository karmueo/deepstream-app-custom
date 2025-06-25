#!/bin/bash

# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本

# 输入参数
# detection配置
ONNX_PATH="yolo_classify_110_IR.onnx"  # 输入ONNX文件路径
# ONNX_PATH="yolov11_ir_drones_p2_2classes.onnx"  # 输入ONNX文件路径
ENGINE_PATH="yolo_classify_110_IR_fp32.engine"           # 输出engine文件路径
# ENGINE_PATH="yolov11_ir_drones_p2_2classes_fp32.engine"           # 输出engine文件路径


# 使用trtexec转换
# /usr/src/tensorrt/bin/trtexec \
#   --onnx=$ONNX_PATH \
#   --saveEngine=$ENGINE_PATH \
#   --minShapes=${INPUT_NAME}:${MIN_BATCH}x3x${HEIGHT}x${WIDTH} \
#   --optShapes=${INPUT_NAME}:${OPT_BATCH}x3x${HEIGHT}x${WIDTH} \
#   --maxShapes=${INPUT_NAME}:${MAX_BATCH}x3x${HEIGHT}x${WIDTH} \
#   --verbose \
#   --fp16

/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --verbose
  # --fp16