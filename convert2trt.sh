#!/bin/bash

# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本

# 输入参数
ONNX_PATH="triton_model/Primary_Detect/1/yolov11_ir_drones_p2_single_target_end2end.onnx"  # 输入ONNX文件路径
ENGINE_PATH="triton_model/Primary_Detect/1/yolov11_ir_drones_p2_single_target_end2end.engine"           # 输出engine文件路径
PRECISION="fp32"                                 # 精度 (fp32/fp16/int8)
MAX_BATCH=4                                      # 最大batch_size

# 动态形状配置（min/opt/max batch）
MIN_BATCH=1
OPT_BATCH=1

# 使用trtexec转换
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --minShapes=images:${MIN_BATCH}x3x640x640 \
  --optShapes=images:${OPT_BATCH}x3x640x640 \
  --maxShapes=images:${MAX_BATCH}x3x640x640 \
  --verbose