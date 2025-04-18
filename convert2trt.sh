#!/bin/bash

# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本

# 输入参数
# detection配置
# ONNX_PATH="src/deepstream-app/models/yolov11_ir_drones_p2_2classes.onnx"  # 输入ONNX文件路径
# ENGINE_PATH="src/deepstream-app/models/yolov11_ir_drones_p2_2classes.engine"           # 输出engine文件路径
# MAX_BATCH=1                                      # 最大batch_size
# HEIGHT=640
# WIDTH=640
# INPUT_NAME="intput"

# 分类配置
# ONNX_PATH="src/deepstream-app/models/efficientnet_110_with_softmax.onnx"  # 输入ONNX文件路径
# ENGINE_PATH="src/deepstream-app/models/efficientnet_110_with_softmax.engine"           # 输出engine文件路径
# MAX_BATCH=8                                      # 最大batch_size
# HEIGHT=224
# WIDTH=224
# INPUT_NAME="input"

# 跟踪配置
# ONNX_PATH="triton_model/Tracking/1/mixformer_v2.onnx"  # 输入ONNX文件路径
# ENGINE_PATH="triton_model/Tracking/1/mixformer_v2.engine"           # 输出engine文件路径
# MAX_BATCH=1                                      # 最大batch_size
# HEIGHT=640
# WIDTH=640
# INPUT_NAME="images"

# 视频分类
ONNX_PATH="triton_model/Video_Classify/1/end2end.onnx"  # 输入ONNX文件路径
ENGINE_PATH="triton_model/Video_Classify/1/end2end.engine"           # 输出engine文件路径

# 动态形状配置（min/opt/max batch）
#MIN_BATCH=1
#OPT_BATCH=1

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