#!/bin/bash
# filepath: /workspace/deepstream-app-custom/src/deepstream-app/models/convert2trt.sh

# 用法: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
# 例如: ./convert2trt.sh yolov11m_110_rgb_640.onnx yolov11m_110_rgb_640.engine fp16

if [ $# -lt 2 ]; then
  echo "Usage: $0 <ONNX_PATH> <ENGINE_PATH> [fp16]"
  exit 1
fi

ONNX_PATH="$1"
ENGINE_PATH="$2"
FP16_FLAG=""

if [ "$3" == "fp16" ]; then
  FP16_FLAG="--fp16"
fi

/usr/src/tensorrt/bin/trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --verbose \
  $FP16_FLAG