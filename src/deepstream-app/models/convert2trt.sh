#!/bin/bash
# filepath: /workspace/deepstream-app-custom/src/deepstream-app/models/convert2trt.sh

# 用法: 
#  静态 batch: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
#  动态 batch: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> dynamic [min_batch] [opt_batch] [max_batch] [fp16]
# 示例: 
#  静态: ./convert2trt.sh yolov11m_110_rgb_640.onnx yolov11m_110_rgb_640.engine fp16
#  动态: ./convert2trt.sh model.onnx model.engine dynamic 1 16 32 fp16

if [ $# -lt 2 ]; then
  echo "Usage:"
  echo "  Static batch: $0 <ONNX_PATH> <ENGINE_PATH> [fp16]"
  echo "  Dynamic batch: $0 <ONNX_PATH> <ENGINE_PATH> dynamic [min_batch] [opt_batch] [max_batch] [fp16]"
  exit 1
fi

ONNX_PATH="$1"
ENGINE_PATH="$2"
FP16_FLAG=""
DYNAMIC_FLAGS=""

# 检查是否是动态 batch 模式
if [ "$3" == "dynamic" ]; then
  # 设置默认的 batch 大小
  MIN_BATCH=${4:-1}
  OPT_BATCH=${5:-16}
  MAX_BATCH=${6:-32}
  
  # 验证 batch 大小值
  if ! [[ "$MIN_BATCH" =~ ^[0-9]+$ ]] || ! [[ "$OPT_BATCH" =~ ^[0-9]+$ ]] || ! [[ "$MAX_BATCH" =~ ^[0-9]+$ ]]; then
    echo "Error: Batch sizes must be integers (min=$MIN_BATCH, opt=$OPT_BATCH, max=$MAX_BATCH)"
    exit 1
  fi
  
  if [ $MIN_BATCH -gt $OPT_BATCH ] || [ $OPT_BATCH -gt $MAX_BATCH ]; then
    echo "Error: Batch sizes must satisfy min_batch <= opt_batch <= max_batch"
    exit 1
  fi
  
  # 设置动态 batch 参数
  DYNAMIC_FLAGS="--minShapes=input:${MIN_BATCH}x3x640x640 --optShapes=input:${OPT_BATCH}x3x640x640 --maxShapes=input:${MAX_BATCH}x3x640x640"
  
  # 检查是否包含 fp16 标志
  if [ "${7}" == "fp16" ]; then
    FP16_FLAG="--fp16"
  fi
else
  # 静态 batch 模式
  if [ "$3" == "fp16" ]; then
    FP16_FLAG="--fp16"
  fi
fi

# 打印转换参数
echo "Converting ONNX model to TensorRT engine:"
echo "  ONNX Path:    $ONNX_PATH"
echo "  Engine Path:  $ENGINE_PATH"
echo "  Dynamic Batch: $([ -n "$DYNAMIC_FLAGS" ] && echo "Enabled (min=$MIN_BATCH, opt=$OPT_BATCH, max=$MAX_BATCH)" || echo "Disabled")"
echo "  FP16:          $([ -n "$FP16_FLAG" ] && echo "Enabled" || echo "Disabled")"

# 执行转换
/usr/src/tensorrt/bin/trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --verbose \
  $DYNAMIC_FLAGS \
  $FP16_FLAG

# 检查转换结果
if [ $? -eq 0 ]; then
  echo ""
  echo "Conversion successful! TensorRT engine saved to: $ENGINE_PATH"
else
  echo ""
  echo "Conversion failed. Check error messages above."
  exit 1
fi