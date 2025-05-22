#!/bin/bash
# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本
# 跟踪配置
ONNX_PATH="tsm_end2end.onnx"  # 输入ONNX文件路径
ENGINE_PATH="tsm_end2end_fp16.engine"           # 输出ENGINE文件路径
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --verbose \
  --fp16