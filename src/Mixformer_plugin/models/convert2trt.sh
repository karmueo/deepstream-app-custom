#!/bin/bash
# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本
# 跟踪配置
ONNX_PATH="ostrack-384-ep300-ce.onnx"  # 输入ONNX文件路径
ENGINE_PATH="ostrack-384-ep300-ce_fp32.engine"           # 输出ENGINE文件路径
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --verbose # --precisionConstraints=obey --fp16