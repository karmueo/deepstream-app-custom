#!/bin/bash
# trtexec_convert.sh - 转换ONNX到TensorRT engine的脚本
# 跟踪配置
ONNX_PATH="uniformerv2_expand_1_softmax.onnx"  # 输入ONNX文件路径
ENGINE_PATH="uniformerv2_e1_end2end_fp32.engine"           # 输出ENGINE文件路径
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --saveEngine=$ENGINE_PATH \
  --verbose
  # --fp16