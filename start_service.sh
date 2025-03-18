#!/bin/bash
# Start Triton Inference Server with the converted models
/opt/tritonserver/bin/tritonserver \
    --model-repository=./triton_model \
    --disable-auto-complete-config \
    --log-verbose=0 \
    > triton.log 2>&1 &

sleep 2

/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /workspace/deepstream-app-custom/src/deepstream-app/configs/deepstream_app_config.txt