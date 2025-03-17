#/bin/bash
# Start Triton Inference Server with the converted models
/opt/tritonserver/bin/tritonserver \
    --model-repository=./triton_model \
    --disable-auto-complete-config \
    --log-verbose=0
