#/bin/bash
# Start Triton Inference Server with the converted models
/opt/tritonserver/bin/tritonserver \
    --model-repository=/workspaces/Deepstream_template/deepstream/triton_model \
    --disable-auto-complete-config \
    --log-verbose=0
