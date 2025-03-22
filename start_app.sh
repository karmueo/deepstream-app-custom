#!/bin/bash
# 设定环境变量export GST_DEBUG_NO_COLOR=1
export GST_DEBUG_NO_COLOR=1
export GST_DEBUG_FILE=./app.log
export GST_DEBUG=4
src/deepstream-app/deepstream-app -c src/deepstream-app/configs/deepstream_app_config.txt