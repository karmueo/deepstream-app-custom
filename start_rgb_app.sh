#!/bin/bash
# 设定环境变量export GST_DEBUG_NO_COLOR=1
export GST_DEBUG_NO_COLOR=1
export GST_DEBUG_FILE=./app_rgb.log
export GST_DEBUG=4
# nohup src/deepstream-app/deepstream-app -c src/deepstream-app/configs/deepstream_app_config.txt &
src/deepstream-app/deepstream-app -c src/deepstream-app/configs/rgb_app_config.txt