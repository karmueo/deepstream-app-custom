#!/bin/bash

# 推流脚本：将 MP4 文件推送到 RTSP 服务器
# 用法: ./push.sh <输入文件.mp4> [rtsp://服务器地址]
# 如果不提供服务器地址，默认使用 rtsp://127.0.0.1/live/test

DEFAULT_RTSP_URL="rtsp://127.0.0.1/live/test"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "用法: $0 <输入文件.mp4> [rtsp://服务器地址]"
    echo "如果不提供服务器地址，默认使用 $DEFAULT_RTSP_URL"
    exit 1
fi

INPUT_FILE=$1
RTSP_URL=${2:-$DEFAULT_RTSP_URL}

# 使用 FFmpeg 直接复制推流（循环播放）
ffmpeg -re -stream_loop -1 -i "$INPUT_FILE" -an -c copy -f rtsp -rtsp_transport tcp "$RTSP_URL"

echo "推流完成"