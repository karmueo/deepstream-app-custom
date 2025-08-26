#!/bin/bash

# DeepStream 应用管理器脚本
# 用法: ./deepstream_manager.sh {start|stop|status|restart}

# 定义应用配置
declare -A APPS=(
    ["ir"]="src/deepstream-app/configs/ir_app_config.txt"
    ["rgb"]="src/deepstream-app/configs/rgb_app_config.txt"
)

# 定义日志文件
IR_LOG="./app_ir.log"
RGB_LOG="./app_rgb.log"

# PID 文件目录
PID_DIR="/tmp/deepstream_pids"
IR_PID="$PID_DIR/ir_app.pid"
RGB_PID="$PID_DIR/rgb_app.pid"

# 创建 PID 目录
mkdir -p "$PID_DIR"

# 设置环境变量
setup_environment() {
    export GST_DEBUG_NO_COLOR=1
    export GST_DEBUG=4
}

# 启动单个应用
start_app() {
    local app_name=$1
    local config_file=$2
    local log_file=$3
    local pid_file=$4
    
    # 检查是否已经在运行
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$app_name 应用已经在运行 (PID: $pid)"
            return 0
        fi
    fi
    
    # 设置特定的调试文件
    export GST_DEBUG_FILE="$log_file"
    
    # 启动应用
    echo "启动 $app_name 应用..."
    nohup src/deepstream-app/deepstream-app -c "$config_file" > "${log_file}.out" 2>&1 &
    local pid=$!
    
    # 保存 PID
    echo $pid > "$pid_file"
    echo "$app_name 应用启动成功 (PID: $pid, 日志: $log_file)"
}

# 停止单个应用
stop_app() {
    local app_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "停止 $app_name 应用 (PID: $pid)..."
            kill "$pid"
            rm -f "$pid_file"
            echo "$app_name 应用已停止"
        else
            echo "$app_name 应用未运行 (PID: $pid 不存在)"
            rm -f "$pid_file"
        fi
    else
        echo "$app_name 应用未运行 (无 PID 文件)"
    fi
}

# 启动所有应用
start_all() {
    echo "正在启动所有 DeepStream 应用..."
    setup_environment
    
    start_app "IR" "${APPS[ir]}" "$IR_LOG" "$IR_PID"
    start_app "RGB" "${APPS[rgb]}" "$RGB_LOG" "$RGB_PID"
    
    echo "所有应用启动完成!"
    echo "查看日志: tail -f $IR_LOG 或 tail -f $RGB_LOG"
}

# 停止所有应用
stop_all() {
    echo "正在停止所有 DeepStream 应用..."
    
    stop_app "IR" "$IR_PID"
    stop_app "RGB" "$RGB_PID"
    
    echo "所有应用停止完成!"
}

# 查看应用状态
check_status() {
    echo "DeepStream 应用状态:"
    echo "===================="
    
    check_app_status "IR" "$IR_PID"
    check_app_status "RGB" "$RGB_PID"
}

# 检查单个应用状态
check_app_status() {
    local app_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "✓ $app_name 应用正在运行 (PID: $pid)"
        else
            echo "✗ $app_name 应用 PID 文件存在但进程不存在 (PID: $pid)"
        fi
    else
        echo "✗ $app_name 应用未运行"
    fi
}

# 重启所有应用
restart_all() {
    stop_all
    sleep 2
    start_all
}

# 显示使用说明
show_usage() {
    echo "使用方法: $0 {start|stop|status|restart}"
    echo "  start   - 启动所有 DeepStream 应用"
    echo "  stop    - 停止所有 DeepStream 应用"
    echo "  status  - 查看应用状态"
    echo "  restart - 重启所有应用"
}

# 主程序
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    status)
        check_status
        ;;
    restart)
        restart_all
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

exit 0