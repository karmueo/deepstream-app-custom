#!/bin/bash

# DeepStream 应用管理器脚本 (带自动重启与日志轮转)
# 用法: ./start_both_apps.sh {start|start-block|stop|status|restart}

########################################
# 基础配置
########################################
declare -A APPS=(
    [ir]="src/deepstream-app/configs/ir_app_config.txt"
    [rgb]="src/deepstream-app/configs/rgb_app_config.txt"
)

BLOCKING_MODE=0                # start-block 时置 1
PIDS=()                        # 已启动子进程 PID 列表

LOG_DIR="./logs"
LOG_RETENTION_DAYS=7
CURRENT_DATE="$(date +%F)"
IR_LOG=""                      # 运行时赋值
RGB_LOG=""                      # 运行时赋值

PID_DIR="/tmp/deepstream_pids"
IR_PID="$PID_DIR/ir_app.pid"
RGB_PID="$PID_DIR/rgb_app.pid"
mkdir -p "$PID_DIR"

CONTROL_DIR="/tmp/deepstream_control"
DISABLE_RESTART_FILE="$CONTROL_DIR/auto_restart.disabled"
mkdir -p "$CONTROL_DIR"

########################################
# 日志与监控相关函数
########################################
rotate_logs() {
    mkdir -p "$LOG_DIR"
    local today="$(date +%F)"
    if [ "$CURRENT_DATE" != "$today" ]; then
        CURRENT_DATE="$today"
    fi
    IR_LOG="$LOG_DIR/ir_${CURRENT_DATE}.log"
    RGB_LOG="$LOG_DIR/rgb_${CURRENT_DATE}.log"
    ln -sf "ir_${CURRENT_DATE}.log" "$LOG_DIR/ir_current.log"
    ln -sf "rgb_${CURRENT_DATE}.log" "$LOG_DIR/rgb_current.log"
    find "$LOG_DIR" -type f -name 'ir_*.log' -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    find "$LOG_DIR" -type f -name 'rgb_*.log' -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
}

monitor_loop() {
    declare -A RESTART_COUNT
    declare -A LAST_RESTART_TS
    local interval=5
    while true; do
        sleep "$interval"
        local now_date="$(date +%F)"
        if [ "$now_date" != "$CURRENT_DATE" ]; then
            rotate_logs
        fi
        for key in ir rgb; do
            local upper
            [ "$key" = ir ] && upper=IR || upper=RGB
            local pid_file log_file config
            [ "$key" = ir ] && pid_file="$IR_PID" log_file="$IR_LOG" || pid_file="$RGB_PID" log_file="$RGB_LOG"
            config="${APPS[$key]}"
            local need_restart=0 cur_pid=""
            if [ -f "$pid_file" ]; then
                cur_pid="$(cat "$pid_file")"
                kill -0 "$cur_pid" 2>/dev/null || need_restart=1
            else
                need_restart=1
            fi
            if [ "$need_restart" = 1 ]; then
                if [ -f "$DISABLE_RESTART_FILE" ]; then
                    echo "[监控] $upper 已停止，自动重启被禁用" >&2
                    continue
                fi
                local now_ts=$(date +%s)
                local count=${RESTART_COUNT[$upper]:-0}
                local last=${LAST_RESTART_TS[$upper]:-0}
                if [ $last -gt 0 ] && [ $((now_ts - last)) -le 60 ] && [ $count -ge 5 ]; then
                    echo "[监控] $upper 频繁崩溃 (60s $count 次)，暂停 60s" >&2
                    sleep 60
                    RESTART_COUNT[$upper]=0
                    LAST_RESTART_TS[$upper]=$now_ts
                    continue
                fi
                echo "[监控] 重启 $upper ..." >&2
                local prev_block=$BLOCKING_MODE
                BLOCKING_MODE=0
                start_app "$key" "$config" "$log_file" "$pid_file"
                BLOCKING_MODE=$prev_block
                RESTART_COUNT[$upper]=$((count+1))
                LAST_RESTART_TS[$upper]=$now_ts
            fi
        done
    done
}

########################################
# 基础函数
########################################
setup_environment() {
    export GST_DEBUG_NO_COLOR=1
    export GST_DEBUG=4
}

start_app() {
    local key=$1 config_file=$2 log_file=$3 pid_file=$4
    local upper
    [ "$key" = ir ] && upper=IR || upper=RGB
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "$upper 已在运行 (PID:$old_pid)"; return 0
        fi
    fi
    export GST_DEBUG_FILE="$log_file"
    echo "启动 $upper (blocking=$BLOCKING_MODE)..."
    if [ "$BLOCKING_MODE" = 1 ]; then
        src/deepstream-app/deepstream-app -c "$config_file" >> "$log_file" 2>&1 &
    else
        nohup src/deepstream-app/deepstream-app -c "$config_file" >> "$log_file" 2>&1 &
    fi
    local pid=$!
    echo $pid > "$pid_file"
    echo "$upper 启动成功 PID:$pid 日志:$log_file"
    [ "$BLOCKING_MODE" = 1 ] && PIDS+=("$pid")
}

stop_app() {
    local key=$1 pid_file=$2 upper
    [ "$key" = ir ] && upper=IR || upper=RGB
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "停止 $upper PID:$pid"
            kill "$pid" 2>/dev/null || true
        else
            echo "$upper PID 文件存在但进程不存在"
        fi
        rm -f "$pid_file"
    else
        echo "$upper 未运行"
    fi
}

check_app_status() {
    local key=$1 pid_file=$2 upper
    [ "$key" = ir ] && upper=IR || upper=RGB
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "✓ $upper 运行中 PID:$pid"
        else
            echo "✗ $upper PID 文件存在但进程不在"
        fi
    else
        echo "✗ $upper 未运行"
    fi
}

start_all() {
    echo "正在启动所有 DeepStream 应用..."
    rotate_logs
    setup_environment
    start_app ir  "${APPS[ir]}"  "$IR_LOG"  "$IR_PID"
    start_app rgb "${APPS[rgb]}" "$RGB_LOG" "$RGB_PID"
    echo "所有应用启动完成!"
    echo "查看日志: tail -f logs/ir_current.log 或 logs/rgb_current.log"
    if [ "$BLOCKING_MODE" = 1 ]; then
        echo "进入监控循环 (Ctrl+C 退出并停止)"
        monitor_loop
    fi
}

stop_all() {
    echo "正在停止所有应用..."
    touch "$DISABLE_RESTART_FILE"
    stop_app ir  "$IR_PID"
    stop_app rgb "$RGB_PID"
    echo "所有应用停止完成"
}

restart_all() {
    stop_all
    rm -f "$DISABLE_RESTART_FILE"
    sleep 2
    start_all
}

check_status() {
    echo "DeepStream 应用状态:"; echo "===================="
    check_app_status ir  "$IR_PID"
    check_app_status rgb "$RGB_PID"
}

show_usage() {
    echo "用法: $0 {start|start-block|stop|status|restart}"
    echo "  start        启动并退出(不阻塞，不监控)"
    echo "  start-block  启动+阻塞+监控+自动重启+日志轮转"
    echo "  stop         停止所有应用并禁用自动重启"
    echo "  status       显示运行状态"
    echo "  restart      停止后再启动 (保留监控)"
}

cleanup_and_exit() {
    echo "收到终止信号，停止应用..."
    stop_all
    exit 0
}
trap cleanup_and_exit SIGINT SIGTERM

########################################
# 主入口
########################################
case "$1" in
    start)
        rm -f "$DISABLE_RESTART_FILE"
        start_all
        ;;
    start-block)
        BLOCKING_MODE=1
        rm -f "$DISABLE_RESTART_FILE"
        start_all
        ;;
    stop)
        stop_all
        ;;
    status)
        check_status
        ;;
    restart)
        BLOCKING_MODE=${BLOCKING_MODE:-0}
        restart_all
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

exit 0