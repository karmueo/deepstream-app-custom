import socket
import json
import time
import argparse
import threading
import sys
import ipaddress

# 改为默认向本地(或指定主机)的单播监听端口 5513 发送控制报文。
# DeepStream 端使用 udpsrc 设置 address=0.0.0.0:5513 表示“在所有网卡监听”，
# 发送端需要指定一个具体 IP；默认使用 127.0.0.1，如需跨主机，请用 --host 指定实际服务器 IP。
UNICAST_HOST_DEFAULT = '127.0.0.1'
PORT_DEFAULT = 5513

def _build_sensor_payload():
    """构造一个包含基础传感器信息的字典(不含 command 字段)."""
    return {
        "timestamp": time.time(),
        "message": f"Multicast message at {time.strftime('%H:%M:%S')}",
        "sensor_data": {
            "temperature": 25.0 + 0.1 * (time.time() % 10),
            "humidity": 60 + int(time.time() % 5)
        }
    }

def _is_multicast(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
        return ip.is_multicast
    except ValueError:
        return False

def send_packet(sock, host, port, payload, silent=False):
    try:
        sock.sendto(json.dumps(payload).encode('utf-8'), (host, port))
        if not silent:
            print(f"[发送端] -> {host}:{port} {payload}")
    except Exception as e:
        print(f"[发送端] 发送失败({host}:{port}): {e}")

def telemetry_loop(stop_event, sock, host, port, rate):
    """可选：后台持续发送无 command 的 telemetry 数据，便于对端刷新 timestamp。"""
    interval = 1.0 / rate if rate > 0 else 1.0
    while not stop_event.is_set():
        payload = _build_sensor_payload()
        send_packet(sock, host, port, payload, silent=True)
        stop_event.wait(interval)

def interactive(loop_telemetry=False, telemetry_rate=2, host=UNICAST_HOST_DEFAULT, port=PORT_DEFAULT):
    """
    键盘交互模式：
      s / start  : 开始录像  (可输入 source_id, start_time, duration)
      e / stop   : 停止录像  (需 source_id)
      r / recon  : 发送重连  (需 source_id, new_uri)
      t          : 发送一次无指令的 telemetry 数据
      q / exit   : 退出
    DeepStream C 侧解析支持的 JSON：
      开始: {"command":"start_record", "source_id":N, "start_time":X?, "duration":Y?}
      停止: {"command":"stop_record",  "source_id":N}
      重连: {"command":"reconnect_rtsp","source_id":N, "new_uri":"rtsp://..."}
    start_time / duration 可省略（在 C 中会使用默认/配置值）。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # 仅当目标是多播地址时才设置 TTL
    if _is_multicast(host):
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        except Exception:
            pass

    stop_event = threading.Event()
    telemetry_thread = None
    if loop_telemetry:
        telemetry_thread = threading.Thread(target=telemetry_loop, args=(stop_event, sock, host, port, telemetry_rate), daemon=True)
        telemetry_thread.start()
        print(f"[发送端] Telemetry 后台线程启动，频率 {telemetry_rate}/s -> {host}:{port}")

    def _input(prompt):
        try:
            return input(prompt)
        except EOFError:
            return 'q'

    print("[发送端] 进入交互模式，输入 'h' 查看帮助。")
    try:
        while True:
            cmd = _input("指令(s=start, e=stop, r=reconnect, t=telemetry, h=help, q=quit)> ").strip().lower()
            if cmd in ('q', 'quit', 'exit'):
                break
            if cmd in ('h', 'help', '?'):
                print(interactive.__doc__)
                continue
            if cmd in ('t', 'telemetry'):
                payload = _build_sensor_payload()
                send_packet(sock, host, port, payload)
                continue
            if cmd in ('s', 'start'):
                si = _input("source_id (默认0): ").strip() or '0'
                start_time = _input("start_time(可空): ").strip()
                duration = _input("duration(可空): ").strip()
                payload = _build_sensor_payload()
                payload.update({"command": "start_record", "source_id": int(si)})
                if start_time != '':
                    payload['start_time'] = int(start_time)
                if duration != '':
                    payload['duration'] = int(duration)
                send_packet(sock, host, port, payload)
                continue
            if cmd in ('e', 'stop'):
                si = _input("source_id (默认0): ").strip() or '0'
                payload = _build_sensor_payload()
                payload.update({"command": "stop_record", "source_id": int(si)})
                send_packet(sock, host, port, payload)
                continue
            if cmd in ('r', 'recon', 'reconnect'):
                si = _input("source_id (默认0): ").strip() or '0'
                new_uri = _input("new_uri (必填, 如 rtsp://...): ").strip()
                if not new_uri:
                    print("[发送端] new_uri 不能为空")
                    continue
                payload = _build_sensor_payload()
                payload.update({"command": "reconnect_rtsp", "source_id": int(si), "new_uri": new_uri})
                send_packet(sock, host, port, payload)
                continue
            print("[发送端] 未知指令，输入 h 查看帮助。")
    except KeyboardInterrupt:
        print("\n[发送端] Ctrl+C 退出")
    finally:
        stop_event.set()
        if telemetry_thread:
            telemetry_thread.join(timeout=1)
        sock.close()
        print("[发送端] 已退出")


def legacy_send_once(loop=False, host=UNICAST_HOST_DEFAULT, port=PORT_DEFAULT):
    """旧模式：按固定频率发送一次或循环发送 reconnect_rtsp JSON（现在默认单播）。"""
    SEND_RATE = 30
    INTERVAL = 1.0 / SEND_RATE
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    if _is_multicast(host):
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        except Exception:
            pass
    try:
        while True:
            start_time = time.time()
            payload = _build_sensor_payload()
            payload.update({
                "command": "reconnect_rtsp",
                "source_id": 0,
                "new_uri": "rtsp://192.168.1.110/live/rgb"
            })
            send_packet(sock, host, port, payload)
            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)
            if not loop:
                break
    except KeyboardInterrupt:
        print("\n[发送端] 用户终止发送")
    finally:
        sock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepStream 控制指令发送 (单播/可选多播 JSON)')
    parser.add_argument('--host', default=UNICAST_HOST_DEFAULT, help='目标主机 IP；DeepStream 端监听 0.0.0.0:5513 时这里填其实际 IP 或 127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=PORT_DEFAULT, help='端口 (默认 5513)')
    parser.add_argument('--legacy', action='store_true', help='使用旧模式(自动发送 reconnect_rtsp)')
    parser.add_argument('--loop', action='store_true', help='旧模式下循环发送')
    parser.add_argument('--telemetry', action='store_true', help='交互模式下开启后台持续发送 telemetry')
    parser.add_argument('--telemetry-rate', type=int, default=2, help='telemetry 每秒发送次数')
    args = parser.parse_args()

    if args.legacy:
        print("[发送端] 进入旧模式 (reconnect_rtsp)")
        legacy_send_once(loop=args.loop, host=args.host, port=args.port)
    else:
        interactive(loop_telemetry=args.telemetry, telemetry_rate=args.telemetry_rate, host=args.host, port=args.port)