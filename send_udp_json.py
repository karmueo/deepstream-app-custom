import socket
import json
import time
import argparse

def send_json_via_multicast(loop=False):
    # 组播配置
    MULTICAST_GROUP = '239.255.255.250'
    PORT = 5000
    SEND_RATE = 30  # 每秒发送次数
    INTERVAL = 1.0 / SEND_RATE  # 每次发送间隔（秒）

    # 创建UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    try:
        while True:
            start_time = time.time()  # 记录循环开始时间
            
            # 构造动态变化的JSON数据（示例：模拟传感器数据）
            data = {
                "timestamp": time.time(),
                "message": f"Multicast message at {time.strftime('%H:%M:%S')}",
                "sensor_data": {
                    "temperature": 25.0 + 0.1 * (time.time() % 10),  # 模拟温度波动
                    "humidity": 60 + int(time.time() % 5)             # 模拟湿度波动
                },
                "command": "reconnect_rtsp",
                "source_id": 0,
                "new_uri": "rtsp://192.168.1.110/live/rgb"
            }
            json_data = json.dumps(data).encode('utf-8')

            # 发送数据
            sock.sendto(json_data, (MULTICAST_GROUP, PORT))
            print(f"[发送端] 已发送: {data}", end='\r')  # \r覆盖上一行输出，避免刷屏

            # 精确控制发送频率
            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)
            if not loop:
                break

    except KeyboardInterrupt:
        print("\n[发送端] 用户终止发送")
    except Exception as e:
        print(f"\n[发送端] 错误: {e}")
    finally:
        sock.close()

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='发送JSON报文到指定的组播组和端口',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
    )
    # parser.add_argument('--group', dest='multicast_group', type=str, 
    #                    default='239.255.255.250',
    #                    help='组播组地址（D类IP地址，如239.255.255.250）')
    # parser.add_argument('--port', type=int, 
    #                    default=5000,
    #                    help='组播端口号（1024-49151）')
    parser.add_argument('--loop', action='store_true',
                       help='循环发送（默认开启）')
    # parser.add_argument('--rate', type=int, 
    #                    default=30,
    #                    help='每秒发送次数（默认30次）')
    args = parser.parse_args()
    print(f"[发送端] 启动")
    send_json_via_multicast(loop=args.loop)