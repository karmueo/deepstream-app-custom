# C-UAV 可见光控制测试工具实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建交互式 Python 工具，通过键盘方向键发送可见光焦距控制命令到光电设备

**Architecture:** 单文件 Python 脚本，使用两个 UDP socket 分别发送命令和接收反馈，终端原始模式捕获方向键

**Tech Stack:** Python 3 标准库（socket, json, termios, sys, time）

---

## 文件结构

```
src/gst-cuavcontrolsink/scripts/
└── cuav_visible_control.py    # 单一实现文件
```

---

## 实现步骤

### Task 1: 基础框架与参数解析

**Files:**
- Create: `src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

- [ ] **Step 1: 创建文件并实现基础框架**

```python
#!/usr/bin/env python3
"""
C-UAV 可见光控制测试工具
通过键盘方向键控制可见光焦距
"""

import argparse
import json
import socket
import struct
import sys
import time
from typing import Tuple

# 默认组播地址
DEFAULT_SEND_GROUP = "230.1.88.51"
DEFAULT_SEND_PORT = 8003
DEFAULT_RECV_GROUP = "230.1.88.51"
DEFAULT_RECV_PORT = 8013

MSG_ID_VISIBLE_CONTROL = 0x7205  # 28989


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-UAV 可见光控制测试工具")
    parser.add_argument("--send-ip", default=DEFAULT_SEND_GROUP, help="发送目标组播地址")
    parser.add_argument("--send-port", type=int, default=DEFAULT_SEND_PORT, help="发送目标端口")
    parser.add_argument("--recv-ip", default=DEFAULT_RECV_GROUP, help="接收组播地址")
    parser.add_argument("--recv-port", type=int, default=DEFAULT_RECV_PORT, help="接收端口")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"发送 -> {args.send_ip}:{args.send_port}")
    print(f"接收 <- {args.recv_ip}:{args.recv_port}")
    print("按 ↑/↓ 调整焦距，按 Q 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 测试基础框架运行**

Run: `cd /home/nvidia/work/deepstream-app-custom && python3 src/gst-cuavcontrolsink/scripts/cuav_visible_control.py --help`
Expected: 显示帮助信息

- [ ] **Step 3: 提交基础框架**

```bash
git add src/gst-cuavcontrolsink/scripts/cuav_visible_control.py
git commit -m "feat: add cuav_visible_control.py skeleton"
```

---

### Task 2: Socket 初始化与组播配置

**Files:**
- Modify: `src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

- [ ] **Step 1: 添加 Socket 初始化函数**

在 `parse_args()` 后添加：

```python
def create_send_socket(group: str, port: int) -> socket.socket:
    """创建发送socket，加入组播"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((group, port))
    except OSError:
        sock.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.1)
    return sock


def create_recv_socket(group: str, port: int) -> socket.socket:
    """创建接收socket，加入组播"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((group, port))
    except OSError:
        sock.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton("0.0.0.0"))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.1)
    return sock
```

- [ ] **Step 2: 修改 main() 初始化 socket**

替换 main() 中的打印语句后添加：

```python
    send_sock = create_send_socket(args.send_ip, args.send_port)
    recv_sock = create_recv_socket(args.recv_ip, args.recv_port)
    print("Socket 初始化完成")
```

- [ ] **Step 3: 添加 socket 关闭代码**

在 `finally` 块中添加（稍后完善）：

```python
    finally:
        send_sock.close()
        recv_sock.close()
```

- [ ] **Step 4: 测试 socket 初始化**

Run: `python3 src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`
Expected: 打印发送/接收地址信息，Socket初始化完成，无报错

- [ ] **Step 5: 提交**

```bash
git add src/gst-cuavcontrolsink/scripts/cuav_visible_control.py
git commit -m "feat: add socket initialization with multicast support"
```

---

### Task 3: 报文构建与发送逻辑

**Files:**
- Modify: `src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

- [ ] **Step 1: 添加报文构建函数**

在 socket 函数后添加：

```python
msg_sn = 0


def get_current_time() -> dict:
    """获取当前时间戳字典"""
    now = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return {
        "yr": now.tm_year,
        "mo": now.tm_mon,
        "dy": now.tm_mday,
        "h": now.tm_hour,
        "min": now.tm_min,
        "sec": now.tm_sec,
        "msec": ms,
    }


def build_visible_control_msg(pt_focal_en: int) -> dict:
    """构建可见光控制报文"""
    global msg_sn
    msg_sn += 1
    time_fields = get_current_time()
    return {
        "msg_id": MSG_ID_VISIBLE_CONTROL,
        "msg_sn": msg_sn,
        "msg_type": 0,
        "tx_sys_id": 5,
        "tx_dev_type": 5,
        "tx_dev_id": 1,
        "tx_subdev_id": 999,
        "rx_sys_id": 999,
        "rx_dev_type": 999,
        "rx_dev_id": 999,
        "rx_subdev_id": 999,
        **time_fields,
        "cont_type": 0,
        "cont_sum": 1,
        "pt_dev_en": 1,
        "pt_ctrl_en": 1,
        "pt_fov_en": 0,
        "pt_fov_h": 0,
        "pt_fov_v": 0,
        "pt_focal_en": pt_focal_en,
        "pt_focal": 0,
        "pt_focus_en": 0,
        "pt_focus": 0,
        "pt_speed_en": 0,
        "pt_focus_speed": 0,
        "pt_bri_en": 0,
        "pt_bri_ctrs": 0,
        "pt_ctrs_en": 0,
        "pt_ctrs": 0,
        "pt_ofr_en": 0,
        "pt_ofr": 0,
        "pt_focus_mode": 0,
        "pt_zoom": 0,
    }


def send_command(sock: socket.socket, target: Tuple[str, int], pt_focal_en: int) -> None:
    """发送控制命令"""
    msg = build_visible_control_msg(pt_focal_en)
    data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    sock.sendto(data, target)
    print(f"[发送] {json.dumps(msg, ensure_ascii=False)}")
```

- [ ] **Step 2: 修改 main() 调用发送**

在 socket 初始化后替换为：

```python
    send_addr = (args.send_ip, args.send_port)

    try:
        while True:
            # 尝试接收反馈
            try:
                data, addr = recv_sock.recvfrom(65535)
                recv_msg = json.loads(data.decode("utf-8"))
                print(f"[接收] {addr} <- {json.dumps(recv_msg, ensure_ascii=False)}")
            except socket.timeout:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        send_sock.close()
        recv_sock.close()
```

- [ ] **Step 3: 测试运行**

Run: `python3 src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`
Expected: 正常运行，无报错

- [ ] **Step 4: 提交**

```bash
git add src/gst-cuavcontrolsink/scripts/cuav_visible_control.py
git commit -m "feat: add message building and send logic"
```

---

### Task 4: 终端原始模式与键盘捕获

**Files:**
- Modify: `src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

- [ ] **Step 1: 添加终端配置函数**

在文件顶部 import 区域添加：

```python
import termios
import tty
```

在 socket 函数前添加：

```python
def setup_terminal() -> list:
    """配置终端为原始模式，返回原始属性用于恢复"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old_settings


def restore_terminal(old_settings: list) -> None:
    """恢复终端设置"""
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key() -> str:
    """读取单个按键，返回按键序列"""
    fd = sys.stdin.fileno()
    return sys.stdin.read(1)
```

- [ ] **Step 2: 修改 main() 添加终端配置**

在 socket 初始化后添加：

```python
    old_terminal_settings = setup_terminal()
```

- [ ] **Step 3: 修改 main() 中的循环添加键盘处理**

替换 while 循环为：

```python
    try:
        while True:
            # 尝试接收反馈
            try:
                data, addr = recv_sock.recvfrom(65535)
                recv_msg = json.loads(data.decode("utf-8"))
                print(f"[接收] {addr} <- {json.dumps(recv_msg, ensure_ascii=False)}")
            except socket.timeout:
                pass

            # 检查键盘输入（非阻塞）
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = read_key()
                if key == 'q' or key == 'Q':
                    print("\n收到退出指令")
                    break
                elif key == '\x1b':  # ESC 开始方向键序列
                    key2 = read_key()
                    key3 = read_key()
                    if key2 == '[':
                        if key3 == 'A':  # UP
                            send_command(send_sock, send_addr, 3)
                        elif key3 == 'B':  # DOWN
                            send_command(send_sock, send_addr, 4)
    except KeyboardInterrupt:
        pass
    finally:
        restore_terminal(old_terminal_settings)
        send_sock.close()
        recv_sock.close()
```

- [ ] **Step 5: 测试方向键捕获**

Run: `python3 src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`
按 ↑ 方向键，预期看到 `[发送] {"msg_id": 28989, "pt_focal_en": 3, ...}`
按 ↓ 方向键，预期看到 `[发送] {"msg_id": 28989, "pt_focal_en": 4, ...}`
按 Q，预期退出

- [ ] **Step 6: 提交**

```bash
git add src/gst-cuavcontrolsink/scripts/cuav_visible_control.py
git commit -m "feat: add terminal raw mode and keyboard control"
```

---

### Task 5: 最终测试与完善

**Files:**
- Modify: `src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

- [ ] **Step 1: 添加启动提示**

在 print 语句区域，添加：

```python
    print("=== C-UAV 可见光控制测试工具 ===")
```

- [ ] **Step 2: 完善注释和文档字符串**

确保每个函数有清晰的 docstring

- [ ] **Step 3: 完整测试**

Run: `python3 src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`
验证：
- 显示启动信息
- 按 ↑/↓ 发送命令
- 按 Q 退出，终端恢复正常

- [ ] **Step 4: 提交最终版本**

```bash
git add src/gst-cuavcontrolsink/scripts/cuav_visible_control.py
git commit -m "feat: complete cuav_visible_control with keyboard focal control"
```

---

## 验证方式

1. **基础运行**：`python3 cuav_visible_control.py --help` 正常显示帮助
2. **Socket 初始化**：运行无报错，能正常初始化两个 socket
3. **方向键捕获**：按 ↑ 发送 `pt_focal_en=3`，按 ↓ 发送 `pt_focal_en=4`
4. **Q 退出**：按 Q 后终端恢复正常，程序退出
5. **反馈接收**：收到设备反馈时打印完整 JSON

---

## 已知限制

- 焦距增减步长由设备决定，本工具仅发送使能命令
- 不支持连接超时重试
- 需在支持组播的网络环境运行