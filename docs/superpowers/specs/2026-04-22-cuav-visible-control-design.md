# C-UAV 可见光控制测试工具设计

## 1. 概述

**文件位置**：`src/gst-cuavcontrolsink/scripts/cuav_visible_control.py`

**功能**：通过键盘交互发送可见光控制命令（焦距调整）到光电设备，并监听设备反馈。

**目标用户**：调试光电设备的可见光控制功能。

---

## 2. 网络配置

| 参数 | 值 |
|------|---|
| 发送地址 | `230.1.88.51:8003` |
| 反馈监听地址 | `230.1.88.51:8013` |
| 协议 | UDP 组播 |
| 报文格式 | JSON |

---

## 3. 键盘控制

| 按键 | 动作 | 发送内容 |
|------|------|---------|
| `↑` (UP) | 焦距增加 | `pt_focal_en=3` |
| `↓` (DOWN) | 焦距减小 | `pt_focal_en=4` |
| `Q` | 退出程序 | 无 |

每次按键发送完整 JSON 报文，设备自行决定步长。

---

## 4. 报文结构

### 4.1 发送报文（报文ID 0x7205）

```json
{
  "msg_id": 28989,
  "msg_sn": <递增序号>,
  "msg_type": 0,
  "tx_sys_id": 5,
  "tx_dev_type": 5,
  "tx_dev_id": 1,
  "tx_subdev_id": 999,
  "rx_sys_id": 999,
  "rx_dev_type": 999,
  "rx_dev_id": 999,
  "rx_subdev_id": 999,
  "yr": <当前年>,
  "mo": <当前月>,
  "dy": <当前日>,
  "h": <当前时>,
  "min": <当前分>,
  "sec": <当前秒>,
  "msec": <当前毫秒>,
  "cont_type": 0,
  "cont_sum": 1,
  "pt_dev_en": 1,
  "pt_ctrl_en": 1,
  "pt_fov_en": 0,
  "pt_fov_h": 0,
  "pt_fov_v": 0,
  "pt_focal_en": 3,
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
  "pt_zoom": 0
}
```

### 4.2 接收反馈

程序监听 `230.1.88.51:8013`，收到反馈后打印完整 JSON 内容和来源地址。

---

## 5. 程序架构

```
cuav_visible_control.py
├── 发送线程/主循环
│   ├── 初始化发送socket (UDP)
│   ├── 初始化接收socket (UDP，加入组播230.1.88.51:8013)
│   ├── 配置终端为原始模式（捕获方向键）
│   └── 循环:
│       ├── 读取按键
│       ├── 根据按键构建JSON报文
│       ├── 发送UDP报文到230.1.88.51:8003
│       ├── 尝试接收反馈（non-blocking）
│       └── 打印发送/接收的完整JSON
└── 退出时恢复终端设置
```

---

## 6. 技术要点

### 6.1 终端配置
- 使用 `termios` 将标准输入设为原始模式
- 使用 `tty.setraw(fd)` 或 `termios.tcsetattr()`
- 捕获字节序列识别方向键：
  - `↑`: `\x1b[A`
  - `↓`: `\x1b[B`

### 6.2 公共头字段
除 `pt_focal_en` 外，其他字段使用固定测试值，确保协议兼容性。

### 6.3 错误处理
- Socket 发送失败：打印错误，继续运行
- Socket 接收超时：忽略，继续监听
- 非法按键：忽略

---

## 7. 使用方式

```bash
python3 cuav_visible_control.py [--send-ip <ip>] [--send-port <port>] [--recv-ip <ip>] [--recv-port <port>]
```

默认参数：
- 发送：`230.1.88.51:8003`
- 接收：`230.1.88.51:8013`

---

## 8. 预期输出示例

```
=== C-UAV 可见光控制测试工具 ===
发送 -> 230.1.88.51:8003
接收 <- 230.1.88.51:8013

按 ↑/↓ 调整焦距，按 Q 退出

[发送] {"msg_id": 28989, "pt_focal_en": 3, ...}
[接收] 192.168.1.100:8013 <- {"msg_id": 28989, "sv_stat": 1, ...}

[发送] {"msg_id": 28989, "pt_focal_en": 4, ...}
[接收] 192.168.1.100:8013 <- {"msg_id": 28989, "sv_stat": 1, ...}
```

---

## 9. 依赖

- Python 3 标准库：`socket`, `json`, `termios`, `sys`, `time`
- 无第三方依赖