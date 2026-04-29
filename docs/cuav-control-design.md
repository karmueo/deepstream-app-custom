# C-UAV 控制发送设计方案

## 1. 背景

当前仓库已经具备两类相关能力：

- `udpjsonmeta`：接收 UDP 组播 JSON，并解析 C-UAV 协议报文。
- `udpmulticast_sink`：将 DeepStream 推理结果按视频帧节奏封装后通过 UDP 组播发送。

新需求是让 `src/deepstream-app` 支持主动发送 C-UAV 控制 JSON 报文，用于控制：

- 光电伺服控制，报文 `0x7204`
- 可见光控制，报文 `0x7205`

发送方式为 UDP 组播，目标协议以仓库根目录的 `C-UAV_PROTOCOL.md` 为主。

## 2. 结论

建议新开发一个独立的 GStreamer 元件用于控制发送，不在现有 `udpmulticast_sink` 上继续扩展。

推荐总体结构如下：

- 新增专用控制发送元件，例如 `cuavcontrolsink` 或 `udpjsoncontrolsink`
- 由 `deepstream-app` 业务层主动调用该元件发送控制命令
- 元件负责网络发送、JSON 封装、协议公共头填充、参数校验和序号管理
- 应用层负责“何时发送、发什么、是否节流”

这是本项目当前最稳妥的方案。

## 3. 为什么不建议复用现有 `udpmulticast_sink`

现有 `src/gst-udpmulticast_sink/gstudpmulticast_sink.cpp` 的职责是：

- 从 GStreamer pipeline 中接收视频 buffer
- 在 `render()` 中按帧提取目标信息
- 按设定帧率发送目标类数据

它不适合作为控制发送通道，主要原因如下：

### 3.1 控制报文不是按帧驱动

伺服控制和可见光控制本质上是事件驱动或状态驱动，而不是“每来一帧就发一次”。

如果复用当前 sink，会出现这些问题：

- 控制命令被错误绑定到视频帧节奏
- 发送频率难以稳定控制
- 容易在无效帧或高帧率场景下重复发送控制
- 后续节流、防抖、去重逻辑会变得复杂

### 3.2 现有实现的协议方向不同

`udpmulticast_sink` 当前处理的是“检测/目标结果外发”，而不是“设备控制下发”。

两类报文虽然都走 UDP 组播 JSON，但职责完全不同：

- 目标结果发送：偏流式数据输出
- 设备控制发送：偏命令通道

把两种职责堆在同一个元件里，会让元件边界失控。

### 3.3 后续扩展成本高

如果未来还要增加：

- 跟踪控制 `0x7203`
- 红外控制 `0x7206`
- 测距控制 `0x7207`
- 图像控制 `0x720B`

继续复用现有 `udpmulticast_sink` 会进一步加重耦合，维护成本会持续上升。

## 4. 推荐设计

## 4.1 新增专用控制发送元件

新增一个专用 GStreamer 元件，建议命名：

- `cuavcontrolsink`
- 或 `udpjsoncontrolsink`

元件职责限定为：

- 创建并维护 UDP 组播 socket
- 配置组播地址、端口、网卡、TTL
- 统一封装 C-UAV 控制 JSON
- 填充公共报文头
- 管理 `msg_sn`
- 校验控制参数合法性
- 发送控制报文

该元件不负责决定业务时机。

## 4.2 由 `deepstream-app` 负责控制调度

`deepstream-app` 中新增控制调度层，由应用在合适的业务节点主动调用元件接口。

应用层职责：

- 根据识别、跟踪、状态变化决定何时发送
- 做发送节流
- 做重复命令抑制
- 做必要的状态机控制

典型例子：

- 目标刚锁定时发送一次伺服角度控制
- 角度变化小于阈值时不重复发送
- 同类控制报文设置最小发送周期

## 4.3 元件对外接口建议

元件需要提供应用侧可直接调用的发送接口，至少包括：

- `gst_cuav_control_sink_send_servo(...)`
- `gst_cuav_control_sink_send_visible_light(...)`

内部建议定义以下结构：

- `CUAVControlCommonConfig`
- `CUAVServoControlPayload`
- `CUAVVisibleLightControlPayload`

其中：

- `CUAVControlCommonConfig` 负责组播网络参数和公共头默认值
- `CUAVServoControlPayload` 对应 `0x7204`
- `CUAVVisibleLightControlPayload` 对应 `0x7205`

## 4.4 配置建议

建议单独增加新的配置组，例如：

```yaml
cuav-control:
  enable: 1
  multicast-ip: 230.1.88.51
  port: 8003
  iface: eno1
  ttl: 1
  tx-sys-id: 999
  tx-dev-type: 1
  tx-dev-id: 999
  tx-subdev-id: 999
  rx-sys-id: 999
  rx-dev-type: 1
  rx-dev-id: 999
  rx-subdev-id: 999
```

不建议把这部分能力继续塞进现有 `sink2 type=7` 配置中。原因是它不是视频输出 sink，而是控制通道配置。

## 5. 协议实现默认策略

## 5.1 JSON 结构

默认使用扁平 JSON。

原因：

- 仓库内现有 `udpjsonmeta` 接收解析已按扁平 JSON 实现
- 当前项目中的 Python 工具和接收逻辑也已适配扁平结构
- 能减少实现复杂度和两套格式并存的风险

## 5.2 报文 ID 默认值

默认按 `C-UAV_PROTOCOL.md` 直接发送：

- 伺服控制：`msg_id = 0x7204`
- 可见光控制：`msg_id = 0x7205`

`msg_type` 默认使用：

- `0`，即控制报文

公共头默认：

- `cont_type = 0`
- `cont_sum = 1`

## 5.3 协议格式

仓库中现有 Python 示例存在一个协议分歧：

- 示例中控制报文使用 `msg_id = 0x7101`
- 再通过 `cmd_id = 0x7204/0x7205` 表示具体控制类型

而 `C-UAV_PROTOCOL.md` 则直接把控制报文定义为：

- `0x7204`
- `0x7205`

当前实现以协议文档为准，固定直接发送具体控制报文 ID，不再支持 `0x7101 + cmd_id` 包装模式。

## 6. 测试方案

实现后至少需要完成以下验证：

### 6.1 网络与格式验证

- 抓包确认报文实际发往目标组播地址和端口
- 确认 JSON 为 UTF-8 编码
- 确认字段完整、命名正确、类型符合协议

### 6.2 伺服控制验证

发送一条 `0x7204` 报文，检查：

- `msg_id`
- `msg_type`
- `msg_sn` 是否递增
- 时间戳是否正确
- `loc_h`、`loc_v`
- `speed_h`、`speed_v`
- `ctrl_en`、`mode_h`、`mode_v`

### 6.3 可见光控制验证

发送一条 `0x7205` 报文，检查：

- `msg_id`
- `pt_focal`
- `pt_focus`
- `pt_focus_mode`
- `pt_zoom`
- `pt_fov_h`
- `pt_fov_v`

### 6.4 稳定性验证

- 高频业务触发时不阻塞主视频 pipeline
- 节流生效，避免控制报文风暴
- 网卡配置错误时日志可诊断
- 组播地址错误时发送失败路径可观测

### 6.5 协议格式验证

- 抓包确认伺服控制直接使用 `msg_id = 0x7204`
- 抓包确认可见光控制直接使用 `msg_id = 0x7205`

## 7. 风险与回滚

## 7.1 风险

### 协议解释不一致

当前文档与仓库示例对控制报文的 `msg_id` 存在差异。

风险表现：

- 报文已经发出
- 对端也能收到
- 但设备不执行控制

### 控制发送与视频主链路耦合过深

如果把控制逻辑直接绑进按帧运行的 sink，会带来：

- 重复发命令
- 控制频率失控
- 难以调试
- 后续维护困难

## 7.2 回滚策略

若新方案联调中发现异常，可按以下顺序回滚：

1. 不改元件结构，只切换到兼容协议模式
2. 若仍不通，先使用独立测试程序单独验证协议链路
3. 待协议确认后，再重新接入 `deepstream-app`

这样可以把“网络/协议问题”和“应用集成问题”分开排查。

## 8. 实施建议

推荐实施顺序：

1. 先实现独立控制发送元件和最小发送接口
2. 先联通 `0x7204`
3. 再补 `0x7205`
4. 再接入 `deepstream-app` 调度层
5. 最后补节流、去重和兼容模式

这样风险最低，也便于逐步验证。
