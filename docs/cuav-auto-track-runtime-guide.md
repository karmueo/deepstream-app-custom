# C-UAV 真实目标自动跟踪运行说明

本文说明当前仓库里 `deepstream-app` 在启用 C-UAV 控制发送后，如何把真实检测目标转换成云台伺服和光电控制报文，并形成闭环。

说明重点是“实际运行链路”，不是协议草案。相关实现主要分布在：

- `src/deepstream-app/deepstream_app.c`
- `src/gst-cuavcontrolsink/gstcuavcontrolsink.c`
- `apps-common/includes/deepstream_cuav_control.h`
- `src/deepstream-app/configs/yml/cuav_control_sink.yml`
- `C-UAV_PROTOCOL.md`

## 1. 这个功能解决什么问题

当检测器和跟踪器已经在视频流里找到真实目标后，程序会自动：

1. 选择一个当前要跟随的目标
2. 计算目标相对画面中心的偏差
3. 把偏差转换成云台水平/俯仰控制量
4. 根据目标大小决定是否调整可见光焦距
5. 通过 UDP 组播发送 C-UAV 控制报文
6. 接收设备反馈，再用反馈修正下一次控制

如果当前没有真实目标，还可以选择：

- 保持最近一次控制状态一段时间
- 走仿真目标继续驱动闭环
- 启动时发送预置位报文
- 做角点 + 变焦循环测试

## 2. 启动前提

建议按下面顺序确认：

1. 已经编译并安装 `src/gst-cuavcontrolsink`
2. `deepstream-app` 已经能正常跑起来
3. `sink5` 已启用，且类型为 `8`
4. `cuav-control-config-file` 正确指向 `cuav_control_sink.yml`
5. 真实设备组播地址和端口正确，默认是 `230.1.88.51:8003`
6. 反馈端口默认是 `230.1.88.51:8013`
7. 如果是多路输入，已经明确了要控制哪一路 `control-source-id`

当前仓库的默认配置入口在：

- `src/deepstream-app/configs/yml/app_config.yml`
- `src/deepstream-app/configs/yml/cuav_control_sink.yml`

## 3. 总体链路

运行时，控制链路可以理解成下面这条线：

```text
检测/跟踪结果
  -> 选目标
  -> 计算中心偏差与目标占比
  -> 计算伺服/焦距命令
  -> 发送 0x7204 / 0x7205 / 0x7206
  -> 设备反馈 0x7201
  -> 更新反馈状态
  -> 下一轮继续修正
```

在代码里，大致对应：

- 目标筛选和控制决策：`src/deepstream-app/deepstream_app.c`
- 控制报文发送：`src/gst-cuavcontrolsink/gstcuavcontrolsink.c`
- 反馈解析与状态缓存：`src/deepstream-app/deepstream_app.c`

## 4. 运行时控制优先级

程序不是一上来就自动跟踪真实目标，而是按优先级处理：

1. 启动预置位
2. 四角 + 变焦循环
3. 自动目标跟踪
4. 发送测试报文

也就是说，只要前面的功能开启，后面的自动跟踪会被前者压住。

### 4.1 启动预置位

如果配置里设置了：

- `corner-home-loc-h-deg`
- `corner-home-loc-v-deg`
- `corner-home-pt-focus`
- `startup-pt-focal-min-enable`
- `startup-pt-focal-min-hold-ms`

程序会先做“回到预置位”的流程，再进入后续控制。

对于云台位置，程序会发送一条 `0x7204`，把水平和垂直位置拉到预置值。

对于可见光，`startup-pt-focal-min-enable=1` 时会在启动时先发送一条 `0x7205`，把焦距拉到最小；随后等待 `startup-pt-focal-min-hold-ms` 指定的时间，再发送 `pt_focal_en=0` 停止。`corner-home-pt-focus` 仍然只负责聚焦预置。

启动预置是否完成，取决于反馈是否达到阈值，或者等待超时。

### 4.2 四角 + 变焦循环

如果开启 `corner-zoom-cycle-enable=1`，程序会优先进入四角测试逻辑，自动在四个角点之间切换，并配合焦距拉近/拉远。

这个模式主要用于联调和校准，不属于真实目标自动跟踪主流程。四角回位阶段不再读取数值型焦距预置，只保留 `corner-home-pt-focus`。

### 4.3 自动跟踪

只有在：

- `auto-track-enable=1`
- 没有开启四角循环
- 已经通过启动预置

时，程序才会进入真实目标的自动跟踪闭环。

### 4.4 发送测试报文

如果 `send-test-on-startup=1`，且没有开启自动跟踪和四角循环，程序会发一组测试报文，用于快速确认控制通道是否通。

## 5. 真实目标如何被选中

自动跟踪只看配置里的一个视频源：

- `control-source-id`

程序会遍历批次里的 `frame_meta_list`，只取 `source_id` 和这个值相等的那一路。

如果这一路里有多个目标，优先级如下：

1. 如果当前已经锁定过一个目标，并且这个目标还在画面里，继续跟随这个目标
2. 否则在当前帧里找评分最高的那个目标

目标要满足两个条件才会进入控制闭环：

- `object_id != UNTRACKED_OBJECT_ID`
- 目标框宽高大于 1 像素

评分逻辑是：

- 优先用 `tracker_confidence`
- 如果没有 tracker 置信度，就退回到检测置信度 `confidence`

这意味着：

- 跟踪器稳定时，程序更倾向于跟随已锁定目标
- 目标丢失后，会重新挑选当前帧里最优目标

## 6. 目标坐标是怎么变成控制量的

程序把目标框转换成归一化偏差，作为控制输入。

### 6.1 中心偏差

对当前目标框，先算中心点：

- `cx = left + width / 2`
- `cy = top + height / 2`

然后归一化到 `[-1, 1]`：

- `err_x = (cx - frame_width / 2) / (frame_width / 2)`
- `err_y = (cy - frame_height / 2) / (frame_height / 2)`

含义是：

- `err_x > 0` 表示目标在画面右侧
- `err_x < 0` 表示目标在画面左侧
- `err_y > 0` 表示目标在画面下方
- `err_y < 0` 表示目标在画面上方

### 6.2 目标大小占比

程序还会算目标高度占画面高度的比例：

- `target_ratio = target_height / frame_height`

这个值用于自动变焦判断：

- 比例太小，说明目标看起来太远，需要拉近
- 比例太大，说明目标太近，需要缩小

### 6.3 历史速度估计

程序会保留最近 `tracking-history-size` 个样本，默认 5 个。

用最早和最新样本估算速度：

- `vel_x = (last.err_x - first.err_x) / dt`
- `vel_y = (last.err_y - first.err_y) / dt`

这个速度不是物理速度，而是图像偏差变化速度，用来做前馈补偿。

## 7. 云台伺服是怎么控制的

云台控制报文对应 `0x7204`。

### 7.1 伺服控制输入

自动控制时，程序会计算：

- 水平位置修正量
- 垂直位置修正量
- 水平速度
- 垂直速度

核心配置项是：

- `center-deadband-x`
- `center-deadband-y`
- `servo-kp-x`
- `servo-kp-y`
- `servo-kv-x`
- `servo-kv-y`
- `servo-dir-x`
- `servo-dir-y`
- `servo-max-step-h`
- `servo-max-step-v`
- `servo-min-speed`
- `servo-max-speed`

### 7.2 控制公式

当目标偏差超过死区时，程序会计算：

- `delta_h = kp_x * err_x + kv_x * vel_x`
- `delta_v = kp_y * err_y + kv_y * vel_y`

然后乘以方向系数：

- `delta_h *= servo_dir_x`
- `delta_v *= servo_dir_y`

最后对单次动作做限幅：

- 水平限幅到 `[-servo-max-step-h, servo-max-step-h]`
- 垂直限幅到 `[-servo-max-step-v, servo-max-step-v]`

### 7.3 伺服基准值从哪里来

如果反馈里有新鲜的云台位置，程序优先用反馈值作为基准：

- `st_loc_h`
- `st_loc_v`

如果反馈过期或没有反馈，就退回到最近一次发送的命令值。

### 7.4 伺服速度怎么定

速度不是固定值，而是按“动作强度”动态计算的。

程序会把下面几个量合成一个控制强度：

- 当前偏差的绝对值
- 单次位置修正占最大步长的比例

然后把强度映射到：

- `servo-min-speed`
- `servo-max-speed`

所以：

- 目标偏差越大，速度越高
- 目标越接近中心，速度越低

### 7.5 伺服报文实际发什么

程序发 `0x7204` 时，主要字段是：

- `dev_id`：由 `servo-dev-id` 决定，默认 `2`
- `dev_en=1`
- `ctrl_en=1`
- `mode_h=0`
- `mode_v=0`
- `speed_en_h=1`
- `speed_h`
- `speed_en_v=1`
- `speed_v`
- `loc_en_h=1`
- `loc_h`
- `loc_en_v=1`
- `loc_v`
- `offset_en=0`
- `offset_h=0`
- `offset_v=0`

这里的含义是：

- 直接给云台一个新的目标角度
- 同时附带速度
- 不依赖脱靶量模式

### 7.6 什么时候会重复发伺服

程序按 `control-period-ms` 节流，默认 100ms 一次。

只要满足时间条件，就可能再次发送伺服命令，即使目标没有明显移动。

这是为了：

- 保持控制闭环持续刷新
- 让设备端能持续收到当前状态

## 8. 可见光是怎么控制的

可见光控制报文对应 `0x7205`。

### 8.1 自动控制重点

自动跟踪里，可见光主要控制的是：

- 焦距 `pt_focal`
- 聚焦值 `pt_focus` 的状态保持

当前自动闭环里，主要是“焦距自动变倍”，不是每次都主动改聚焦。

### 8.2 焦距如何计算

程序先看目标高度占比 `target_ratio`。

如果目标太小：

- 说明看起来太远
- 程序会让焦距增大，也就是拉近

如果目标太大：

- 说明看起来太近
- 程序会让焦距减小，也就是缩小

对应配置项：

- `zoom-target-ratio-min`
- `zoom-target-ratio-max`
- `zoom-deadband`
- `pt-focal-min`
- `pt-focal-max`

### 8.3 焦距基准值从哪里来

如果之前已经发过可见光控制，并且反馈/本地状态还有效，程序会基于最近一次焦距继续修正。

如果需要在启动时先把镜头拉到最小焦距，就开启 `startup-pt-focal-min-enable`，并用 `startup-pt-focal-min-hold-ms` 调整保持时间。

程序会先发送 `pt_focal_en=4`，随后按 `startup-pt-focal-min-hold-ms` 指定的时间发送 `pt_focal_en=0`。

数值型焦距预置已废弃，不再读取。

### 8.4 可见光报文实际发什么

自动跟踪时，程序发 `0x7205` 的主要字段是：

- `pt_dev_en=1`
- `pt_ctrl_en=1`
- `pt_fov_en=0`
- `pt_focal_en=1`
- `pt_focal`
- `pt_focus_en=0`
- `pt_focus`
- `pt_speed_en=0`
- `pt_focus_speed=0`
- `pt_bri_en=0`
- `pt_ctrs_en=0`
- `pt_ofr_en=0`
- `pt_focus_mode=0`
- `pt_zoom=0`

也就是说：

- 自动跟踪阶段主要是改焦距
- 聚焦值更多是作为状态保持和预置使用

### 8.5 什么时候不再发相同焦距

可见光控制有去重逻辑。

如果当前已经进入自动闭环，且新算出来的焦距和上一次值差别很小，程序会抑制这次发送。

如果设备会把 `pt_focal_en` 解释为持续变焦命令，那么 `visible-focal-hold-ms` 会在到时后强制补发一次 `pt_focal_en=0` 用来停止镜头连续变化；这个值过小会缩短拉近时间，过大会让镜头持续变焦太久。

这样做是为了减少：

- 无意义报文
- 网络带宽浪费
- 设备端抖动

## 9. 红外控制是怎么接入的

如果 `infrared-control-enable=1`，程序会同时计算红外焦距控制，并发送 `0x7206`。

红外控制逻辑与可见光类似，也是根据目标占比决定焦距增减，只是参数范围不同：

- `ir-zoom-kp`
- `ir-zoom-max-step`
- `ir-focal-min`
- `ir-focal-max`
- `ir-focus-default`

如果你当前设备只想用可见光联动，可以把：

- `infrared-control-enable=0`

关闭。

## 10. 反馈是怎么参与闭环的

设备反馈通常是 `0x7201`，程序会把它写入本地状态缓存。

反馈里重点字段包括：

- `st_loc_h`
- `st_loc_v`
- `pt_focal`
- `pt_focus`
- `ir_focal`
- `ir_focus`
- `sv_stat`
- `trk_dev`
- `pt_trk_link`
- `ir_trk_link`
- `trk_stat`

写入后，下一轮控制就会优先用这些反馈值作为基准。

### 10.1 反馈新鲜度

程序把反馈分成“新鲜”和“过期”两种状态。

如果反馈太久没更新，后续控制会退回本地最后一次命令值，避免一直依赖旧反馈。

### 10.2 打印反馈

如果打开了：

- `print-upstream-state=1`

程序会把上游解析到的状态报文打印出来，方便确认反馈是否真的进入闭环。

## 11. 目标丢失后会发生什么

当当前锁定目标消失时，程序不会立刻清空状态，而是会先保留一段时间。

对应配置：

- `target-lost-hold-ms`

默认 1000ms。

这个窗口内的行为是：

- 继续保持最近一次锁定状态
- 允许短暂遮挡、抖动或漏检

超过这个时间后：

- 清空锁定目标
- 允许重新选择新目标

这样做是为了避免目标短暂消失时，云台来回跳变。

## 12. 模拟目标什么时候会用到

如果没有真实目标，但你又想继续验证控制闭环，可以打开：

- `simulate-target-enable=1`

此时程序会构造一个随时间变化的模拟目标，继续驱动：

- 云台伺服
- 可见光变焦
- 红外变焦（如果开启）

模拟目标的运动幅度、占比和周期都可以配置：

- `simulate-target-amplitude-x`
- `simulate-target-amplitude-y`
- `simulate-target-ratio-min`
- `simulate-target-ratio-max`
- `simulate-target-period-ms`

这个模式适合：

- 没有真实目标时测试自动闭环
- 验证报文格式
- 验证控制频率和日志

## 13. 配置项怎么理解

下面这些参数最关键：

| 参数 | 作用 |
| --- | --- |
| `auto-track-enable` | 是否根据检测/跟踪结果自动发控制报文 |
| `control-source-id` | 只读取哪一路视频源的检测结果 |
| `control-period-ms` | 控制报文最小发送周期 |
| `tracking-history-size` | 速度估计的历史样本数 |
| `center-deadband-x/y` | 目标靠近中心时不再修正的死区 |
| `servo-kp-x/y` | 伺服位置误差增益 |
| `servo-kv-x/y` | 伺服速度前馈增益 |
| `servo-dir-x/y` | 伺服方向修正 |
| `servo-max-step-h/v` | 单次最大修正量 |
| `servo-min-speed/max-speed` | 伺服速度上下限 |
| `zoom-target-ratio-min/max` | 焦距控制目标占比区间 |
| `visible-focal-hold-ms` | 发送 `pt_focal_en` 后强制补发 `pt_focal_en=0` 的等待时间 |
| `visible-light-control-enable` | 是否发可见光控制 |
| `infrared-control-enable` | 是否发红外控制 |
| `servo-dev-id` | 伺服控制报文里的设备选择 |
| `target-lost-hold-ms` | 目标消失后保持锁定的时间 |
| `simulate-target-enable` | 是否用仿真目标替代真实目标 |
| `send-test-on-startup` | 启动后是否先发测试报文 |

## 14. 一个真实运行例子

假设：

- `auto-track-enable=1`
- `control-source-id=0`
- `visible-light-control-enable=1`
- `servo-dev-id=2`
- 画面里出现一个已被 tracker 跟踪的目标

程序会这样做：

1. 在第 0 路视频里找目标
2. 选择当前锁定目标，或者最高评分目标
3. 计算目标中心与画面中心的偏差
4. 计算速度前馈
5. 生成一条 `0x7204` 伺服报文，让云台向目标中心移动
6. 计算目标大小占比
7. 生成一条 `0x7205` 可见光报文，调整焦距
8. 设备返回 `0x7201`
9. 程序更新本地反馈
10. 下一帧继续修正，直到目标进入死区

当目标已经基本居中且大小合适时：

- 伺服动作会变小
- 发送频率会按节流维持
- 焦距不会再大幅变化

## 15. 调试建议

### 15.1 先看日志

推荐重点看这些关键字：

- `[cuav][control][auto]`
- `[cuav][control][sim]`
- `[cuav][startup-preset]`
- `[cuav][corner-zoom]`
- `[cuav][eo-system]`
- `[cuav][servo]`

### 15.2 再看抓包

抓包时重点确认：

- 目的地址是 `230.1.88.51`
- 端口是 `8003`
- 报文 `msg_id` 正确
- JSON 字段完整

### 15.3 再看反馈

如果能收到 `8013` 端口反馈，说明控制闭环至少在网络层已经形成。

### 15.4 再看状态是否更新

如果打开 `print-upstream-state=1`，可以直接观察：

- 云台位置是否变化
- 焦距是否变化
- 跟踪状态是否正常

## 16. 常见问题

### 16.1 为什么有目标却不发控制

常见原因：

- `auto-track-enable=0`
- `control-source-id` 选错了
- 目标没有被 tracker 跟踪到
- 还卡在启动预置或角点循环
- `control-period-ms` 太大，看起来像没有更新

### 16.2 为什么只动一次

常见原因：

- 目标已经进入死区
- 焦距变化被去重抑制
- 反馈过期，程序退回本地保持值

### 16.3 为什么焦距没动

常见原因：

- `visible-light-control-enable=0`
- 目标占比一直落在目标区间内
- `zoom-deadband` 太大
- `pt_focal` 已经到上下限

### 16.4 为什么多路视频时跟错目标

常见原因：

- `control-source-id` 没选对
- 目标锁定后，其他路再出现高分目标，但当前路才是控制源

## 17. 风险与回滚

### 17.1 风险

这是会真实驱动设备的控制链路。

如果组播地址、端口、网卡或 `servo-dev-id` 配错，程序可能会把控制报文发给错误设备。

### 17.2 回滚方式

最稳妥的回滚方式是：

1. 把 `sink5.enable` 改成 `0`
2. 或把 `auto-track-enable` 改成 `0`
3. 或把 `visible-light-control-enable` / `infrared-control-enable` 关闭
4. 或临时把组播地址改到测试网段

如果只是想恢复旧的测试模式：

- 保持 `cuav-control-config-file` 不变
- 关闭自动跟踪
- 只保留 `send-test-on-startup=1`

## 18. 最小可执行启动路径

如果你只想快速验证“真实目标 -> 云台/光电闭环”，建议先跑这个最小配置：

```yaml
sink5:
  enable: 1
  type: 8
  cuav-control-config-file: cuav_control_sink.yml
```

并在 `cuav_control_sink.yml` 里确保：

- `auto-track-enable=1`
- `visible-light-control-enable=1`
- `control-source-id` 正确
- `multicast-ip=230.1.88.51`
- `port=8003`

然后启动：

```bash
./build/deepstream-app -c ./src/deepstream-app/configs/yml/app_config.yml
```

如果一切正常，真实目标进入后就会看到云台和可见光开始跟随调整。
