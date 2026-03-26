/*
 * C-UAV control sender config definitions
 */
#ifndef _NVGSTDS_CUAV_CONTROL_H_
#define _NVGSTDS_CUAV_CONTROL_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    gboolean enable; /* 是否启用控制发送 */
    gchar *multicast_ip; /* 测试/目标组播地址 */
    guint port; /* 测试/目标组播端口 */
    gchar *iface; /* 绑定网卡名 */
    guint ttl; /* 组播 TTL */
    gboolean compat_cmd_wrapper; /* 是否使用 0x7101 + cmd_id 兼容模式 */
    gboolean debug; /* 是否打印调试日志 */
    gboolean print_upstream_state; /* 是否打印上游 udpjsonmeta 中转的状态报文 */
    guint tx_sys_id; /* 发送方系统号 */
    guint tx_dev_type; /* 发送方设备类型 */
    guint tx_dev_id; /* 发送方设备 ID */
    guint tx_subdev_id; /* 发送方子设备 ID */
    guint rx_sys_id; /* 接收方系统号 */
    guint rx_dev_type; /* 接收方设备类型 */
    guint rx_dev_id; /* 接收方设备 ID */
    guint rx_subdev_id; /* 接收方子设备 ID */
    gboolean send_test_on_startup; /* pipeline 启动后是否自动发送测试报文 */
    gboolean auto_track_enable; /* 是否启用基于检测/跟踪结果的自动控制 */
    guint control_source_id; /* 自动控制使用的 source_id */
    guint control_period_ms; /* 自动控制最小发送周期 */
    guint target_lost_hold_ms; /* 目标丢失后的保持时间 */
    guint tracking_history_size; /* 速度估计使用的历史样本数 */
    gdouble center_deadband_x; /* 水平中心死区，归一化到 [-1, 1] */
    gdouble center_deadband_y; /* 垂直中心死区，归一化到 [-1, 1] */
    gdouble servo_kp_x; /* 水平位置误差增益 */
    gdouble servo_kp_y; /* 垂直位置误差增益 */
    gdouble servo_kv_x; /* 水平速度前馈增益 */
    gdouble servo_kv_y; /* 垂直速度前馈增益 */
    gint servo_dir_x; /* 水平控制方向，1 或 -1 */
    gint servo_dir_y; /* 垂直控制方向，1 或 -1 */
    gdouble servo_max_step_h; /* 单次水平位置最大调整量（度） */
    gdouble servo_max_step_v; /* 单次垂直位置最大调整量（度） */
    guint servo_min_speed; /* 伺服最小速度 */
    guint servo_max_speed; /* 伺服最大速度 */
    gdouble zoom_target_ratio_min; /* 目标高度占比下限 */
    gdouble zoom_target_ratio_max; /* 目标高度占比上限 */
    gdouble zoom_deadband; /* 变倍死区 */
    gdouble zoom_kp; /* 焦距闭环增益 */
    gdouble zoom_max_step; /* 单次焦距最大调整量 */
    gboolean visible_light_control_enable; /* 是否发送可见光控制 */
    gboolean infrared_control_enable; /* 是否发送红外控制 */
    guint servo_dev_id; /* 伺服报文 dev_id，0=可见光 1=热成像 2=两者 */
    gdouble pt_focal_min; /* 可见光焦距最小值 */
    gdouble pt_focal_max; /* 可见光焦距最大值 */
    gdouble ir_zoom_kp; /* 红外焦距闭环增益 */
    gdouble ir_zoom_max_step; /* 红外单次焦距最大调整量 */
    gdouble ir_focal_min; /* 红外焦距最小值 */
    gdouble ir_focal_max; /* 红外焦距最大值 */
    guint ir_focus_default; /* 红外默认聚焦值 */
    gboolean simulate_target_enable; /* 无目标时是否启用仿真目标 */
    gdouble simulate_target_amplitude_x; /* 仿真目标水平摆动幅度，归一化 */
    gdouble simulate_target_amplitude_y; /* 仿真目标垂直摆动幅度，归一化 */
    gdouble simulate_target_ratio_min; /* 仿真目标高度占比下限 */
    gdouble simulate_target_ratio_max; /* 仿真目标高度占比上限 */
    guint simulate_target_period_ms; /* 仿真目标运动周期 */
    gboolean real_device_test_enable; /* 是否启用真实设备单项控制测试 */
    gboolean real_device_test_zoom_only_enable; /* 是否仅执行变焦测试 */
    guint real_device_test_item; /* 真实设备测试项选择：0=all 1=servo_h 2=servo_v 3=visible_focal */
    guint real_device_test_command_interval_ms; /* 单项测试发指令间隔 */
    guint real_device_test_observe_timeout_ms; /* 单项测试观察超时 */
    guint real_device_test_settle_ms; /* 单项测试结束后的稳定等待时间 */
    guint real_device_test_repeat_count; /* 每个测试项重复次数 */
    gdouble servo_effect_threshold_h; /* 水平伺服生效判定阈值 */
    gdouble servo_effect_threshold_v; /* 垂直伺服生效判定阈值 */
    gdouble focal_effect_threshold; /* 焦距生效判定阈值 */
    gdouble ir_focal_effect_threshold; /* 红外焦距生效判定阈值 */
    guint state_stale_timeout_ms; /* 设备状态新鲜度超时 */
} NvDsCuavControlConfig;

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_CUAV_CONTROL_H_ */
