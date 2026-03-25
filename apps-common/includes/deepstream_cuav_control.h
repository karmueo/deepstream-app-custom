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
    guint tx_sys_id; /* 发送方系统号 */
    guint tx_dev_type; /* 发送方设备类型 */
    guint tx_dev_id; /* 发送方设备 ID */
    guint tx_subdev_id; /* 发送方子设备 ID */
    guint rx_sys_id; /* 接收方系统号 */
    guint rx_dev_type; /* 接收方设备类型 */
    guint rx_dev_id; /* 接收方设备 ID */
    guint rx_subdev_id; /* 接收方子设备 ID */
    gboolean send_test_on_startup; /* pipeline 启动后是否自动发送测试报文 */
} NvDsCuavControlConfig;

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_CUAV_CONTROL_H_ */
