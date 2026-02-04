/*
 * UDP JSON Meta config definitions for custom udpjsonmeta GStreamer plugin
 */
#ifndef _NVGSTDS_UDPJSON_META_H_
#define _NVGSTDS_UDPJSON_META_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    gboolean enable; /* 是否启用 */
    gchar *multicast_ip; /* 组播地址 */
    gchar *iface; /* 绑定网卡名 */
    guint recv_buf_size; /* 接收缓冲区大小 */
    guint cache_ttl_ms; /* 缓存有效期(毫秒) */
    guint max_cache_size; /* 最大缓存条目数 */
    /* C-UAV 协议配置 */
    gboolean enable_cuav_parser; /* 是否启用 C-UAV 协议解析 */
    guint cuav_port; /* C-UAV 组播端口 */
    guint cuav_ctrl_port; /* C-UAV 控制/引导端口 */
    gboolean enable_cuav_debug; /* 是否启用 C-UAV 调试打印 */
} NvDsUdpJsonMetaConfig;

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_UDPJSON_META_H_ */
