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
    guint port; /* 组播端口 */
    gchar *iface; /* 绑定网卡名 */
    guint recv_buf_size; /* 接收缓冲区大小 */
    gchar *json_key; /* JSON 值键 */
    gchar *object_id_key; /* JSON 目标ID键 */
    gchar *source_id_key; /* JSON 源ID键 */
    guint cache_ttl_ms; /* 缓存有效期(毫秒) */
    guint max_cache_size; /* 最大缓存条目数 */
} NvDsUdpJsonMetaConfig;

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_UDPJSON_META_H_ */
