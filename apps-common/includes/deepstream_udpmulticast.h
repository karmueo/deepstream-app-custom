/*
 * UDP Multicast config definitions for custom udpmulticast GStreamer source
 */
#ifndef _NVGSTDS_UDP_MULTICAST_H_
#define _NVGSTDS_UDP_MULTICAST_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    gboolean enable;          /* enable udpmulticast source */
    guint gpu_id;             /* (reserved for future zero-copy usage) */
    gchar *multicast_ip;      /* e.g. "239.255.0.1" */
    guint port;               /* UDP port */
    gchar *iface;             /* network interface name (e.g. "eth0"), optional */
    guint recv_buf_size;      /* socket receive buffer (bytes), 0 = default */
} NvDsUdpMulticastConfig;

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_UDP_MULTICAST_H_ */
