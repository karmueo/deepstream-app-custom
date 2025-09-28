#ifndef __GST__UDPMULTICAST_SINK_H__
#define __GST__UDPMULTICAST_SINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#ifdef __cplusplus
#include <map>
#endif

#define PACKAGE "_udpmulticast_sink"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "My plugin for Deepstream Network"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty plugin"
#define URL "https://github.com/karmueo/"

G_BEGIN_DECLS

typedef struct _Gstudpmulticast_sinkClass Gstudpmulticast_sinkClass;
typedef struct _Gstudpmulticast_sink Gstudpmulticast_sink;
typedef struct _SendData SendData;
typedef struct _BboxInfo BboxInfo;
typedef struct _DetectAnalysis DetectAnalysis;

#define GST_TYPE_UDPMULTICAST_SINK (gst_udpmulticast_sink_get_type())
#define GST_UDPMULTICAST_SINK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_UDPMULTICAST_SINK, Gstudpmulticast_sink))

struct _Gstudpmulticast_sink
{
    GstBaseSink parent;

    guint unique_id;

    guint gpu_id;

    int sockfd;
    struct sockaddr_in multicast_addr;
    // configurable multicast params
    gchar *ip;   // multicast ip string
    guint  port; // multicast port
};

struct _Gstudpmulticast_sinkClass
{
    GstBaseSinkClass parent_class;
};

struct _BboxInfo
{
    float left;
    float top;
    float width;
    float height;
};

// 统计信息结构体 (only used in C++)
#ifdef __cplusplus
struct _DetectAnalysis {
    guint64 frameNum;
    std::map<guint16, guint> primaryClassCountMap;
    std::map<guint16, guint> secondaryClassCountMap;
    guint16 minPixel;  // 目标最小像素值
    guint16 meanPixel; // 平均像素值
};
#else
typedef struct _DetectAnalysis {
    guint64 frameNum;
    guint16 minPixel;
    guint16 meanPixel;
} _DetectAnalysis;
#endif

GType gst_udpmulticast_sink_get_type(void);

G_END_DECLS

#endif /* __GST__UDPMULTICAST_SINK_H__ */
