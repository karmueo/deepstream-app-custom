#ifndef __GST__MYNETWORK_H__
#define __GST__MYNETWORK_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>

#define PACKAGE "_mynetwork"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "My plugin for Deepstream Network"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty plugin"
#define URL "https://github.com/karmueo/"

G_BEGIN_DECLS

typedef struct _GstmynetworkClass GstmynetworkClass;
typedef struct _Gstmynetwork Gstmynetwork;
typedef struct _SendData SendData;
typedef struct _BboxInfo BboxInfo;
typedef struct _DetectAnalysis DetectAnalysis;

#define GST_TYPE_MYNETWORK (gst_mynetwork_get_type())
#define GST_MYNETWORK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MYNETWORK, Gstmynetwork))

struct _Gstmynetwork
{
    GstBaseSink parent;

    guint unique_id;

    guint gpu_id;

    int sockfd;
    sockaddr_in multicast_addr;
};

struct _GstmynetworkClass
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

#pragma pack(1)
struct _SendData
{
    BboxInfo detect_info;
    gint class_id;
    guint64 object_id;
    gfloat confidence;
    guint64 ntp_timestamp;
    guint source_id;
};

// 统计信息结构体
struct _DetectAnalysis
{
    guint64 frameNum;
    std::map<guint16, guint> primaryClassCountMap;
    std::map<guint16, guint> secondaryClassCountMap;
    // 目标最小像素值
    guint16 minPixel;
    // 平均像素值
    guint16 meanPixel;
};

GType gst_mynetwork_get_type(void);

G_END_DECLS

#endif /* __GST__MYNETWORK_H__ */
