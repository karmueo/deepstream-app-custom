/**
 * SECTION:element-_mynetwork
 *
 * FIXME:Describe _mynetwork here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! _mynetwork ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/gstinfo.h>
// #include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include <gst/base/gstbasetransform.h>
#include <gst/gstelement.h>
#include <gst/gstinfo.h>
#include "nvbufsurface.h"
#include "gstmynetwork.h"
#include "cuda_runtime_api.h"
#include <math.h>

/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
/* 启用将转换后的 cvmat 写入文件 */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

GST_DEBUG_CATEGORY_STATIC(gst_mynetwork_debug);
#define GST_CAT_DEFAULT gst_mynetwork_debug

#define CHECK_CUDA_STATUS(cuda_status, error_str)                                  \
    do                                                                             \
    {                                                                              \
        if ((cuda_status) != cudaSuccess)                                          \
        {                                                                          \
            g_print("Error: %s in %s at line %d (%s)\n",                           \
                    error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
            goto error;                                                            \
        }                                                                          \
    } while (0)

/* Filter signals and args */
enum
{
    /* FILL ME */
    LAST_SIGNAL
};

enum
{
    PROP_0,
    PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE("sink",
                                                                   GST_PAD_SINK,
                                                                   GST_PAD_ALWAYS,
                                                                   GST_STATIC_CAPS("ANY"));

static void gst_mynetwork_set_property(GObject *object,
                                       guint property_id,
                                       const GValue *value,
                                       GParamSpec *pspec);
static void gst_mynetwork_get_property(GObject *object,
                                       guint property_id,
                                       GValue *value,
                                       GParamSpec *pspec);
static void gst_mynetwork_finalize(GObject *object);

#define gst_mynetwork_parent_class parent_class
G_DEFINE_TYPE(Gstmynetwork, gst_mynetwork, GST_TYPE_BASE_SINK);

static gboolean gst_mynetwork_set_caps(GstBaseSink *sink, GstCaps *caps);

static GstFlowReturn gst_mynetwork_render(GstBaseSink *sink, GstBuffer *buf);
static gboolean gst_mynetwork_start(GstBaseSink *sink);
static gboolean gst_mynetwork_stop(GstBaseSink *sink);

/* GObject vmethod implementations */

/* initialize the _mynetwork's class */
static void
gst_mynetwork_class_init(GstmynetworkClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class;
    GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS(klass);

    gstelement_class = (GstElementClass *)klass;

    gobject_class->set_property = gst_mynetwork_set_property;
    gobject_class->get_property = gst_mynetwork_get_property;
    gobject_class->finalize = gst_mynetwork_finalize;

    base_sink_class->render = GST_DEBUG_FUNCPTR(gst_mynetwork_render);
    base_sink_class->start = GST_DEBUG_FUNCPTR(gst_mynetwork_start);
    base_sink_class->stop = GST_DEBUG_FUNCPTR(gst_mynetwork_stop);
    base_sink_class->set_caps = GST_DEBUG_FUNCPTR(gst_mynetwork_set_caps);

    gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                              &sink_factory);

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
                                         "DsMyNetwork plugin",
                                         "DsMyNetwork Plugin",
                                         "Process a infer mst network on objects / full frame",
                                         "ShenChangli "
                                         "@ karmueo@163.com");
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad callback functions
 * initialize instance structure
 * 初始化新element
 * 实例化 pads 并将它们添加到element中
 * 设置 pad 回调函数
 * 初始化实例结构
 */
static void
gst_mynetwork_init(Gstmynetwork *self)
{
    // 初始化一些参数
    self->gpu_id = 0;

    // 创建UDP Socket
    self->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (self->sockfd < 0)
    {
        GST_ERROR("Failed to create socket");
        return;
    }

    // 设置组播地址
    memset(&self->multicast_addr, 0, sizeof(self->multicast_addr));
    self->multicast_addr.sin_family = AF_INET;
    self->multicast_addr.sin_addr.s_addr = inet_addr("239.255.255.250");
    self->multicast_addr.sin_port = htons(5000);

    // 设置TTL（可选）
    int ttl = 32;
    if (setsockopt(self->sockfd, IPPROTO_IP, IP_MULTICAST_TTL,
                   &ttl, sizeof(ttl)) < 0)
    {
        GST_WARNING("Failed to set multicast TTL");
    }

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/**
 * 当元素从上游元素接收到输入缓冲区时调用。
 */
static GstFlowReturn
gst_mynetwork_render(GstBaseSink *sink, GstBuffer *buf)
{
    Gstmynetwork *self = GST_MYNETWORK(sink);

    NvDsBatchMeta *batch_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvBufSurface *surface = NULL;
    GstMapInfo in_map_info;
    SendData send_data;
    float obj2center_distance = 9999;

    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ))
    {
        g_print("Error: Failed to map gst buffer\n");
        goto error;
    }
    nvds_set_input_system_timestamp(buf, GST_ELEMENT_NAME(self));
    surface = (NvBufSurface *)in_map_info.data;

    batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        frame_meta = (NvDsFrameMeta *)(l_frame->data);
        // 获取源分辨率
        guint source_width = frame_meta->source_frame_width;
        guint source_height = frame_meta->source_frame_height;
        // 计算视频中心点坐标
        float center_x = source_width / 2;
        float center_y = source_height / 2;

        NvOSD_RectParams rect_params;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if ((obj_meta->class_id >= 0))
            {
                // 计算目标的中心坐标
                float obj_center_x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2;
                float obj_center_y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2;
                // 计算目标与视频中心的距离
                float distance = fabs(obj_center_x - center_x) + fabs(obj_center_y - center_y);
                if (distance < obj2center_distance)
                {
                    obj2center_distance = distance;
                    memset(&send_data, 0, sizeof(send_data));
                    send_data.class_id = obj_meta->class_id;
                    send_data.confidence = obj_meta->confidence;
                    send_data.ntp_timestamp = frame_meta->ntp_timestamp;
                    send_data.source_id = frame_meta->source_id;
                    send_data.detect_info.left = obj_meta->rect_params.left;
                    send_data.detect_info.top = obj_meta->rect_params.top;
                    send_data.detect_info.width = obj_meta->rect_params.width;
                    send_data.detect_info.height = obj_meta->rect_params.height;
                }
            }
        }
        sendto(self->sockfd, &send_data, sizeof(send_data), 0,
               (struct sockaddr *)&self->multicast_addr, sizeof(self->multicast_addr));
    }
error:

    nvds_set_output_system_timestamp(buf, GST_ELEMENT_NAME(self));
    gst_buffer_unmap(buf, &in_map_info);
    return GST_FLOW_OK;
}

/**
 * 在元素从 ​READY​ 状态切换到 PLAYING/​PAUSED​ 状态时调用
 */
static gboolean
gst_mynetwork_start(GstBaseSink *sink)
{
    g_print("gst_mynetwork_start\n");
    Gstmynetwork *self = GST_MYNETWORK(sink);
    NvBufSurfaceCreateParams create_params = {0};

    CHECK_CUDA_STATUS(cudaSetDevice(self->gpu_id),
                      "Unable to set cuda device");
    return TRUE;
error:
    return FALSE;
}

/**
 * @brief 在元素从 PLAYING/​PAUSED 状态切换到 ​READY​​ 状态时调用
 *
 * @param trans 指向 GstBaseTransform 结构的指针。
 * @return 始终返回 TRUE。
 */
static gboolean
gst_mynetwork_stop(GstBaseSink *sink)
{
    g_print("gst_mynetwork_stop\n");
    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_mynetwork_set_caps(GstBaseSink *sink, GstCaps *caps)
{
    Gstmynetwork *dsmynetwork = GST_MYNETWORK(sink);

    return TRUE;

error:
    return FALSE;
}

void gst_mynetwork_set_property(GObject *object,
                                guint property_id,
                                const GValue *value,
                                GParamSpec *pspec)
{
    Gstmynetwork *self = GST_MYNETWORK(object);

    GST_DEBUG_OBJECT(self, "set_property");
}

void gst_mynetwork_get_property(GObject *object, guint property_id,
                                GValue *value, GParamSpec *pspec)
{
    Gstmynetwork *self = GST_MYNETWORK(object);

    GST_DEBUG_OBJECT(self, "get_property");
}

// 对象销毁前的清理回调函数
void gst_mynetwork_finalize(GObject *object)
{
    Gstmynetwork *self = GST_MYNETWORK(object);

    if (self->sockfd >= 0)
    {
        close(self->sockfd);
        self->sockfd = -1;
    }

    GST_DEBUG_OBJECT(self, "finalize");

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
_mynetwork_init(GstPlugin *plugin)
{
    /* debug category for filtering log messages
     *
     * exchange the string 'Template _mynetwork' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_mynetwork_debug,
                            "_mynetwork",
                            0,
                            "_mynetwork plugin");

    return gst_element_register(plugin,
                                "_mynetwork",
                                GST_RANK_PRIMARY,
                                GST_TYPE_MYNETWORK);
}

/* PACKAGE: this is usually set by meson depending on some _INIT macro
 * in meson.build and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use meson to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirst_mynetwork"
#endif

/* gstreamer looks for this structure to register _mynetworks
 *
 * exchange the string 'Template _mynetwork' with your _mynetwork description
 */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  _mynetwork,
                  DESCRIPTION,
                  _mynetwork_init,
                  "7.1",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)