/**
 * SECTION:element-_udpmulticast_sink
 *
 * FIXME:Describe _udpmulticast_sink here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! _udpmulticast_sink ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/gstinfo.h>
// #include "nvdsmeta.h"
#include "eo_protocol_parser.h"
#include "gstudpmulticast_sink.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include <gst/base/gstbasetransform.h>
#include <gst/gstelement.h>
#include <gst/gstinfo.h>

#include "cuda_runtime_api.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <math.h>

/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
/* 启用将转换后的 cvmat 写入文件 */
/* #define DSEXAMPLE_DEBUG */
static GQuark         _dsmeta_quark = 0;
static DetectAnalysis _detctAnalysis = {0};

GST_DEBUG_CATEGORY_STATIC(gst_udpmulticast_sink_debug);
#define GST_CAT_DEFAULT gst_udpmulticast_sink_debug

#define CHECK_CUDA_STATUS(cuda_status, error_str)                              \
    do                                                                         \
    {                                                                          \
        if ((cuda_status) != cudaSuccess)                                      \
        {                                                                      \
            g_print("Error: %s in %s at line %d (%s)\n", error_str, __FILE__,  \
                    __LINE__, cudaGetErrorName(cuda_status));                  \
            goto error;                                                        \
        }                                                                      \
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
    PROP_SILENT,
    PROP_IP,
    PROP_PORT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("ANY"));

static void gst_udpmulticast_sink_set_property(GObject      *object,
                                       guint         property_id,
                                       const GValue *value,
                                       GParamSpec   *pspec);
static void gst_udpmulticast_sink_get_property(GObject    *object,
                                       guint       property_id,
                                       GValue     *value,
                                       GParamSpec *pspec);
static void gst_udpmulticast_sink_finalize(GObject *object);

#define gst_udpmulticast_sink_parent_class parent_class
G_DEFINE_TYPE(Gstudpmulticast_sink, gst_udpmulticast_sink, GST_TYPE_BASE_SINK);

static gboolean gst_udpmulticast_sink_set_caps(GstBaseSink *sink, GstCaps *caps);

static GstFlowReturn gst_udpmulticast_sink_render(GstBaseSink *sink, GstBuffer *buf);
static gboolean      gst_udpmulticast_sink_start(GstBaseSink *sink);
static gboolean      gst_udpmulticast_sink_stop(GstBaseSink *sink);

/* GObject vmethod implementations */

/* initialize the _udpmulticast_sink's class */
static void gst_udpmulticast_sink_class_init(Gstudpmulticast_sinkClass *klass)
{
    GObjectClass     *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass  *gstelement_class;
    GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS(klass);

    gstelement_class = (GstElementClass *)klass;

    gobject_class->set_property = gst_udpmulticast_sink_set_property;
    gobject_class->get_property = gst_udpmulticast_sink_get_property;
    gobject_class->finalize = gst_udpmulticast_sink_finalize;

    base_sink_class->render = GST_DEBUG_FUNCPTR(gst_udpmulticast_sink_render);
    base_sink_class->start = GST_DEBUG_FUNCPTR(gst_udpmulticast_sink_start);
    base_sink_class->stop = GST_DEBUG_FUNCPTR(gst_udpmulticast_sink_stop);
    base_sink_class->set_caps = GST_DEBUG_FUNCPTR(gst_udpmulticast_sink_set_caps);

    gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                              &sink_factory);

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(
        gstelement_class, "DsUdpMulticastSink plugin", "DsUdpMulticastSink Plugin",
        "Process a infer mst network on objects / full frame",
        "ShenChangli "
        "@ karmueo@163.com");

    /* install properties */
    g_object_class_install_property(
        gobject_class, PROP_SILENT,
        g_param_spec_boolean("silent", "Silent", "Produce verbose output ?",
                             TRUE, G_PARAM_READWRITE));
    g_object_class_install_property(
        gobject_class, PROP_IP,
        g_param_spec_string(
            "ip", "Multicast IP", "Multicast destination IP", "239.255.255.250",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(
        gobject_class, PROP_PORT,
        g_param_spec_uint(
            "port", "Multicast Port", "Multicast destination port", 1, 65535,
            5000, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    _detctAnalysis.minPixel = 9999;
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
static void gst_udpmulticast_sink_init(Gstudpmulticast_sink *self)
{
    // 初始化一些参数
    self->gpu_id = 0;
    // default values
    self->ip = g_strdup("239.255.255.250");
    self->port = 5000;

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
    self->multicast_addr.sin_addr.s_addr = inet_addr(self->ip);
    self->multicast_addr.sin_port = htons(self->port);

    // 设置TTL（可选）
    int ttl = 32;
    if (setsockopt(self->sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl,
                   sizeof(ttl)) < 0)
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
static GstFlowReturn gst_udpmulticast_sink_render(GstBaseSink *sink, GstBuffer *buf)
{
    Gstudpmulticast_sink *self = GST_UDPMULTICAST_SINK(sink);

    NvDsBatchMeta  *batch_meta = NULL;
    NvDsMetaList   *l_frame = NULL;
    NvDsFrameMeta  *frame_meta = NULL;
    NvDsMetaList   *l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    NvBufSurface *surface = NULL;
    GstMapInfo    in_map_info;
    std::vector<EOTargetInfo> targetInfos;

    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ))
    {
        g_print("Error: Failed to map gst buffer\n");
        goto error;
    }
    nvds_set_input_system_timestamp(buf, GST_ELEMENT_NAME(self));
    surface = (NvBufSurface *)in_map_info.data;

    batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    // 记录每一帧的信息
    // 帧号
    // 目标大类和配置文件中的大类如果一致则+1
    // 目标小类和配置文件中的小类如果一致则+1
    // 目标最小像素数
    // 目标平均像素数
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        frame_meta = (NvDsFrameMeta *)(l_frame->data);
        _detctAnalysis.frameNum = frame_meta->frame_num + 1;
        // 获取源分辨率
        guint source_width = frame_meta->source_frame_width;
        guint source_height = frame_meta->source_frame_height;
        // 计算视频中心点坐标
        float center_x = (float)source_width / 2.0f;
        float center_y = (float)source_height / 2.0f;

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if ((obj_meta->class_id >= 0))
            {
                // 为每个检测目标构造并发送EOTargetInfo
                // 统计检测识别
                std::map<guint16, guint>::iterator it =
                    _detctAnalysis.primaryClassCountMap.find(obj_meta->class_id);
                if (it != _detctAnalysis.primaryClassCountMap.end())
                {
                    // 如果找到了，就+1
                    it->second++;
                }
                else
                {
                    // 如果没找到，就插入一个新的
                    _detctAnalysis.primaryClassCountMap.insert(
                        std::pair<guint16, guint>(obj_meta->class_id, 1));
                }

                // 处理二次分类，取第一个 classifier 的第一个 label
                // 作为输出（若存在）
                for (NvDsMetaList *l_class = obj_meta->classifier_meta_list;
                     l_class != NULL; l_class = l_class->next)
                {
                    NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *)l_class->data;
                    for (NvDsMetaList *l_label = cmeta->label_info_list;
                         l_label != NULL; l_label = l_label->next)
                    {
                        NvDsLabelInfo *label = (NvDsLabelInfo *)l_label->data;
                        // 统计计数
                        auto it2 = _detctAnalysis.secondaryClassCountMap.find(
                            label->result_class_id);
                        if (it2 != _detctAnalysis.secondaryClassCountMap.end())
                            it2->second++;
                        else
                            _detctAnalysis.secondaryClassCountMap.insert(
                                std::pair<guint16, guint>(label->result_class_id, 1));
                    }
                }

                // 记录最像素数和平均像素数
                guint16 pixel = obj_meta->rect_params.width * obj_meta->rect_params.height;
                if (pixel < _detctAnalysis.minPixel)
                {
                    _detctAnalysis.minPixel = pixel;
                }
                guint totalObjNum = _detctAnalysis.primaryClassCountMap.size();
                _detctAnalysis.meanPixel =
                    (_detctAnalysis.meanPixel * (totalObjNum - 1) + pixel) / totalObjNum;

                // 为当前目标创建EOTargetInfo并添加到向量中
                EOTargetInfo targetInfo;
                memset(&targetInfo, 0, sizeof(targetInfo));

                // 填充时间信息
                time_t     now = time(nullptr);
                struct tm *tm_info = localtime(&now);
                targetInfo.year = tm_info->tm_year + 1900;
                targetInfo.month = tm_info->tm_mon + 1;
                targetInfo.day = tm_info->tm_mday;
                targetInfo.hour = tm_info->tm_hour;
                targetInfo.minute = tm_info->tm_min;
                targetInfo.second = tm_info->tm_sec;
                targetInfo.msec = 0.0f;

                // 填充设备和目标信息
                targetInfo.deviceType = DeviceType::VISIBLE_LIGHT;
                targetInfo.targetID = obj_meta->object_id; // 使用目标ID
                targetInfo.targetStatus = TargetStatus::NORMAL;
                targetInfo.trackMode = TrackMode::DETECTION_TRACK;

                // 填充位置信息（示例值）
                // targetInfo.fovAngle = 45.0f;
                // targetInfo.longitude = 116.404;
                // targetInfo.latitude = 39.915;
                // targetInfo.altitude = 50.0;
                // targetInfo.fovHorizontal = 0.0f;
                // targetInfo.fovVertical = 0.0f;
                // targetInfo.enuAzimuth = 0.0f;
                // targetInfo.enuElevation = 0.0f;

                // 填充目标检测信息
                targetInfo.offsetHorizontal =
                    (int)(obj_meta->rect_params.left + obj_meta->rect_params.width / 2);
                targetInfo.offsetVertical =
                    (int)(obj_meta->rect_params.top + obj_meta->rect_params.height / 2);
                targetInfo.targetRect = (int)(obj_meta->rect_params.width * obj_meta->rect_params.height);

                // 映射目标类型
                switch (obj_meta->class_id)
                {
                case 0:
                    targetInfo.targetClass = TargetClass::SMALL_BIRD;
                    break;
                case 1:
                    targetInfo.targetClass = TargetClass::UAV;
                    break;
                default:
                    targetInfo.targetClass = TargetClass::UNKNOWN;
                    break;
                }

                targetInfo.targetConfidence = obj_meta->confidence;
                // targetInfo.targetDistance = 1000.0f;

                // 添加到目标列表
                targetInfos.push_back(targetInfo);
            }
        }

        // 如果有检测到的目标，打包并发送所有目标信息
        if (!targetInfos.empty())
        {
            static uint16_t sendCount = 0;
            sendCount++;
            std::vector<uint8_t> message =
                EOProtocolParser::PackEOTargetMessage(
                    targetInfos, sendCount,
                    0,                  // 发送方站号
                    SystemType::EO,     // 光电系统
                    0,                  // 接收方站号
                    SystemType::FUSION, // 融合系统
                    0,                  // 发送子系统
                    0                   // 接收子系统
                );

            // 发送打包后的报文
            if (!message.empty())
            {
                ssize_t sent =
                    sendto(self->sockfd, message.data(), message.size(), 0,
                           (struct sockaddr *)&self->multicast_addr,
                           sizeof(self->multicast_addr));
                if (sent < 0)
                {
                    GST_WARNING("Failed to send EO target message with %zu targets: %s",
                                targetInfos.size(), strerror(errno));
                }
                else
                {
                    GST_DEBUG("Successfully sent EO target message with %zu targets, size: %zu bytes",
                              targetInfos.size(), message.size());
                }
            }
            
            // 清空目标列表，为下一帧准备
            targetInfos.clear();
        }

        // 把_detctAnalysis写入日志
        // 为了更方便定位，添加标志
        GST_INFO("<===================================");
        GST_INFO("frameNum: %lu", _detctAnalysis.frameNum);
        for (std::map<guint16, guint>::iterator it =
                 _detctAnalysis.primaryClassCountMap.begin();
             it != _detctAnalysis.primaryClassCountMap.end(); it++)
        {
            GST_INFO("primaryClassCountMap: %d, %d", it->first, it->second);
        }
        for (std::map<guint16, guint>::iterator it =
                 _detctAnalysis.secondaryClassCountMap.begin();
             it != _detctAnalysis.secondaryClassCountMap.end(); it++)
        {
            GST_INFO("secondaryClassCountMap: %d, %d", it->first, it->second);
        }
        GST_INFO("minPixel: %d", _detctAnalysis.minPixel);
        GST_INFO("meanPixel: %d", _detctAnalysis.meanPixel);
        GST_INFO("===================================>");
    }
error:

    nvds_set_output_system_timestamp(buf, GST_ELEMENT_NAME(self));
    gst_buffer_unmap(buf, &in_map_info);
    return GST_FLOW_OK;
}

/**
 * 在元素从 ​READY​ 状态切换到 PLAYING/​PAUSED​ 状态时调用
 */
static gboolean gst_udpmulticast_sink_start(GstBaseSink *sink)
{
    g_print("gst_udpmulticast_sink_start\n");
    Gstudpmulticast_sink            *self = GST_UDPMULTICAST_SINK(sink);
    NvBufSurfaceCreateParams create_params = {0};

    CHECK_CUDA_STATUS(cudaSetDevice(self->gpu_id), "Unable to set cuda device");
    return TRUE;
error:
    return FALSE;
}

/**
 * @brief 在元素从 PLAYING/​PAUSED 状态切换到 ​READY​​
 * 状态时调用
 *
 * @param trans 指向 GstBaseTransform 结构的指针。
 * @return 始终返回 TRUE。
 */
static gboolean gst_udpmulticast_sink_stop(GstBaseSink *sink)
{
    g_print("gst_udpmulticast_sink_stop\n");
    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean gst_udpmulticast_sink_set_caps(GstBaseSink *sink, GstCaps *caps)
{
    Gstudpmulticast_sink *dsudpmulticast_sink = GST_UDPMULTICAST_SINK(sink);

    return TRUE;

error:
    return FALSE;
}

static void gst_udpmulticast_sink_set_property(GObject      *object,
                                       guint         property_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
    Gstudpmulticast_sink *self = GST_UDPMULTICAST_SINK(object);
    switch (property_id)
    {
    case PROP_SILENT:
        /* self->silent = g_value_get_boolean(value); */
        break;
    case PROP_IP:
        g_free(self->ip);
        self->ip = g_value_dup_string(value);
        if (self->ip)
            self->multicast_addr.sin_addr.s_addr = inet_addr(self->ip);
        break;
    case PROP_PORT:
        self->port = g_value_get_uint(value);
        self->multicast_addr.sin_port = htons(self->port);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    }
}

static void gst_udpmulticast_sink_get_property(GObject    *object,
                                       guint       property_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
    Gstudpmulticast_sink *self = GST_UDPMULTICAST_SINK(object);
    switch (property_id)
    {
    case PROP_SILENT:
        /* g_value_set_boolean(value, self->silent); */
        break;
    case PROP_IP:
        g_value_set_string(value, self->ip);
        break;
    case PROP_PORT:
        g_value_set_uint(value, self->port);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    }
}

static void gst_udpmulticast_sink_finalize(GObject *object)
{
    Gstudpmulticast_sink *self = GST_UDPMULTICAST_SINK(object);
    if (self->sockfd >= 0)
    {
        close(self->sockfd);
        self->sockfd = -1;
    }
    g_clear_pointer(&self->ip, g_free);
    GST_DEBUG_OBJECT(self, "finalize");
    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean _udpmulticast_sink_init(GstPlugin *plugin)
{
    /* debug category for filtering log messages
     *
     * exchange the string 'Template _udpmulticast_sink' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_udpmulticast_sink_debug, "udpmulticast_sink", 0,
                            "udpmulticast_sink plugin");

    return gst_element_register(plugin, "udpmulticast_sink", GST_RANK_PRIMARY,
                                GST_TYPE_UDPMULTICAST_SINK);
}

/* PACKAGE: this is usually set by meson depending on some _INIT macro
 * in meson.build and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use meson to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirst_udpmulticast_sink"
#endif

/* gstreamer looks for this structure to register _udpmulticast_sinks
 *
 * exchange the string 'Template udpmulticast_sink' with your _udpmulticast_sink description
 */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  udpmulticast_sink,
                  DESCRIPTION,
                  _udpmulticast_sink_init,
                  "7.1",
                  LICENSE,
                  BINARY_PACKAGE,
                  URL)