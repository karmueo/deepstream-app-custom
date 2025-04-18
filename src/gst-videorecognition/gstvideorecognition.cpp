/**
 * SECTION:element-_videorecognition
 *
 * FIXME:Describe _videorecognition here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! _videorecognition ! fakesink silent=TRUE
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
#include "gstvideorecognition.h"
#include "cuda_runtime_api.h"
#include <math.h>
#include <map>
#include <cstdio>
#include "process.h"

/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
/* 启用将转换后的 cvmat 写入文件 */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

GST_DEBUG_CATEGORY_STATIC(gst_videorecognition_debug);
#define GST_CAT_DEFAULT gst_videorecognition_debug

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                                                           \
    ({                                                                                                         \
        int _errtype = 0;                                                                                      \
        do                                                                                                     \
        {                                                                                                      \
            if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) &&        \
                (surface->gpuId != object->gpu_id))                                                            \
            {                                                                                                  \
                GST_ELEMENT_ERROR(object, RESOURCE, FAILED,                                                    \
                                  ("Input surface gpu-id doesnt match with configured gpu-id for element,"     \
                                   " please allocate input using unified memory, or use same gpu-ids"),        \
                                  ("surface-gpu-id=%d,%s-gpu-id=%d", surface->gpuId, GST_ELEMENT_NAME(object), \
                                   object->gpu_id));                                                           \
                _errtype = 1;                                                                                  \
            }                                                                                                  \
        } while (0);                                                                                           \
        _errtype;                                                                                              \
    })

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
static GstStaticPadTemplate gst_videorecognition_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("ANY"));

static GstStaticPadTemplate gst_videorecognition_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("ANY"));

static void gst_videorecognition_set_property(GObject *object,
                                              guint property_id,
                                              const GValue *value,
                                              GParamSpec *pspec);
static void gst_videorecognition_get_property(GObject *object,
                                              guint property_id,
                                              GValue *value,
                                              GParamSpec *pspec);
static void gst_videorecognition_finalize(GObject *object);

#define gst_videorecognition_parent_class parent_class
G_DEFINE_TYPE(Gstvideorecognition, gst_videorecognition, GST_TYPE_BASE_TRANSFORM);

static gboolean gst_videorecognition_set_caps(GstBaseTransform *btrans,
                                              GstCaps *incaps, GstCaps *outcaps);

static gboolean gst_videorecognition_start(GstBaseTransform *btrans);
static gboolean gst_videorecognition_stop(GstBaseTransform *btrans);
static GstFlowReturn gst_videorecognition_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf);

/* GObject vmethod implementations */

/* initialize the _videorecognition's class */
static void
gst_videorecognition_class_init(GstvideorecognitionClass *klass)
{
    GObjectClass *gobject_class;
    GstElementClass *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;
    gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Overide base class functions */
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_videorecognition_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_videorecognition_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_videorecognition_finalize);

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_videorecognition_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_videorecognition_stop);
    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_videorecognition_set_caps);

    gstbasetransform_class->transform_ip =
        GST_DEBUG_FUNCPTR(gst_videorecognition_transform_ip);

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&gst_videorecognition_src_template));
    gst_element_class_add_pad_template(gstelement_class,
                                       gst_static_pad_template_get(&gst_videorecognition_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(gstelement_class,
                                         "DsVideoRecognition plugin",
                                         "DsVideoRecognition Plugin",
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
gst_videorecognition_init(Gstvideorecognition *self)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(self);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);

    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);
    // 初始化一些参数
    self->gpu_id = 0;
    /* Initialize all property variables to default values */
    self->unique_id = 15;
    self->gpu_id = 0;
    self->frame_num = 0;
    self->video_recognition = new tsnTrt("/workspace/deepstream-app-custom/triton_model/Video_Classify/1/end2end.engine", 224);

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_videorecognition_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    GstMapInfo in_map_info;
    GstFlowReturn flow_ret = GST_FLOW_ERROR;

    {
        Process process;
        process.testFunc("/workspace/deepstream-app-custom/arm_wrestling.mp4");
        std::vector<cv::Mat> frames = process.sampleFrames(250);
        // 确保frames每一帧类型为CV_32FC3
        /* if (frames[0].type() != CV_32FC3)
        {
            std::cerr << "Error: Frame type is not CV_32FC3" << std::endl;
            return GST_FLOW_ERROR;
        } */
        const int input_size = 1 * 250 * 3 * 224 * 224;
        float *input_data = new float[input_size];
        // 数据预处理
        process.convertCvInputToTensorRT(frames, input_data, 250, 224, 224);

        if (self->video_recognition)
        {
            tsnTrt *tsnPtr = dynamic_cast<tsnTrt *>(self->video_recognition);
            tsnPtr->prepare_input("input", 250, input_data);
            tsnPtr->prepare_output("output");
            tsnPtr->do_inference();
            float *output_data = new float[tsnPtr->GetOutputSize()];
            tsnPtr->get_output(output_data);
            RECOGNITION rec = tsnPtr->parse_output(output_data);
            std::cout << "Class ID: " << rec.class_id << ", Class Name: " << rec.class_name
                      << ", Score: " << rec.score << std::endl;
        }
        else
        {
            std::cerr << "Error: video_recognition is null" << std::endl;
        }
    }

    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;

    self->frame_num++;

    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ))
    {
        g_print("Error: Failed to map gst buffer\n");
        goto error;
    }

    nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(self));
    surface = (NvBufSurface *)in_map_info.data;

    GST_DEBUG_OBJECT(self,
                     "Processing Frame %" G_GUINT64_FORMAT " Surface %p\n",
                     self->frame_num, surface);

    if (CHECK_NVDS_MEMORY_AND_GPUID(self, surface))
        goto error;

    batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (batch_meta == nullptr)
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("NvDsBatchMeta not found for input buffer."), (NULL));
        return GST_FLOW_ERROR;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsMetaList *l_obj = NULL;
        NvDsObjectMeta *obj_meta = NULL;
        frame_meta = (NvDsFrameMeta *)(l_frame->data);

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
        }
    }

    flow_ret = GST_FLOW_OK;
error:

    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(self));
    gst_buffer_unmap(inbuf, &in_map_info);
    return flow_ret;
}

/**
 * 在元素从 ​READY​ 状态切换到 PLAYING/​PAUSED​ 状态时调用
 */
static gboolean
gst_videorecognition_start(GstBaseTransform *btrans)
{
    g_print("gst_videorecognition_start\n");
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
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
gst_videorecognition_stop(GstBaseTransform *btrans)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    g_print("gst_videorecognition_stop\n");
    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_videorecognition_set_caps(GstBaseTransform *btrans, GstCaps *incaps,
                              GstCaps *outcaps)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    /* Save the input video information, since this will be required later. */
    gst_video_info_from_caps(&self->video_info, incaps);

    return TRUE;

error:
    return FALSE;
}

void gst_videorecognition_set_property(GObject *object,
                                       guint property_id,
                                       const GValue *value,
                                       GParamSpec *pspec)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(object);

    GST_DEBUG_OBJECT(self, "set_property");
}

void gst_videorecognition_get_property(GObject *object, guint property_id,
                                       GValue *value, GParamSpec *pspec)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(object);

    GST_DEBUG_OBJECT(self, "get_property");
}

// 对象销毁前的清理回调函数
void gst_videorecognition_finalize(GObject *object)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(object);

    GST_DEBUG_OBJECT(self, "finalize");

    if (self->video_recognition)
    {
        delete self->video_recognition;
        self->video_recognition = NULL;
    }

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
videorecognition_init(GstPlugin *plugin)
{
    /* debug category for filtering log messages
     *
     * exchange the string 'Template _videorecognition' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_videorecognition_debug,
                            "videorecognition",
                            0,
                            "videorecognition plugin");

    return gst_element_register(plugin,
                                "videorecognition",
                                GST_RANK_PRIMARY,
                                GST_TYPE_VIDEORECOGNITION);
}

#ifndef PACKAGE
#define PACKAGE "videorecognition"
#endif
/* gstreamer looks for this structure to register videorecognitions
 *
 * exchange the string 'Template _videorecognition' with your _videorecognition description
 */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    _videorecognition,
    "Video recognition plugin",
    videorecognition_init,
    "7.1",
    LICENSE,
    BINARY_PACKAGE,
    URL)