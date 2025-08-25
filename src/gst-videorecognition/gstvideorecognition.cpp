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
#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstvideorecognition.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <cstdio>
#include <gst/base/gstbasetransform.h>
#include <gst/gstelement.h>
#include <gst/gstinfo.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <math.h>
// #include "gstnvdsinfer.h"
#include <opencv2/opencv.hpp>

/* enable to write transformed cvmat to files */
/* #define DSEXAMPLE_DEBUG */
/* 启用将转换后的 cvmat 写入文件 */
/* #define DSEXAMPLE_DEBUG */
static GQuark _dsmeta_quark = 0;

GST_DEBUG_CATEGORY_STATIC(gst_videorecognition_debug);
#define GST_CAT_DEFAULT gst_videorecognition_debug

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)                           \
    ({                                                                         \
        int _errtype = 0;                                                      \
        do                                                                     \
        {                                                                      \
            if ((surface->memType == NVBUF_MEM_DEFAULT ||                      \
                 surface->memType == NVBUF_MEM_CUDA_DEVICE) &&                 \
                (surface->gpuId != object->gpu_id))                            \
            {                                                                  \
                GST_ELEMENT_ERROR(object, RESOURCE, FAILED,                    \
                                  ("Input surface gpu-id doesnt match with "   \
                                   "configured gpu-id for element,"            \
                                   " please allocate input using unified "     \
                                   "memory, or use same gpu-ids"),             \
                                  ("surface-gpu-id=%d,%s-gpu-id=%d",           \
                                   surface->gpuId, GST_ELEMENT_NAME(object),   \
                                   object->gpu_id));                           \
                _errtype = 1;                                                  \
            }                                                                  \
        } while (0);                                                           \
        _errtype;                                                              \
    })

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

// 缩放方式
enum GstVRScaleMode
{
    GST_VR_SCALE_ASPECT = 0, // 等比保持长宽比（可能带 padding）
    GST_VR_SCALE_STRETCH =
        1,                // 直接拉伸到目标尺寸（不保持长宽比，不加 padding）
    GST_VR_SCALE_NONE = 2 // 不缩放：直接按 crop_rect 提取原始大小 ROI
};

/* Filter signals and args */
enum
{
    /* FILL ME */
    LAST_SIGNAL
};

enum
{
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_GPU_DEVICE_ID,
    PROP_PROCESSING_WIDTH,
    PROP_PROCESSING_HEIGHT,
    PROP_MODEL_CLIP_LENGTH,
    PROP_NUM_CLIPS,
    PROP_TRT_ENGINE_NAME,
    PROP_MODEL_TYPE
};

#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 32
#define DEFAULT_PROCESSING_HEIGHT 32
#define DEFAULT_PROCESSING_MODEL_CLIP_LENGTH 8
#define DEFAULT_PROCESSING_NUM_CLIPS 4
#define DEFAULT_TRT_ENGINE_NAME                                                \
    "/workspace/deepstream-app-custom/src/gst-videorecognition/models/"        \
    "uniformerv2_e1_end2end_fp32.engine"
// 0 = multi-frame image classification (默认), 1 = video recognition (tsn /
// uniformer 等时序模型)
#define DEFAULT_MODEL_TYPE 0

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate gst_videorecognition_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("ANY"));

static GstStaticPadTemplate gst_videorecognition_src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("ANY"));

static void gst_videorecognition_set_property(GObject      *object,
                                              guint         property_id,
                                              const GValue *value,
                                              GParamSpec   *pspec);
static void gst_videorecognition_get_property(GObject *object,
                                              guint property_id, GValue *value,
                                              GParamSpec *pspec);
static void gst_videorecognition_finalize(GObject *object);

#define gst_videorecognition_parent_class parent_class
G_DEFINE_TYPE(Gstvideorecognition, gst_videorecognition,
              GST_TYPE_BASE_TRANSFORM);

static gboolean gst_videorecognition_set_caps(GstBaseTransform *btrans,
                                              GstCaps          *incaps,
                                              GstCaps          *outcaps);

static gboolean      gst_videorecognition_start(GstBaseTransform *btrans);
static gboolean      gst_videorecognition_stop(GstBaseTransform *btrans);
static GstFlowReturn gst_videorecognition_transform_ip(GstBaseTransform *btrans,
                                                       GstBuffer        *inbuf);

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding required by hardware. Previously converted RGBA to
 * RGB/BGR via OpenCV; now the intermediate buffer is allocated directly as RGB
 * so no extra color conversion is needed if downstream expects RGB.
 * 等比缩放或拉伸到处理分辨率；中间缓冲区直接为 RGB 三通道，不再进行 RGBA->BGR
 * 转换。
 */
/**
 * @brief 获取转换后的OpenCV Mat对象，并进行缩放和格式转换。
 *
 * 该函数从输入的NvBufSurface缓冲区中提取指定索引的图像区域（可选裁剪），
 * 按照目标处理分辨率进行缩放，同时保持或拉伸宽高比，直接得到 RGB (CV_8UC3)
 * 数据保存到 out_cvMat 中。
 *
 * @param[in]  self
 * 指向Gstvideorecognition实例的指针，包含处理参数和中间缓冲区。
 * @param[in]  input_buf           输入的NvBufSurface缓冲区指针。
 * @param[in]  idx                 输入缓冲区中要处理的surface索引。
 * @param[in]  crop_rect_params    指定裁剪区域的矩形参数（NvOSD_RectParams）。
 * @param[out] ratio               输出缩放比例，保持宽高比。
 * @param[in]  input_width         输入图像宽度。
 * @param[in]  input_height        输入图像高度。
 *
 * @return
 *   - GST_FLOW_OK      成功转换并生成Mat对象。
 *   - GST_FLOW_ERROR   转换或映射过程中发生错误。
 */
static GstFlowReturn
get_converted_mat(NvBufSurface *input_buf, Gstvideorecognition *self, gint idx,
                  NvOSD_RectParams *crop_rect_params, gdouble &ratio,
                  gint input_width, gint input_height,
                  GstVRScaleMode scale_mode, cv::Mat &out_cvMat)
{
    NvBufSurfTransform_Error       err;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams       transform_params;
    NvBufSurfTransformRect         src_rect;
    NvBufSurfTransformRect         dst_rect;
    NvBufSurface                   ip_surf;
    ip_surf = *input_buf;
    guint pad_x, pad_y;

    ip_surf.numFilled = ip_surf.batchSize = 1;
    ip_surf.surfaceList = &(input_buf->surfaceList[idx]);

    gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
    gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
    gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
    gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);

    guint dest_width = self->processing_width;
    guint dest_height = self->processing_height;
    if (scale_mode == GST_VR_SCALE_NONE)
    {
        // 不缩放：目标尺寸用对象裁剪尺寸
        dest_width = src_width;
        dest_height = src_height;
        ratio = 1.0;
    }
    else if (scale_mode == GST_VR_SCALE_ASPECT)
    {
        // 保持纵横比
        double hdest = self->processing_width * src_height / (double)src_width;
        double wdest = self->processing_height * src_width / (double)src_height;
        if (hdest <= self->processing_height)
        {
            dest_width = self->processing_width;
            dest_height = static_cast<guint>(hdest);
        }
        else
        {
            dest_width = static_cast<guint>(wdest);
            dest_height = self->processing_height;
        }
    }
    else
    {
        // 直接缩放填满，不保持纵横比
        dest_width = self->processing_width;
        dest_height = self->processing_height;
    }

    /* 为转换配置参数 */
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id = 0;
    transform_config_params.cuda_stream = self->cuda_stream;

    /* Set the transform session parameters for the conversions executed in this
     * thread. */
    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success)
    {
        GST_ELEMENT_ERROR(
            self, STREAM, FAILED,
            ("NvBufSurfTransformSetSessionParams failed with error %d", err),
            (NULL));
        goto error;
    }

    /* 计算缩放比率。等比时为实际等比系数；拉伸时给出最小系数供外部参考 */
    ratio = MIN(1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0))
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__),
                          (NULL));
        goto error;
    }

    /* 为src和dst设置ROI */
    src_rect = {(guint)src_top, (guint)src_left, (guint)src_width,
                (guint)src_height};

    if (scale_mode == GST_VR_SCALE_ASPECT)
    {
        // 计算上下左右 padding，使 dst_rect 居中
        pad_x = (self->processing_width > dest_width
                     ? (self->processing_width - dest_width) / 2
                     : 0);
        pad_y = (self->processing_height > dest_height
                     ? (self->processing_height - dest_height) / 2
                     : 0);
        dst_rect = {pad_y, pad_x, (guint)dest_width, (guint)dest_height};
    }
    else if (scale_mode == GST_VR_SCALE_NONE)
    {
        // 不缩放：目标缓冲区即对象尺寸，无 padding
        pad_x = pad_y = 0;
        dst_rect = {0u, 0u, (guint)dest_width, (guint)dest_height};
    }
    else
    {
        // 无 padding，直接填满
        pad_x = pad_y = 0;
        dst_rect = {0u, 0u, (guint)dest_width, (guint)dest_height};
    }

    /* Set the transform parameters */
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                      NVBUFSURF_TRANSFORM_CROP_SRC |
                                      NVBUFSURF_TRANSFORM_CROP_DST;
    transform_params.transform_filter = NvBufSurfTransformInter_Bilinear;

    // 若 NONE 模式且对象尺寸与当前 inter_buf 不同，需要重新分配
    if (scale_mode == GST_VR_SCALE_NONE &&
        (self->inter_buf->surfaceList[0].width != (int)dest_width ||
         self->inter_buf->surfaceList[0].height != (int)dest_height))
    {
        NvBufSurfaceDestroy(self->inter_buf);
        NvBufSurfaceCreateParams create_params = {0};
        create_params.gpuId = self->gpu_id;
        create_params.width = dest_width;
        create_params.height = dest_height;
        create_params.size = 0;
        create_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_CUDA_PINNED;
        if (NvBufSurfaceCreate(&self->inter_buf, 1, &create_params) != 0)
        {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                              ("Recreate inter_buf failed"), (NULL));
            goto error;
        }
    }
    else
    {
        /* Memset the memory */
        NvBufSurfaceMemSet(self->inter_buf, 0, 0, 0);
    }

    GST_DEBUG_OBJECT(self, "Scaling and converting input buffer\n");

    /* 转换缩放+格式转换（如果有）。 */
    err = NvBufSurfTransform(&ip_surf, self->inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success)
    {
        GST_ELEMENT_ERROR(
            self, STREAM, FAILED,
            ("NvBufSurfTransform failed with error %d while converting buffer",
             err),
            (NULL));
        goto error;
    }
    /* 映射缓冲区，以便CPU访问 */
    if (NvBufSurfaceMap(self->inter_buf, 0, 0, NVBUF_MAP_READ) != 0)
    {
        goto error;
    }
    if (self->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY)
    {
        /* 缓存映射的数据以访问CPU */
        NvBufSurfaceSyncForCpu(self->inter_buf, 0, 0);
    }

    /* 直接映射为RGB三通道 (CV_8UC3)，不再使用 CV_8UC4 和额外颜色转换。*/
    if (out_cvMat.empty())
    {
        out_cvMat = cv::Mat(dest_height, dest_width, CV_8UC3,
                            self->inter_buf->surfaceList[0].mappedAddr.addr[0],
                            self->inter_buf->surfaceList[0].pitch);
    }
    else
    {
        out_cvMat.release();
        out_cvMat = cv::Mat(dest_height, dest_width, CV_8UC3,
                            self->inter_buf->surfaceList[0].mappedAddr.addr[0],
                            self->inter_buf->surfaceList[0].pitch);
    }

    if (NvBufSurfaceUnMap(self->inter_buf, 0, 0))
    {
        goto error;
    }

    return GST_FLOW_OK;

error:
    return GST_FLOW_ERROR;
}

/* GObject vmethod implementations */

/* initialize the _videorecognition's class */
static void gst_videorecognition_class_init(GstvideorecognitionClass *klass)
{
    GObjectClass          *gobject_class;
    GstElementClass       *gstelement_class;
    GstBaseTransformClass *gstbasetransform_class;

    gobject_class = (GObjectClass *)klass;
    gstelement_class = (GstElementClass *)klass;
    gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Overide base class functions */
    gobject_class->set_property =
        GST_DEBUG_FUNCPTR(gst_videorecognition_set_property);
    gobject_class->get_property =
        GST_DEBUG_FUNCPTR(gst_videorecognition_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_videorecognition_finalize);

    gstbasetransform_class->start =
        GST_DEBUG_FUNCPTR(gst_videorecognition_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_videorecognition_stop);
    gstbasetransform_class->set_caps =
        GST_DEBUG_FUNCPTR(gst_videorecognition_set_caps);

    gstbasetransform_class->transform_ip =
        GST_DEBUG_FUNCPTR(gst_videorecognition_transform_ip);

    /* Install properties */
    g_object_class_install_property(
        gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint(
            "unique-id", "Unique ID",
            "Unique ID for the element. Can be used to identify output of the "
            "element",
            0, G_MAXUINT, DEFAULT_UNIQUE_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint(
            "gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0, G_MAXUINT, 0,
            GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                        GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(
        gobject_class, PROP_PROCESSING_WIDTH,
        g_param_spec_int(
            "processing-width", "Processing Width",
            "Width of the input buffer to algorithm", 1, G_MAXINT,
            DEFAULT_PROCESSING_WIDTH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_PROCESSING_HEIGHT,
        g_param_spec_int(
            "processing-height", "Processing Height",
            "Height of the input buffer to algorithm", 1, G_MAXINT,
            DEFAULT_PROCESSING_HEIGHT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_MODEL_CLIP_LENGTH,
        g_param_spec_int(
            "model-clip-length", "Model Clip Length",
            "Length of the clip used by the model", 1, G_MAXINT,
            DEFAULT_PROCESSING_MODEL_CLIP_LENGTH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_NUM_CLIPS,
        g_param_spec_int(
            "num-clips", "Number of Clips", "Number of clips used by the model",
            1, G_MAXINT, DEFAULT_PROCESSING_NUM_CLIPS,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_TRT_ENGINE_NAME,
        g_param_spec_string(
            "trt-engine-name", "TensorRT Engine Name",
            "Name of the TensorRT engine file to use for inference",
            DEFAULT_TRT_ENGINE_NAME,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_MODEL_TYPE,
        g_param_spec_int(
            "model-type", "Model Type",
            "0=multi-frame image classification (object crop windowed); "
            "1=video recognition temporal clips",
            0, 1, DEFAULT_MODEL_TYPE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    /* Set sink and src pad capabilities */
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_videorecognition_src_template));
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_videorecognition_sink_template));

    /* Set metadata describing the element */
    gst_element_class_set_details_simple(
        gstelement_class, "DsVideoRecognition plugin",
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
static void gst_videorecognition_init(Gstvideorecognition *self)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(self);

    /* We will not be generating a new buffer. Just adding / updating
     * metadata. */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);

    /* We do not want to change the input caps. Set to passthrough. transform_ip
     * is still called. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

    // 初始化一些参数，不要写死，写配置
    self->gpu_id = PROP_GPU_DEVICE_ID;
    /* Initialize all property variables to default values */
    self->unique_id = DEFAULT_UNIQUE_ID;
    self->frame_num = 0;
    self->processing_width = DEFAULT_PROCESSING_WIDTH;
    self->processing_height = DEFAULT_PROCESSING_HEIGHT;
    // self->processing_frame_interval = 5;
    self->processing_frame_interval = 1;
    // 根据模型选择num_clips和clip_length和processing_width
    // self->model_num_clips = 1;
    self->model_num_clips = DEFAULT_PROCESSING_NUM_CLIPS;
    // self->model_clip_length = 32;
    self->model_clip_length = DEFAULT_PROCESSING_MODEL_CLIP_LENGTH;
    self->max_history_frames = self->processing_frame_interval *
                                   self->model_clip_length *
                                   self->model_num_clips +
                               self->model_num_clips * 2;
    self->trtProcessPtr = new Process(self->max_history_frames);
    const char *trt_engine_file = DEFAULT_TRT_ENGINE_NAME;
    strncpy(self->trt_engine_name, trt_engine_file,
            sizeof(self->trt_engine_name) - 1);
    self->trt_engine_name[sizeof(self->trt_engine_name) - 1] = '\0';
    self->recognitionResultPtr = new RECOGNITION();
    self->recognitionResultPtr->class_id = -1;
    self->recognitionResultPtr->class_name.clear();
    self->recognitionResultPtr->score = 0.f;
    self->model_type = DEFAULT_MODEL_TYPE;
    self->frame_classifier =
        nullptr; // 延迟到 set_property 根据 model-type 初始化
    self->video_recognition = nullptr;
    memset(self->frame_cls_scores, 0, sizeof(self->frame_cls_scores));
    // 初始化单目标分类窗口参数
    self->cls_window_size = 10; // 保底
    self->cls_window_index = 0;
    self->cls_window_count = 0;
    for (int i = 0; i < 10; ++i)
    {
        for (int c = 0; c < 3; ++c)
            self->cls_window_scores[i][c] = 0.f;
    }

    /* This quark is required to identify NvDsMeta when iterating through
     * the buffer metadatas */
    if (!_dsmeta_quark)
        _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn gst_videorecognition_transform_ip(GstBaseTransform *btrans,
                                                       GstBuffer        *inbuf)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    GstMapInfo           in_map_info;
    GstFlowReturn        flow_ret = GST_FLOW_ERROR;
    gdouble              scale_ratio = 1.0;

    NvBufSurface  *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList  *l_frame = NULL;

    self->frame_num++;

    memset(&in_map_info, 0, sizeof(in_map_info));
    // 把inbuf映射到in_map_info
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
                          ("NvDsBatchMeta not found for input buffer."),
                          (NULL));
        return GST_FLOW_ERROR;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        // 如果是视频识别模式(model_type==1)，在处理对象前先对整帧做缓存收集
        if (self->model_type == 1)
        {
            NvDsFrameMeta   *v_frame_meta = (NvDsFrameMeta *)(l_frame->data);
            NvOSD_RectParams full_rect_params;
            full_rect_params.left = 0;
            full_rect_params.top = 0;
            full_rect_params.width =
                surface->surfaceList[v_frame_meta->batch_id].width;
            full_rect_params.height =
                surface->surfaceList[v_frame_meta->batch_id].height;
            full_rect_params.border_width = 0;
            full_rect_params.border_color = (NvOSD_ColorParams){0};
            cv::Mat full_frame_mat;
            if (get_converted_mat(
                    surface, self, v_frame_meta->batch_id, &full_rect_params,
                    scale_ratio,
                    surface->surfaceList[v_frame_meta->batch_id].width,
                    surface->surfaceList[v_frame_meta->batch_id].height,
                    GST_VR_SCALE_STRETCH, full_frame_mat) == GST_FLOW_OK &&
                !full_frame_mat.empty())
            {
                // 推入帧缓冲
                if (self->trtProcessPtr)
                {
                    self->trtProcessPtr->addFrame(full_frame_mat);
                    // 满足长度后做一次推理
                    if (self->trtProcessPtr->getCurrentFrameLength() ==
                            (int)self->max_history_frames &&
                        self->video_recognition)
                    {
                        std::vector<float> input_data;
                        // 采样并生成 (Nclips, C, T, H, W) 展平后的输入
                        // (当前实现为 NTCHW 展平)
                        self->trtProcessPtr->convertCvInputToNtchwTensorRT(
                            input_data, self->model_num_clips,
                            self->model_clip_length, self->processing_height,
                            self->processing_width,
                            self->processing_frame_interval);
                        tsnTrt *tsnPtr =
                            dynamic_cast<tsnTrt *>(self->video_recognition);
                        if (tsnPtr)
                        {
                            // 仅第一次调用时 prepare_output
                            bool ok_prepare_output = true;
                            if (tsnPtr->GetOutputSize() == 0)
                                ok_prepare_output =
                                    tsnPtr->prepare_output("/Softmax_output_0");
                            if (tsnPtr->prepare_input("input",
                                                      self->model_num_clips,
                                                      self->model_clip_length,
                                                      input_data.data()) &&
                                ok_prepare_output)
                            {
                                if (tsnPtr->do_inference())
                                {
                                    float *output_data =
                                        new float[tsnPtr->GetOutputSize() /
                                                  sizeof(float)];
                                    tsnPtr->get_output(output_data);
                                    RECOGNITION result =
                                        tsnPtr->parse_output(output_data);
                                    self->recognitionResultPtr->class_id =
                                        result.class_id;
                                    self->recognitionResultPtr->class_name =
                                        result.class_name;
                                    self->recognitionResultPtr->score =
                                        result.score;
                                    delete[] output_data;
                                }
                            }
                        }
                        self->trtProcessPtr->clearFrames();
                    }
                }
            }
        }
        // 单目标，不需要历史 map 清理
        NvDsMetaList   *l_obj = NULL;
        NvDsObjectMeta *obj_meta = NULL;
        frame_meta = (NvDsFrameMeta *)(l_frame->data);
        if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] ==
            NULL)
        {
            if (NvBufSurfaceMap(surface, frame_meta->batch_id, 0,
                                NVBUF_MAP_READ_WRITE) != 0)
            {
                GST_ELEMENT_ERROR(
                    self, STREAM, FAILED,
                    ("%s:buffer map to be accessed by CPU failed", __func__),
                    (NULL));
                return GST_FLOW_ERROR;
            }
        }

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            cv::Mat target_img;
            /* Scale and convert the frame */
            if (get_converted_mat(
                    surface, self, frame_meta->batch_id, &obj_meta->rect_params,
                    scale_ratio,
                    surface->surfaceList[frame_meta->batch_id].width,
                    surface->surfaceList[frame_meta->batch_id].height,
                    GST_VR_SCALE_STRETCH, // 可改为 GST_VR_SCALE_STRETCH
                                          // 以开启拉伸模式
                    target_img) != GST_FLOW_OK)
            {
                GST_ELEMENT_ERROR(self, STREAM, FAILED,
                                  ("get_converted_mat failed"), (NULL));
                goto error;
            }

            // model-type==0: 对对象裁剪图执行单帧分类 + 多帧加权融合
            if (self->model_type == 0 && self->frame_classifier &&
                !target_img.empty())
            {
                cv::Mat resized;
                // target_img 现在已经是 RGB; 若尺寸不同再调整
                if (target_img.cols != 32 || target_img.rows != 32)
                {
                    cv::resize(target_img, resized, cv::Size(32, 32), 0, 0,
                               cv::INTER_LINEAR);
                }
                else
                {
                    resized = target_img;
                }
                cv::Mat &rgb = resized; // 直接使用 RGB
                // 把图片保存下来
                // cv::imwrite("/workspace/deepstream-app-custom/src/gst-videorecognition/build/target_img2.jpg", rgb);
                std::vector<float> input(3 * 32 * 32);
                int                hw = 32 * 32;
                for (int y = 0; y < 32; ++y)
                {
                    const unsigned char *row = rgb.ptr<unsigned char>(y);
                    for (int x = 0; x < 32; ++x)
                    {
                        int pos = y * 32 + x;
                        int base = x * 3;
                        input[0 * hw + pos] = row[base + 0] / 255.0f;
                        input[1 * hw + pos] = row[base + 1] / 255.0f;
                        input[2 * hw + pos] = row[base + 2] / 255.0f;
                    }
                }
                float output[3] = {0};
                if (self->frame_classifier->infer(input.data(), output))
                {
                    const char *names[3] = {"未知", "鸟", "无人机"};
                    // 运行期保护，避免为0
                    if (self->cls_window_size <= 0 ||
                        self->cls_window_size > 1000)
                    {
                        self->cls_window_size = 10;
                    }
                    // 写入环形缓冲
                    int idx = self->cls_window_index % self->cls_window_size;
                    for (int c = 0; c < 3; ++c)
                        self->cls_window_scores[idx][c] = output[c];
                    self->cls_window_index++;
                    if (self->cls_window_count < self->cls_window_size)
                        self->cls_window_count++;
                    // 计算窗口内平均分:
                    // 若未填满只统计已有；填满后统计全部环形槽
                    int   usable = self->cls_window_count;
                    float avg[3] = {0.f, 0.f, 0.f};
                    if (usable == self->cls_window_size)
                    {
                        // 已完整覆盖，直接累加所有槽
                        for (int i = 0; i < self->cls_window_size; ++i)
                        {
                            for (int c = 0; c < 3; ++c)
                                avg[c] += self->cls_window_scores[i][c];
                        }
                    }
                    else
                    {
                        // 部分填充：槽按顺序 0..usable-1
                        for (int i = 0; i < usable; ++i)
                        {
                            for (int c = 0; c < 3; ++c)
                                avg[c] += self->cls_window_scores[i][c];
                        }
                    }
                    for (int c = 0; c < 3; ++c)
                        avg[c] /= (float)usable;
                    int best = 0;
                    for (int c = 1; c < 3; ++c)
                        if (avg[c] > avg[best])
                            best = c;
                    float               best_prob = avg[best];
                    NvDsClassifierMeta *cls_meta =
                        nvds_acquire_classifier_meta_from_pool(batch_meta);
                    cls_meta->unique_component_id =
                        100; // window accumulation result
                    NvDsLabelInfo *li =
                        nvds_acquire_label_info_meta_from_pool(batch_meta);
                    li->result_class_id = best;
                    li->result_prob = best_prob;
                    snprintf(li->result_label, MAX_LABEL_SIZE, "%s(%.2f)W%d",
                             names[best], best_prob, self->cls_window_count);
                    li->result_label[MAX_LABEL_SIZE - 1] = '\0';
                    nvds_add_label_info_meta_to_classifier(cls_meta, li);
                    nvds_add_classifier_meta_to_object(obj_meta, cls_meta);
                }
            }

            // model-type==1: 使用时序视频识别结果（当达到阈值）
            if (self->model_type == 1)
            {
                self->trtProcessPtr->addFrame(target_img);

                // 多帧推理
                if (self->trtProcessPtr->getCurrentFrameLength() ==
                    self->max_history_frames)
                {
                    std::vector<float> input_data;
                    // 数据预处理
                    /* self->trtProcessPtr->convertCvInputToTensorRT(
                        input_data,
                        self->model_clip_length,
                        self->processing_width,
                        self->processing_height,
                        self->processing_frame_interval); */
                    self->trtProcessPtr->convertCvInputToNtchwTensorRT(
                        input_data, self->model_num_clips,
                        self->model_clip_length, self->processing_width,
                        self->processing_height,
                        self->processing_frame_interval);
                    /* self->trtProcessPtr->loadImagesFromDirectory2(
                        "/workspace/deepstream-app-custom/src/deepstream-app/110_video_frames/bird/bird_1/0/",
                        input_data,
                        self->model_num_clips,
                        self->model_clip_length,
                        self->processing_width,
                        self->processing_height); */
                    /* self->trtProcessPtr->loadImagesFromDirectory(
                        "/workspace/deepstream-app-custom/src/deepstream-app/110_video_frames/bird/bird_1/0/",
                        input_data,
                        self->model_clip_length,
                        self->processing_width,
                        self->processing_height,
                        self->processing_frame_interval); */

                    if (self->video_recognition)
                    {
                        tsnTrt *tsnPtr =
                            dynamic_cast<tsnTrt *>(self->video_recognition);
                        tsnPtr->prepare_input("input", self->model_num_clips,
                                              self->model_clip_length,
                                              input_data.data());
                        tsnPtr->prepare_output("/Softmax_output_0");
                        tsnPtr->do_inference();
                        float *output_data = new float[tsnPtr->GetOutputSize()];
                        tsnPtr->get_output(output_data);
                        RECOGNITION result = tsnPtr->parse_output(
                            output_data); // result是一个RECOGNITION对象
                        self->recognitionResultPtr->class_id = result.class_id;
                        self->recognitionResultPtr->class_name =
                            result.class_name;
                        self->recognitionResultPtr->score = result.score;

                        delete[] output_data;
                    }
                    else
                    {
                        std::cerr << "Error: video_recognition is null"
                                  << std::endl;
                    }
                    self->trtProcessPtr->clearFrames();
                }

                // 将视频识别结果写入元数据（与图片分类类似）
                if (self->recognitionResultPtr->score >= 0.5)
                {
                    NvDsClassifierMeta *classifier_meta =
                        nvds_acquire_classifier_meta_from_pool(batch_meta);
                    classifier_meta->unique_component_id = 9; // video recognition component id
                    NvDsLabelInfo *label_info =
                        nvds_acquire_label_info_meta_from_pool(batch_meta);
                    label_info->result_class_id =
                        self->recognitionResultPtr->class_id;
                    label_info->result_prob =
                        self->recognitionResultPtr->score;
                    // 简单类别名映射（可改成根据 self->recognitionResultPtr->class_name ）
                    if (label_info->result_class_id == 0)
                    {
                        strncpy(label_info->result_label, "鸟",
                                MAX_LABEL_SIZE - 1);
                    }
                    else if (label_info->result_class_id == 1)
                    {
                        strncpy(label_info->result_label, "无人机",
                                MAX_LABEL_SIZE - 1);
                    }
                    else
                    {
                        strncpy(label_info->result_label, "unknown",
                                MAX_LABEL_SIZE - 1);
                    }
                    label_info->result_label[MAX_LABEL_SIZE - 1] = '\0';
                    nvds_add_label_info_meta_to_classifier(classifier_meta,
                                                           label_info);
                    nvds_add_classifier_meta_to_object(obj_meta,
                                                       classifier_meta);
                }
            }
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
static gboolean gst_videorecognition_start(GstBaseTransform *btrans)
{
    g_print("gst_videorecognition_start\n");
    Gstvideorecognition     *self = GST_VIDEORECOGNITION(btrans);
    NvBufSurfaceCreateParams create_params = {0};

    CHECK_CUDA_STATUS(cudaSetDevice(self->gpu_id), "Unable to set cuda device");

    /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
     * required. Can be skipped if custom algorithm can work directly on
     * NV12/RGBA. */
    create_params.gpuId = self->gpu_id;
    create_params.width = self->processing_width;
    create_params.height = self->processing_height;
    create_params.size = 0;
    // 使用 RGB 三通道中间缓冲区
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
    if (NvBufSurfaceCreate(&self->inter_buf, 1, &create_params) != 0)
    {
        GST_ERROR("Error: Could not allocate internal buffer for dsexample");
        goto error;
    }
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
static gboolean gst_videorecognition_stop(GstBaseTransform *btrans)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    g_print("gst_videorecognition_stop\n");
    return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean gst_videorecognition_set_caps(GstBaseTransform *btrans,
                                              GstCaps *incaps, GstCaps *outcaps)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(btrans);
    /* Save the input video information, since this will be required later. */
    gst_video_info_from_caps(&self->video_info, incaps);

    return TRUE;

error:
    return FALSE;
}

void gst_videorecognition_set_property(GObject *object, guint property_id,
                                       const GValue *value, GParamSpec *pspec)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(object);
    GST_LOG_OBJECT(self, "set_property id=%u (%s)", property_id,
                   pspec ? pspec->name : "?");

    switch (property_id)
    {
    case PROP_UNIQUE_ID:
        self->unique_id = g_value_get_uint(value);
        break;
    case PROP_GPU_DEVICE_ID:
        self->gpu_id = g_value_get_uint(value);
        break;
    case PROP_PROCESSING_WIDTH:
        self->processing_width = g_value_get_int(value);
        if (self->processing_width <= 0)
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Processing width must be greater than 0"),
                              (NULL));
            return;
        }
        break;
    case PROP_PROCESSING_HEIGHT:
        self->processing_height = g_value_get_int(value);
        if (self->processing_height <= 0)
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Processing height must be greater than 0"),
                              (NULL));
            return;
        }
        break;
    case PROP_MODEL_CLIP_LENGTH:
        self->model_clip_length = g_value_get_int(value);
        if (self->model_clip_length <= 0 || self->model_clip_length > 128)
        {
            self->model_clip_length = DEFAULT_PROCESSING_MODEL_CLIP_LENGTH;
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Model clip length must be greater than 0 and "
                               "less than or equal to 128"),
                              (NULL));

            return;
        }
        break;
    case PROP_NUM_CLIPS:
        self->model_num_clips = g_value_get_int(value);
        if (self->model_num_clips <= 0)
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("num_clips must be greater than 0"), (NULL));
            return;
        }
        // 更新最大历史帧数
        self->max_history_frames = self->processing_frame_interval *
                                       self->model_clip_length *
                                       self->model_num_clips +
                                   self->model_num_clips * 2;
        if (self->trtProcessPtr)
        {
            delete self->trtProcessPtr;
            self->trtProcessPtr = new Process(self->max_history_frames);
        }
        break;
    case PROP_TRT_ENGINE_NAME:
    {
        const gchar *s = g_value_get_string(value);
        const gchar *engine = s && *s ? s : DEFAULT_TRT_ENGINE_NAME;
        strncpy(self->trt_engine_name, engine,
                sizeof(self->trt_engine_name) - 1);
        self->trt_engine_name[sizeof(self->trt_engine_name) - 1] = '\0';
        // 先清理现有引擎资源
        if (self->frame_classifier)
        {
            delete self->frame_classifier;
            self->frame_classifier = nullptr;
        }
        if (self->video_recognition)
        {
            delete self->video_recognition;
            self->video_recognition = nullptr;
        }
        break;
    }
    case PROP_MODEL_TYPE:
    {
        gint mt = g_value_get_int(value);
        GST_INFO_OBJECT(self, "PROP_MODEL_TYPE incoming=%d (current=%d)", mt,
                        self->model_type);
        if (mt == 0)
        {
            self->model_type = 0;
            // 切换到分类模式：确保分类器存在
            if (!self->frame_classifier)
            {
                self->frame_classifier = new ImageClsTrt(self->trt_engine_name);
                if (!self->frame_classifier->prepare())
                {
                    GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                                      ("frame classifier prepare failed for %s",
                                       self->trt_engine_name),
                                      (NULL));
                }
            }
            if (self->video_recognition)
            {
                delete self->video_recognition;
                self->video_recognition = nullptr;
            }
            GST_INFO_OBJECT(self,
                            "Switched to model-type=0 (image classification)");
        }
        else if (mt == 1)
        {
            self->model_type = 1;
            // 切换到视频识别模式：释放分类器 (若以后需要重新启用)
            if (self->frame_classifier)
            {
                delete self->frame_classifier;
                self->frame_classifier = nullptr;
            }
            if (!self->video_recognition)
            {
                self->video_recognition =
                    new tsnTrt(self->trt_engine_name, self->processing_width);
                GST_INFO_OBJECT(self,
                                "Initialized video recognition engine with %s",
                                self->trt_engine_name);
            }
            GST_INFO_OBJECT(
                self,
                "Switched to model-type=1 (video recognition temporal clips)");
        }
        else
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Invalid model-type %d (expected 0 or 1)", mt),
                              (NULL));
        }
        break;
    }

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
}

void gst_videorecognition_get_property(GObject *object, guint property_id,
                                       GValue *value, GParamSpec *pspec)
{
    Gstvideorecognition *self = GST_VIDEORECOGNITION(object);

    switch (property_id)
    {
    case PROP_UNIQUE_ID:
        g_value_set_uint(value, self->unique_id);
        break;
    case PROP_GPU_DEVICE_ID:
        g_value_set_uint(value, self->gpu_id);
        break;
    case PROP_PROCESSING_WIDTH:
        g_value_set_int(value, self->processing_width);
        break;
    case PROP_PROCESSING_HEIGHT:
        g_value_set_int(value, self->processing_height);
        break;
    case PROP_MODEL_CLIP_LENGTH:
        g_value_set_int(value, self->model_clip_length);
        break;
    case PROP_MODEL_TYPE:
    {
        g_value_set_int(value, self->model_type);
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
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

    if (self->recognitionResultPtr)
    {
        delete self->recognitionResultPtr;
        self->recognitionResultPtr = NULL;
    }
    if (self->frame_classifier)
    {
        delete self->frame_classifier;
        self->frame_classifier = NULL;
    }

    if (self->inter_buf)
    {
        NvBufSurfaceDestroy(self->inter_buf);
        self->inter_buf = NULL;
    }
    if (self->cuda_stream)
    {
        cudaStreamDestroy(self->cuda_stream);
        self->cuda_stream = NULL;
    }
    if (self->trtProcessPtr)
    {
        delete self->trtProcessPtr;
        self->trtProcessPtr = NULL;
    }

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean videorecognition_init(GstPlugin *plugin)
{
    /* debug category for filtering log messages
     *
     * exchange the string 'Template _videorecognition' with your description
     */
    GST_DEBUG_CATEGORY_INIT(gst_videorecognition_debug, "videorecognition", 0,
                            "videorecognition plugin");

    return gst_element_register(plugin, "videorecognition", GST_RANK_PRIMARY,
                                GST_TYPE_VIDEORECOGNITION);
}

#ifndef PACKAGE
#define PACKAGE "videorecognition"
#endif
/* gstreamer looks for this structure to register videorecognitions
 *
 * exchange the string 'Template _videorecognition' with your _videorecognition
 * description
 */
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, _videorecognition,
                  "Video recognition plugin", videorecognition_init, "7.1",
                  LICENSE, BINARY_PACKAGE, URL)