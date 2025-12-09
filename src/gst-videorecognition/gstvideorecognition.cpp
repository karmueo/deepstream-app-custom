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
#include "x3dTrt.h"
#include <cstdio>
#include <gst/base/gstbasetransform.h>
#include <gst/gstelement.h>
#include <gst/gstinfo.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <map>

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
    PROP_SAMPLING_RATE,
    PROP_TRT_ENGINE_NAME,
    PROP_LABELS_FILE
};

#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 64
#define DEFAULT_PROCESSING_HEIGHT 64
#define DEFAULT_PROCESSING_MODEL_CLIP_LENGTH 32
#define DEFAULT_SAMPLING_RATE 5
#define DEFAULT_TRT_ENGINE_NAME                                                \
    "/workspace/deepstream-app-custom/src/gst-videorecognition/models/"        \
    "x3d.engine"
#define DEFAULT_LABELS_FILE "labels.txt"

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
 * @brief 获取转换后的OpenCV Mat对象，智能裁剪和缩放。
 *
 * 该函数从输入的NvBufSurface缓冲区中提取指定索引的图像区域，根据输出尺寸和裁剪区域
 * 的关系，采用智能裁剪和缩放策略：
 * 1. 如果输出尺寸大于裁剪区域，则以裁剪区域中心为中心，从原图裁剪出输出尺寸的区域。
 * 2. 如果输出尺寸小于裁剪区域，则先从原图裁剪出以裁剪区域中心为中心、以较大边为基准的正方形，
 *    然后缩放到输出尺寸。
 * 3. 超出原图范围的部分用灰色[128,128,128]填充。
 *
 * @param[in]  input_buf           输入的NvBufSurface缓冲区指针。
 * @param[in]  self                指向Gstvideorecognition实例的指针。
 * @param[in]  idx                 输入缓冲区中要处理的surface索引。
 * @param[in]  crop_rect_params    指定裁剪区域的矩形参数（NvOSD_RectParams）。
 * @param[in]  input_width         输入图像宽度。
 * @param[in]  input_height        输入图像高度。
 * @param[in]  output_width        期望的输出图像宽度。
 * @param[in]  output_height       期望的输出图像高度。
 * @param[out] out_cvMat           输出转换后的OpenCV Mat对象（RGB格式，CV_8UC3）。
 *
 * @return
 *   - GST_FLOW_OK      成功转换并生成Mat对象。
 *   - GST_FLOW_ERROR   转换或映射过程中发生错误。
 */
static GstFlowReturn
get_converted_mat(NvBufSurface *input_buf, Gstvideorecognition *self, gint idx,
                  NvOSD_RectParams *crop_rect_params,
                  gint input_width, gint input_height,
                  gint output_width, gint output_height, cv::Mat &out_cvMat)
{
    NvBufSurfTransform_Error       err;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams       transform_params;
    NvBufSurfTransformRect         src_rect;
    NvBufSurfTransformRect         dst_rect;
    NvBufSurface                   ip_surf;
    
    // 声明所有变量在函数开头
    gint crop_center_x, crop_center_y;
    gint final_src_left, final_src_top, final_src_width, final_src_height;
    gint final_dest_width, final_dest_height;
    bool need_resize = false;
    bool need_padding = false;
    gint actual_src_left, actual_src_top, actual_src_width, actual_src_height;
    gint actual_src_right, actual_src_bottom;
    gint dst_offset_x, dst_offset_y;
    gint dst_w, dst_h;
    float scale;
    NvBufSurfaceCreateParams create_params;
    cv::Mat temp_mat;
    
    ip_surf = *input_buf;

    ip_surf.numFilled = ip_surf.batchSize = 1;
    ip_surf.surfaceList = &(input_buf->surfaceList[idx]);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0))
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__),
                          (NULL));
        goto error;
    }

    // 计算裁剪区域的中心点
    crop_center_x = crop_rect_params->left + crop_rect_params->width / 2;
    crop_center_y = crop_rect_params->top + crop_rect_params->height / 2;

    // 判断使用哪种裁剪策略
    if (output_width >= crop_rect_params->width && 
        output_height >= crop_rect_params->height)
    {
        // 情况1: 输出尺寸大于裁剪区域，直接以中心裁剪输出尺寸的区域
        gint max_side;
        final_src_width = output_width;
        final_src_height = output_height;
        final_src_left = crop_center_x - output_width / 2;
        final_src_top = crop_center_y - output_height / 2;
        final_dest_width = output_width;
        final_dest_height = output_height;
        need_resize = false;
    }
    else
    {
        // 情况2: 输出尺寸小于裁剪区域的某个边，裁剪正方形然后缩放
        gint max_side = std::max(crop_rect_params->width, crop_rect_params->height);
        final_src_width = max_side;
        final_src_height = max_side;
        final_src_left = crop_center_x - max_side / 2;
        final_src_top = crop_center_y - max_side / 2;
        final_dest_width = output_width;
        final_dest_height = output_height;
        need_resize = true;
    }

    // 检查是否超出原图边界，需要padding
    if (final_src_left < 0 || final_src_top < 0 ||
        final_src_left + final_src_width > input_width ||
        final_src_top + final_src_height > input_height)
    {
        need_padding = true;
        // 计算实际可裁剪的区域（与原图的交集）
        actual_src_left = std::max(0, final_src_left);
        actual_src_top = std::max(0, final_src_top);
        actual_src_right = std::min(final_src_left + final_src_width, input_width);
        actual_src_bottom = std::min(final_src_top + final_src_height, input_height);
        actual_src_width = actual_src_right - actual_src_left;
        actual_src_height = actual_src_bottom - actual_src_top;
    }
    else
    {
        // 不需要padding，直接使用final_src作为actual_src
        actual_src_left = final_src_left;
        actual_src_top = final_src_top;
        actual_src_width = final_src_width;
        actual_src_height = final_src_height;
    }

    // 对齐到2的倍数（硬件要求）
    actual_src_left = GST_ROUND_UP_2((unsigned int)actual_src_left);
    actual_src_top = GST_ROUND_UP_2((unsigned int)actual_src_top);
    actual_src_width = GST_ROUND_DOWN_2((unsigned int)actual_src_width);
    actual_src_height = GST_ROUND_DOWN_2((unsigned int)actual_src_height);

    /* 配置转换参数 */
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id = self->gpu_id;
    transform_config_params.cuda_stream = self->cuda_stream;

    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success)
    {
        GST_ELEMENT_ERROR(
            self, STREAM, FAILED,
            ("NvBufSurfTransformSetSessionParams failed with error %d", err),
            (NULL));
        goto error;
    }

    // 重新分配或清空中间缓冲区
    if (self->inter_buf->surfaceList[0].width != final_dest_width ||
        self->inter_buf->surfaceList[0].height != final_dest_height)
    {
        NvBufSurfaceDestroy(self->inter_buf);
        memset(&create_params, 0, sizeof(create_params));
        create_params.gpuId = self->gpu_id;
        create_params.width = final_dest_width;
        create_params.height = final_dest_height;
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

    // 用灰色填充整个缓冲区（处理padding情况）
    if (need_padding)
    {
        if (NvBufSurfaceMap(self->inter_buf, 0, 0, NVBUF_MAP_WRITE) != 0)
        {
            goto error;
        }
        // 填充为灰色 [128, 128, 128]
        for (int y = 0; y < final_dest_height; y++)
        {
            unsigned char *row = (unsigned char *)self->inter_buf->surfaceList[0].mappedAddr.addr[0] +
                                 y * self->inter_buf->surfaceList[0].pitch;
            for (int x = 0; x < final_dest_width; x++)
            {
                row[x * 3 + 0] = 128; // R
                row[x * 3 + 1] = 128; // G
                row[x * 3 + 2] = 128; // B
            }
        }
        NvBufSurfaceUnMap(self->inter_buf, 0, 0);
    }
    else
    {
        NvBufSurfaceMemSet(self->inter_buf, 0, 0, 0);
    }

    // 设置源和目标矩形
    src_rect = {(guint)actual_src_top, (guint)actual_src_left, 
                (guint)actual_src_width, (guint)actual_src_height};

    if (need_padding)
    {
        // 计算在目标缓冲区中的位置（居中对齐）
        dst_offset_x = (final_src_left < 0) ? (-final_src_left) : 0;
        dst_offset_y = (final_src_top < 0) ? (-final_src_top) : 0;
        
        if (need_resize)
        {
            // 需要缩放，计算缩放后的偏移
            scale = (float)final_dest_width / final_src_width;
            dst_offset_x = (gint)(dst_offset_x * scale);
            dst_offset_y = (gint)(dst_offset_y * scale);
            dst_w = (gint)(actual_src_width * scale);
            dst_h = (gint)(actual_src_height * scale);
            dst_rect = {(guint)dst_offset_y, (guint)dst_offset_x, (guint)dst_w, (guint)dst_h};
        }
        else
        {
            dst_rect = {(guint)dst_offset_y, (guint)dst_offset_x, 
                       (guint)actual_src_width, (guint)actual_src_height};
        }
    }
    else
    {
        if (need_resize)
        {
            dst_rect = {0, 0, (guint)final_dest_width, (guint)final_dest_height};
        }
        else
        {
            dst_rect = {0, 0, (guint)actual_src_width, (guint)actual_src_height};
        }
    }

    /* 设置转换参数 */
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER |
                                      NVBUFSURF_TRANSFORM_CROP_SRC |
                                      NVBUFSURF_TRANSFORM_CROP_DST;
    transform_params.transform_filter = NvBufSurfTransformInter_Bilinear;

    GST_DEBUG_OBJECT(self, "Transforming buffer: src[%d,%d,%d,%d] -> dst[%d,%d,%d,%d]\n",
                     src_rect.left, src_rect.top, src_rect.width, src_rect.height,
                     dst_rect.left, dst_rect.top, dst_rect.width, dst_rect.height);

    /* 执行转换 */
    err = NvBufSurfTransform(&ip_surf, self->inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success)
    {
        GST_ELEMENT_ERROR(
            self, STREAM, FAILED,
            ("NvBufSurfTransform failed with error %d while converting buffer", err),
            (NULL));
        goto error;
    }

    /* 映射缓冲区以便CPU访问 */
    if (NvBufSurfaceMap(self->inter_buf, 0, 0, NVBUF_MAP_READ) != 0)
    {
        goto error;
    }
    if (self->inter_buf->memType == NVBUF_MEM_SURFACE_ARRAY)
    {
        NvBufSurfaceSyncForCpu(self->inter_buf, 0, 0);
    }

    /* 映射为RGB三通道 (CV_8UC3) */
    temp_mat = cv::Mat(final_dest_height, final_dest_width, CV_8UC3,
                       self->inter_buf->surfaceList[0].mappedAddr.addr[0],
                       self->inter_buf->surfaceList[0].pitch);
    
    // 深拷贝到输出Mat，避免unmap后数据失效
    out_cvMat = temp_mat.clone();

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
        gobject_class, PROP_SAMPLING_RATE,
        g_param_spec_int(
            "sampling-rate", "Sampling Rate",
            "Sampling rate for frame extraction (e.g., 5 means take every 5th "
            "frame)",
            1, G_MAXINT, DEFAULT_SAMPLING_RATE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_TRT_ENGINE_NAME,
        g_param_spec_string(
            "trt-engine-name", "TensorRT Engine Name",
            "Name of the TensorRT engine file to use for inference",
            DEFAULT_TRT_ENGINE_NAME,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(
        gobject_class, PROP_LABELS_FILE,
        g_param_spec_string(
            "labels-file", "Labels File",
            "Path to the file containing class labels (one per line)",
            DEFAULT_LABELS_FILE,
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
    self->processing_frame_interval = 1;
    self->model_clip_length = DEFAULT_PROCESSING_MODEL_CLIP_LENGTH;
    self->model_sampling_rate = DEFAULT_SAMPLING_RATE;
    // X3D需要：num_frames * sampling_rate 的总帧数
    self->max_history_frames =
        self->model_clip_length * self->model_sampling_rate + 10;
    self->trtProcessPtr = new Process(self->max_history_frames);
    const char *trt_engine_file = DEFAULT_TRT_ENGINE_NAME;
    strncpy(self->trt_engine_name, trt_engine_file,
            sizeof(self->trt_engine_name) - 1);
    self->trt_engine_name[sizeof(self->trt_engine_name) - 1] = '\0';
    self->recognitionResultPtr = new RECOGNITION();
    self->recognitionResultPtr->class_id = -1;
    self->recognitionResultPtr->class_name.clear();
    self->recognitionResultPtr->score = 0.f;
    self->video_recognition = nullptr;
    
    // 初始化标签文件路径
    const char *labels_file = DEFAULT_LABELS_FILE;
    strncpy(self->labels_file, labels_file, sizeof(self->labels_file) - 1);
    self->labels_file[sizeof(self->labels_file) - 1] = '\0';
    
    // 初始化标签映射
    self->labels_map = (void*)(new std::map<int, std::string>());

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

        // 遍历检测到的目标
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            
            // 获取单目标跟踪框
            NvOSD_RectParams track_rect_params = obj_meta->rect_params;
            
            cv::Mat track_frame_mat;
            if (get_converted_mat(
                    surface, self, frame_meta->batch_id, &track_rect_params,
                    surface->surfaceList[frame_meta->batch_id].width,
                    surface->surfaceList[frame_meta->batch_id].height,
                    self->processing_width, self->processing_height,
                    track_frame_mat) == GST_FLOW_OK &&
                !track_frame_mat.empty())
            {
                // 推入帧缓冲
                if (self->trtProcessPtr)
                {
                    self->trtProcessPtr->addFrame(track_frame_mat);
                    // 满足长度后做一次推理
                    if (self->trtProcessPtr->getCurrentFrameLength() ==
                            (int)self->max_history_frames &&
                        self->video_recognition)
                    {
                        std::vector<float> input_data;
                        // X3D预处理：采样帧，归一化
                        self->trtProcessPtr->convertCvInputToX3dTensorRT(
                            input_data, self->model_clip_length,
                            self->processing_height, self->processing_width,
                            self->model_sampling_rate);

                        X3dTrt *x3dPtr =
                            dynamic_cast<X3dTrt *>(self->video_recognition);
                        if (x3dPtr)
                        {
                            // 仅第一次调用时 prepare_output
                            bool ok_prepare_output = true;
                            if (x3dPtr->GetOutputSize() == 0)
                                ok_prepare_output =
                                    x3dPtr->prepare_output("output");
                            if (x3dPtr->prepare_input(
                                    "input", self->model_clip_length,
                                    self->processing_height, self->processing_width,
                                    input_data.data()) &&
                                ok_prepare_output)
                            {
                                if (x3dPtr->do_inference())
                                {
                                    float *output_data =
                                        new float[x3dPtr->GetOutputSize() /
                                                  sizeof(float)];
                                    x3dPtr->get_output(output_data);
                                    RECOGNITION result =
                                        x3dPtr->parse_output(output_data);
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

            // 将视频识别结果写入对象元数据
            if (self->recognitionResultPtr->score >= 0.5)
            {
                NvDsClassifierMeta *classifier_meta =
                    nvds_acquire_classifier_meta_from_pool(batch_meta);
                classifier_meta->unique_component_id = 9; // X3D component id
                NvDsLabelInfo *label_info =
                    nvds_acquire_label_info_meta_from_pool(batch_meta);
                label_info->result_class_id =
                    self->recognitionResultPtr->class_id;
                label_info->result_prob = self->recognitionResultPtr->score;
                
                // 从标签映射中查找类别名
                const char *label_name = "unknown";
                if (self->labels_map)
                {
                    std::map<int, std::string> *labels = (std::map<int, std::string>*)self->labels_map;
                    if (labels->find(label_info->result_class_id) != labels->end())
                    {
                        label_name = (*labels)[label_info->result_class_id].c_str();
                    }
                }
                
                strncpy(label_info->result_label, label_name, MAX_LABEL_SIZE - 1);
                label_info->result_label[MAX_LABEL_SIZE - 1] = '\0';
                
                nvds_add_label_info_meta_to_classifier(classifier_meta,
                                                       label_info);
                nvds_add_classifier_meta_to_object(obj_meta, classifier_meta);
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
    
    // 加载类别标签文件
    if (strlen(self->labels_file) > 0)
    {
        std::ifstream label_file(self->labels_file);
        if (label_file.is_open())
        {
            std::string line;
            int class_id = 0;
            std::map<int, std::string> *labels = (std::map<int, std::string>*)self->labels_map;
            labels->clear();
            
            while (std::getline(label_file, line))
            {
                // 去除行首尾空白字符
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                
                if (!line.empty())
                {
                    (*labels)[class_id] = line;
                    class_id++;
                }
            }
            label_file.close();
            g_print("Successfully loaded %d labels from %s\n", class_id, self->labels_file);
        }
        else
        {
            GST_WARNING_OBJECT(self, "Failed to open labels file: %s, will use 'unknown' for all classes", self->labels_file);
        }
    }

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
        // 更新最大历史帧数
        self->max_history_frames =
            self->model_clip_length * self->model_sampling_rate + 10;
        if (self->trtProcessPtr)
        {
            delete self->trtProcessPtr;
            self->trtProcessPtr = new Process(self->max_history_frames);
        }
        break;
    case PROP_SAMPLING_RATE:
        self->model_sampling_rate = g_value_get_int(value);
        if (self->model_sampling_rate <= 0)
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("sampling_rate must be greater than 0"), (NULL));
            return;
        }
        // 更新最大历史帧数
        self->max_history_frames =
            self->model_clip_length * self->model_sampling_rate + 10;
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
        if (self->video_recognition)
        {
            delete self->video_recognition;
            self->video_recognition = nullptr;
        }
        // 初始化X3D引擎
        self->video_recognition =
            new X3dTrt(self->trt_engine_name, self->processing_width);
        GST_INFO_OBJECT(self, "Initialized X3D recognition engine with %s",
                        self->trt_engine_name);
        break;
    }
    case PROP_LABELS_FILE:
    {
        const gchar *s = g_value_get_string(value);
        const gchar *labels = s && *s ? s : DEFAULT_LABELS_FILE;
        strncpy(self->labels_file, labels, sizeof(self->labels_file) - 1);
        self->labels_file[sizeof(self->labels_file) - 1] = '\0';
        GST_INFO_OBJECT(self, "Set labels file to: %s", self->labels_file);
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
    case PROP_SAMPLING_RATE:
        g_value_set_int(value, self->model_sampling_rate);
        break;
    case PROP_LABELS_FILE:
        g_value_set_string(value, self->labels_file);
        break;
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
    
    if (self->labels_map)
    {
        delete (std::map<int, std::string>*)self->labels_map;
        self->labels_map = NULL;
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