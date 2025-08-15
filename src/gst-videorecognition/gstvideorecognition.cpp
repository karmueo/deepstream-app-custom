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
#include "nvbufsurftransform.h"
#include <gst/gstelement.h>
// #include "gstnvdsinfer.h"
#include <opencv2/opencv.hpp>

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

// 缩放方式
enum GstVRScaleMode {
    GST_VR_SCALE_ASPECT = 0,   // 等比保持长宽比（可能带 padding）
    GST_VR_SCALE_STRETCH = 1   // 直接拉伸到目标尺寸（不保持长宽比，不加 padding）
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
    PROP_TRT_ENGINE_NAME
};

#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 224
#define DEFAULT_PROCESSING_HEIGHT 224
#define DEFAULT_PROCESSING_MODEL_CLIP_LENGTH 8
#define DEFAULT_PROCESSING_NUM_CLIPS 4
#define DEFAULT_TRT_ENGINE_NAME "/workspace/deepstream-app-custom/src/gst-videorecognition/models/uniformerv2_e1_end2end_fp32.engine"

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

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio. Remove the padding required by hardware and convert from RGBA to RGB
 * using openCV. These steps can be skipped if the algorithm can work with
 * padded data and/or can work with RGBA.
 * 等比缩放视频帧到处理分辨率。
 * 或等比裁剪和缩放的目标到处理分辨率。
 * 删除硬件所需的padding，从RGBA转换为RGB。
 * 如果算法可以直接使用padding和RGBA则可以跳过这些步骤。
 */
/**
 * @brief 获取转换后的OpenCV Mat对象，并进行缩放和格式转换。
 *
 * 该函数从输入的NvBufSurface缓冲区中提取指定索引的图像区域（可选裁剪），
 * 按照目标处理分辨率进行缩放，同时保持宽高比，并将图像格式从RGBA转换为BGR，保存到self->cvmat中。
 *
 * @param[in]  self                指向Gstvideorecognition实例的指针，包含处理参数和中间缓冲区。
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
static GstFlowReturn get_converted_mat(NvBufSurface *input_buf,
                                       Gstvideorecognition *self,
                                       gint idx,
                                       NvOSD_RectParams *crop_rect_params,
                                       gdouble &ratio,
                                       gint input_width,
                                       gint input_height,
                                       GstVRScaleMode scale_mode,
                                       cv::Mat &out_cvMat)
{
    NvBufSurfTransform_Error err;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    NvBufSurfTransformRect src_rect;
    NvBufSurfTransformRect dst_rect;
    NvBufSurface ip_surf;
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
    if (scale_mode == GST_VR_SCALE_ASPECT) {
        // 保持纵横比
        double hdest = self->processing_width * src_height / (double)src_width;
        double wdest = self->processing_height * src_width / (double)src_height;
        if (hdest <= self->processing_height) {
            dest_width = self->processing_width;
            dest_height = static_cast<guint>(hdest);
        } else {
            dest_width = static_cast<guint>(wdest);
            dest_height = self->processing_height;
        }
    } else {
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
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
        goto error;
    }

    /* 计算缩放比率。等比时为实际等比系数；拉伸时给出最小系数供外部参考 */
    ratio = MIN(1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0))
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
        goto error;
    }

    /* 为src和dst设置ROI */
    src_rect = {(guint)src_top, (guint)src_left, (guint)src_width, (guint)src_height};

    if (scale_mode == GST_VR_SCALE_ASPECT) {
        // 计算上下左右 padding，使 dst_rect 居中
        pad_x = (self->processing_width > dest_width ? (self->processing_width - dest_width) / 2 : 0);
        pad_y = (self->processing_height > dest_height ? (self->processing_height - dest_height) / 2 : 0);
        dst_rect = {pad_y, pad_x, (guint)dest_width, (guint)dest_height};
    } else {
        // 无 padding，直接填满
        pad_x = pad_y = 0;
        dst_rect = {0u, 0u, (guint)dest_width, (guint)dest_height};
    }

    /* Set the transform parameters */
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag =
        NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
        NVBUFSURF_TRANSFORM_CROP_DST;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;

    /* Memset the memory */
    NvBufSurfaceMemSet(self->inter_buf, 0, 0, 0);

    GST_DEBUG_OBJECT(self, "Scaling and converting input buffer\n");

    /* 转换缩放+格式转换（如果有）。 */
    err = NvBufSurfTransform(&ip_surf, self->inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success)
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("NvBufSurfTransform failed with error %d while converting buffer", err),
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

    /* Use openCV to remove padding and convert RGBA to BGR. Can be skipped if
     * algorithm can handle padded RGBA data. */
    // 判断out_cvMat是否已经分配内存
    if (out_cvMat.empty())
    {
        out_cvMat = cv::Mat(self->processing_height, self->processing_width,
                            CV_8UC4, self->inter_buf->surfaceList[0].mappedAddr.addr[0],
                            self->inter_buf->surfaceList[0].pitch);
    }
    else
    {
        // 如果已经分配了内存，则删除原有的内存
        out_cvMat.release();
        out_cvMat = cv::Mat(self->processing_height, self->processing_width,
                            CV_8UC4, self->inter_buf->surfaceList[0].mappedAddr.addr[0],
                            self->inter_buf->surfaceList[0].pitch);
    }
    cv::cvtColor(out_cvMat, out_cvMat, cv::COLOR_RGBA2BGR);

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

    /* Install properties */
    g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
                                    g_param_spec_uint("unique-id",
                                                      "Unique ID",
                                                      "Unique ID for the element. Can be used to identify output of the element",
                                                      0,
                                                      G_MAXUINT,
                                                      DEFAULT_UNIQUE_ID,
                                                      (GParamFlags)(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_GPU_DEVICE_ID,
                                    g_param_spec_uint("gpu-id",
                                                      "Set GPU Device ID",
                                                      "Set GPU Device ID",
                                                      0,
                                                      G_MAXUINT,
                                                      0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    g_object_class_install_property(gobject_class, PROP_PROCESSING_WIDTH,
                                    g_param_spec_int("processing-width",
                                                     "Processing Width",
                                                     "Width of the input buffer to algorithm",
                                                     1,
                                                     G_MAXINT,
                                                     DEFAULT_PROCESSING_WIDTH,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PROCESSING_HEIGHT,
                                    g_param_spec_int("processing-height",
                                                     "Processing Height",
                                                     "Height of the input buffer to algorithm",
                                                     1,
                                                     G_MAXINT,
                                                     DEFAULT_PROCESSING_HEIGHT,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_MODEL_CLIP_LENGTH,
                                    g_param_spec_int("model-clip-length",
                                                     "Model Clip Length",
                                                     "Length of the clip used by the model",
                                                     1,
                                                     G_MAXINT,
                                                     DEFAULT_PROCESSING_MODEL_CLIP_LENGTH,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_NUM_CLIPS,
                                    g_param_spec_int("num-clips",
                                                     "Number of Clips",
                                                     "Number of clips used by the model",
                                                     1,
                                                     G_MAXINT,
                                                     DEFAULT_PROCESSING_NUM_CLIPS,
                                                     (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_TRT_ENGINE_NAME,
                                    g_param_spec_string("trt-engine-name",
                                                        "TensorRT Engine Name",
                                                        "Name of the TensorRT engine file to use for inference",
                                                        DEFAULT_TRT_ENGINE_NAME,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

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
    self->max_history_frames = self->processing_frame_interval * self->model_clip_length * self->model_num_clips + self->model_num_clips * 2;
    self->trtProcessPtr = new Process(self->max_history_frames);
    const char *trt_engine_file = DEFAULT_TRT_ENGINE_NAME;
    self->video_recognition = new tsnTrt(
        trt_engine_file,
        self->processing_width);
    self->recognitionResultPtr = new RECOGNITION();
    self->frame_classifier = new ImageClsTrt("/workspace/deepstream-app-custom/src/deepstream-app/models/yolov11m_classify_ir_fp32.engine");
    if (!self->frame_classifier->prepare()) {
        g_printerr("[videorecognition] frame classifier prepare failed\n");
    }
    memset(self->frame_cls_scores, 0, sizeof(self->frame_cls_scores));

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
    gdouble scale_ratio = 1.0;

    NvBufSurface *surface = NULL;
    NvDsBatchMeta *batch_meta = NULL;
    NvDsFrameMeta *frame_meta = NULL;
    NvDsMetaList *l_frame = NULL;

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
                          ("NvDsBatchMeta not found for input buffer."), (NULL));
        return GST_FLOW_ERROR;
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsMetaList *l_obj = NULL;
        NvDsObjectMeta *obj_meta = NULL;
        frame_meta = (NvDsFrameMeta *)(l_frame->data);
        if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] == NULL)
        {
            if (NvBufSurfaceMap(surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0)
            {
                GST_ELEMENT_ERROR(self, STREAM, FAILED,
                                  ("%s:buffer map to be accessed by CPU failed", __func__), (NULL));
                return GST_FLOW_ERROR;
            }
        }

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            cv::Mat target_img;
            /* Scale and convert the frame */
            if (get_converted_mat(surface,
                                  self,
                                  frame_meta->batch_id,
                                  &obj_meta->rect_params,
                                  scale_ratio,
                                  surface->surfaceList[frame_meta->batch_id].width,
                                  surface->surfaceList[frame_meta->batch_id].height,
                                  GST_VR_SCALE_STRETCH, // 可改为 GST_VR_SCALE_STRETCH 以开启拉伸模式
                                  target_img) != GST_FLOW_OK)
            {
                GST_ELEMENT_ERROR(self, STREAM, FAILED,
                                  ("get_converted_mat failed"), (NULL));
                goto error;
            }

            // 对对象裁剪图(target_img, BGR)执行单帧分类
            if (self->frame_classifier && !target_img.empty()) {
                cv::Mat resized, rgb;
                // target_img 已是 processing_height x processing_width (一般 224x224)，若不同再调整
                if (target_img.cols != 224 || target_img.rows != 224) {
                    cv::resize(target_img, resized, cv::Size(224, 224));
                } else {
                    resized = target_img;
                }
                cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
                // 把图片保存下来
                cv::imwrite("target_img2.jpg", rgb);
                std::vector<float> input(3 * 224 * 224);
                int hw = 224 * 224;
                for (int y = 0; y < 224; ++y) {
                    const unsigned char *row = rgb.ptr<unsigned char>(y);
                    for (int x = 0; x < 224; ++x) {
                        int pos = y * 224 + x; int base = x * 3;
                        input[0 * hw + pos] = row[base + 0] / 255.0f;
                        input[1 * hw + pos] = row[base + 1] / 255.0f;
                        input[2 * hw + pos] = row[base + 2] / 255.0f;
                    }
                }
                float output[3] = {0};
                if (self->frame_classifier->infer(input.data(), output)) {
                    // 直接附着到该对象，组件 ID 100
                    NvDsClassifierMeta *cls_meta = nvds_acquire_classifier_meta_from_pool(batch_meta);
                    cls_meta->unique_component_id = 100; // object-level single-frame cls
                    const char *names[3] = {"未知", "鸟", "无人机"};
                    for (int c = 0; c < 3; ++c) {
                        NvDsLabelInfo *li = nvds_acquire_label_info_meta_from_pool(batch_meta);
                        li->result_class_id = c;
                        li->result_prob = output[c];
                        strncpy(li->result_label, names[c], MAX_LABEL_SIZE - 1);
                        li->result_label[MAX_LABEL_SIZE - 1] = '\0';
                        nvds_add_label_info_meta_to_classifier(cls_meta, li);
                    }
                    nvds_add_classifier_meta_to_object(obj_meta, cls_meta);
                }
            }

            if (self->recognitionResultPtr->score >= 0.5)
            {
                NvDsClassifierMeta *classifier_meta =
                    nvds_acquire_classifier_meta_from_pool(batch_meta);

                classifier_meta->unique_component_id = 9;
                NvDsLabelInfo *label_info =
                    nvds_acquire_label_info_meta_from_pool(batch_meta);

                label_info->result_class_id = self->recognitionResultPtr->class_id;
                label_info->result_prob = self->recognitionResultPtr->score;
                if (label_info->result_class_id == 0)
                {
                    strncpy(label_info->result_label, "鸟", MAX_LABEL_SIZE - 1);
                    label_info->result_label[MAX_LABEL_SIZE - 1] = '\0'; // 确保空字符结尾
                }
                else if (label_info->result_class_id == 1)
                {
                    strncpy(label_info->result_label, "无人机", MAX_LABEL_SIZE - 1);
                    label_info->result_label[MAX_LABEL_SIZE - 1] = '\0'; // 确保空字符结尾
                }
                else
                {
                    // 可选：处理其他 class_id 的情况，例如设置一个默认标签或留空
                    strncpy(label_info->result_label, "unknown", MAX_LABEL_SIZE - 1);
                    label_info->result_label[MAX_LABEL_SIZE - 1] = '\0';
                }
                nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
                nvds_add_classifier_meta_to_object(obj_meta, classifier_meta);
            }
        }
    }

    // self->trtProcessPtr->addFrame(target_img);
    // // 多帧推理
    // if (self->trtProcessPtr->getCurrentFrameLength() == self->max_history_frames)
    // {
    //     std::vector<float> input_data;
    //     // 数据预处理
    //     /* self->trtProcessPtr->convertCvInputToTensorRT(
    //         input_data,
    //         self->model_clip_length,
    //         self->processing_width,
    //         self->processing_height,
    //         self->processing_frame_interval); */
    //     self->trtProcessPtr->convertCvInputToNtchwTensorRT(
    //         input_data,
    //         self->model_num_clips,
    //         self->model_clip_length,
    //         self->processing_width,
    //         self->processing_height,
    //         self->processing_frame_interval);
    //     /* self->trtProcessPtr->loadImagesFromDirectory2(
    //         "/workspace/deepstream-app-custom/src/deepstream-app/110_video_frames/bird/bird_1/0/",
    //         input_data,
    //         self->model_num_clips,
    //         self->model_clip_length,
    //         self->processing_width,
    //         self->processing_height); */
    //     /* self->trtProcessPtr->loadImagesFromDirectory(
    //         "/workspace/deepstream-app-custom/src/deepstream-app/110_video_frames/bird/bird_1/0/",
    //         input_data,
    //         self->model_clip_length,
    //         self->processing_width,
    //         self->processing_height,
    //         self->processing_frame_interval); */

    //     if (self->video_recognition)
    //     {
    //         tsnTrt *tsnPtr = dynamic_cast<tsnTrt *>(self->video_recognition);
    //         tsnPtr->prepare_input("input", self->model_num_clips, self->model_clip_length, input_data.data());
    //         tsnPtr->prepare_output("/Softmax_output_0");
    //         tsnPtr->do_inference();
    //         float *output_data = new float[tsnPtr->GetOutputSize()];
    //         tsnPtr->get_output(output_data);
    //         RECOGNITION result = tsnPtr->parse_output(output_data); // result是一个RECOGNITION对象
    //         self->recognitionResultPtr->class_id = result.class_id;
    //         self->recognitionResultPtr->class_name = result.class_name;
    //         self->recognitionResultPtr->score = result.score;

    //         std::cout << "Class ID: " << self->recognitionResultPtr->class_id << ", Class Name: " << self->recognitionResultPtr->class_name
    //                   << ", Score: " << self->recognitionResultPtr->score << std::endl;

    //         delete[] output_data;
    //     }
    //     else
    //     {
    //         std::cerr << "Error: video_recognition is null" << std::endl;
    //     }
    //     self->trtProcessPtr->clearFrames();
    // }

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

    /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
     * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
    create_params.gpuId = self->gpu_id;
    create_params.width = self->processing_width;
    create_params.height = self->processing_height;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
    if (NvBufSurfaceCreate(&self->inter_buf, 1,
                           &create_params) != 0)
    {
        GST_ERROR("Error: Could not allocate internal buffer for dsexample");
        goto error;
    }
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
                              ("Processing width must be greater than 0"), (NULL));
            return;
        }
        break;
    case PROP_PROCESSING_HEIGHT:
        self->processing_height = g_value_get_int(value);
        if (self->processing_height <= 0)
        {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Processing height must be greater than 0"), (NULL));
            return;
        }
        break;
    case PROP_MODEL_CLIP_LENGTH:
        self->model_clip_length = g_value_get_int(value);
        if (self->model_clip_length <= 0 || self->model_clip_length > 128)
        {
            self->model_clip_length = DEFAULT_PROCESSING_MODEL_CLIP_LENGTH;
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                              ("Model clip length must be greater than 0 and less than or equal to 128"), (NULL));
           
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
        self->max_history_frames = self->processing_frame_interval * self->model_clip_length * self->model_num_clips + self->model_num_clips * 2;
        if (self->trtProcessPtr)
        {
            delete self->trtProcessPtr;
            self->trtProcessPtr = new Process(self->max_history_frames);
        }
        break;
    case PROP_TRT_ENGINE_NAME:
    {
        auto s = g_value_get_string(value);
        const gchar *trt_engine_name = s ? s : DEFAULT_TRT_ENGINE_NAME;
        if (self->video_recognition)
        {
            delete self->video_recognition;
            self->video_recognition = NULL;
        }
        self->video_recognition = new tsnTrt(trt_engine_name, self->processing_width);
        if (!self->video_recognition)
        {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                              ("Failed to create video recognition object with engine %s", trt_engine_name), (NULL));
            return;
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