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

    /* 保持纵横比 */
    double hdest = self->processing_width * src_height / (double)src_width;
    double wdest = self->processing_height * src_width / (double)src_height;
    guint dest_width, dest_height;

    if (hdest <= self->processing_height)
    {
        dest_width = self->processing_width;
        dest_height = hdest;
    }
    else
    {
        dest_width = wdest;
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

    /* 计算缩放比率，同时保持长宽比 */
    ratio = MIN(1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

    if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0))
    {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
                          ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
        goto error;
    }

    /* 为src和dst设置ROI */
    src_rect = {(guint)src_top, (guint)src_left, (guint)src_width, (guint)src_height};

    // 计算上下左右 padding，使 dst_rect 居中
    pad_x = (self->processing_width > dest_width ? (self->processing_width - dest_width) / 2 : 0);
    pad_y = (self->processing_height > dest_height ? (self->processing_height - dest_height) / 2 : 0);
    dst_rect = {pad_y, pad_x, (guint)dest_width, (guint)dest_height};

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
    // TODO: 初始化一些参数
    self->gpu_id = 0;
    /* Initialize all property variables to default values */
    self->unique_id = 15;
    self->gpu_id = 0;
    self->frame_num = 0;
    self->processing_width = 224;
    self->processing_height = 224;
    // self->processing_frame_interval = 5;
    self->processing_frame_interval = 1;
    // 根据模型选择num_clips和clip_length和processing_width
    // self->model_num_clips = 1;
    self->model_num_clips = 4;
    // self->model_clip_length = 32;
    self->model_clip_length = 16;
    self->max_history_frames = self->processing_frame_interval * self->model_clip_length * self->model_num_clips + self->model_num_clips * 2;
    self->trtProcessPtr = new Process(self->max_history_frames);
    self->video_recognition = new tsnTrt(
        "/workspace/deepstream-app-custom/src/gst-videorecognition/models/uniformerv2_end2end_fp32.engine",
        self->processing_width);
    self->recognitionResultPtr = new RECOGNITION();

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

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
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
                                  target_img) != GST_FLOW_OK)
            {
                GST_ELEMENT_ERROR(self, STREAM, FAILED,
                                  ("get_converted_mat failed"), (NULL));
                goto error;
            }
            self->trtProcessPtr->addFrame(target_img);

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
                    strncpy(label_info->result_label, "bird", MAX_LABEL_SIZE - 1);
                    label_info->result_label[MAX_LABEL_SIZE - 1] = '\0'; // 确保空字符结尾
                }
                else if (label_info->result_class_id == 1)
                {
                    strncpy(label_info->result_label, "uav", MAX_LABEL_SIZE - 1);
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

    // 多帧推理
    if (self->trtProcessPtr->getCurrentFrameLength() == self->max_history_frames)
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
            input_data,
            self->model_num_clips,
            self->model_clip_length,
            self->processing_width,
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
            tsnTrt *tsnPtr = dynamic_cast<tsnTrt *>(self->video_recognition);
            tsnPtr->prepare_input("input", self->model_num_clips, self->model_clip_length, input_data.data());
            tsnPtr->prepare_output("/Softmax_output_0");
            tsnPtr->do_inference();
            float *output_data = new float[tsnPtr->GetOutputSize()];
            tsnPtr->get_output(output_data);
            RECOGNITION result = tsnPtr->parse_output(output_data); // result是一个RECOGNITION对象
            self->recognitionResultPtr->class_id = result.class_id;
            self->recognitionResultPtr->class_name = result.class_name;
            self->recognitionResultPtr->score = result.score;

            std::cout << "Class ID: " << self->recognitionResultPtr->class_id << ", Class Name: " << self->recognitionResultPtr->class_name
                      << ", Score: " << self->recognitionResultPtr->score << std::endl;

            delete[] output_data;
        }
        else
        {
            std::cerr << "Error: video_recognition is null" << std::endl;
        }
        self->trtProcessPtr->clearFrames();
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