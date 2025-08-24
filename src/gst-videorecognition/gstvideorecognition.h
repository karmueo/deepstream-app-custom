#ifndef __GST__VIDEORECOGNITION_H__
#define __GST__VIDEORECOGNITION_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "gstnvdsmeta.h"
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>
#include "tsnTrt.h"
#include "videorecognitionTrt.h"
#include "process.h"
#include "imageClsTrt.h"
#include "nvbufsurface.h"

#define PACKAGE "videorecognition"
#define VERSION "1.0.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "My videorecognition plugin for Deepstream Network"
#define BINARY_PACKAGE "NVIDIA DeepStream 3rdparty plugin"
#define URL "https://github.com/karmueo/"

G_BEGIN_DECLS

typedef struct _GstvideorecognitionClass GstvideorecognitionClass;
typedef struct _Gstvideorecognition Gstvideorecognition;

#define GST_TYPE_VIDEORECOGNITION (gst_videorecognition_get_type())
#define GST_VIDEORECOGNITION(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_VIDEORECOGNITION, Gstvideorecognition))

struct _Gstvideorecognition
{
    GstBaseTransform base_trans;

    guint unique_id;

    guint gpu_id;

    // Frame number of the current input buffer
    guint64 frame_num;

    // Input video info (resolution, color format, framerate, etc)
    GstVideoInfo video_info;

    VideoRecognitionTRT *video_recognition;

    // CUDA Stream used for allocating the CUDA task
    cudaStream_t cuda_stream;

    // Resolution at which frames/objects should be processed
    gint processing_width;
    gint processing_height;
    gint model_clip_length;
    gint model_num_clips;
    gint processing_frame_interval;

    // 用于RGBA转换的中间临时缓冲区
    NvBufSurface *inter_buf;

    // 最大保存历史帧数
    guint32 max_history_frames;

    Process *trtProcessPtr;

    ImageClsTrt *frame_classifier;      // yolov11m_classify_ir_fp32.engine
    float frame_cls_scores[3];          // 最近一帧 softmax 结果
    // 视频识别结果结构
    RECOGNITION *recognitionResultPtr;
    // 单目标分类: 最近 N(<=10) 帧窗口加权
    int cls_window_size = 10;           // 固定窗口长度
    float cls_window_scores[10][3];     // 环形缓冲保存 softmax
    int cls_window_index = 0;           // 写入位置
    int cls_window_count = 0;           // 已填充帧数 (<= window_size)
    // 分类器 TensorRT 引擎路径（来自 trt-engine-name 属性）
    char trt_engine_name[512];
    // 0 = 多帧图片分类模式 (默认)，1 = 视频时序识别模式
    gint model_type;
};

struct _GstvideorecognitionClass
{
    GstBaseTransformClass parent_class;
};

GType gst_videorecognition_get_type(void);

G_END_DECLS

#endif /* __GST__VIDEORECOGNITION_H__ */
