#pragma once

// #include "mixformer_trt.h"
#include "nvdstracker.h"
#include "suTrack_trt.h"
#include "ostrack_trt.h"

static float IOU(const cv::Rect &srcRect, const cv::Rect &dstRect);

struct TrackInfo
{
    int64_t trackId;
    int64_t age;
    int64_t miss;
    DrOBB bbox;
};

enum MODEL_NAME
{
    MODEL_SUTRACK = 0,   // SUTRACK model
    MODEL_OSTRACK,       // OSTRACK model
    MODEL_MIXFORMERV2
};

struct TARGET_MANAGEMENT
{
    float    expandFactor            = 1.0f;  // 跟踪框膨胀
    uint16_t probationAge            = 5;     // 跟踪成功多少次后进入跟踪状态，默认5次
    uint16_t maxMiss                 = 10;    // 最大连续丢失次数，超过后认为跟踪失败，默认10次
    float    scoreThreshold          = 0.3f;  // 跟踪分数阈值，默认0.3
    float    iouThreshold            = 0.5f;  // IOU阈值，默认0.5
    float    trackBoxWidthThreshold  = 0.3f;  // 跟踪框宽度阈值，默认0.3
    float    trackBoxHeightThreshold = 0.3f;  // 跟踪框高度阈值，默认0.3
    uint32_t maxTrackAge             = 30;    // 最大跟踪年龄，超过后认为跟踪失败，默认30
};

struct TRACKER_CONFIG
{
    std::string       modelRootPath = "../Mixformer_plugin/models";  // 模型根路径
    MODEL_NAME        modelName     = MODEL_SUTRACK;                 // 模型类型
    uint8_t           modelType     = 0;                             // 模型类型，0:FP32, 1:FP16
    TARGET_MANAGEMENT targetManagement;                              // 目标管理配置
    uint32_t          confirmAgeThreshold = 5;                       // 确认跟踪的年龄阈值，默认5
};

class DeepTracker
{

public:
    DeepTracker(const std::string &engine_name, const TRACKER_CONFIG &trackerConfig);
    ~DeepTracker();

    TrackInfo update(const cv::Mat &img, const NvMOTObjToTrackList *detectObjList, const uint32_t frameNum);

    bool isTracked() const
    {
        return is_tracked_;
    }

    TrackInfo getTrackInfo() const
    {
        return trackInfo_;
    }

    void updatePastFrameObjBatch(NvDsTargetMiscDataBatch *pastFrameObjBatch);

private:
    bool is_tracked_;
    int64_t age_;
    int64_t trackId_;
    uint32_t miss_;
    NvMOTObjToTrack *objectToTrack_;
    std::unique_ptr<BaseTrackTRT> trackerPtr_;
    TrackInfo trackInfo_;
    uint32_t frameNum_;
    NvDsTargetMiscDataFrame *list_;
    uint32_t list_capacity_ = 30;  // 默认容量为30
    uint32_t list_size_;
    bool     enableTrackCenterStable_;          // 是否启用跟踪中心位置稳定判断
    uint32_t trackCenterStablePixelThreshold_;  // 跟踪中心位置稳定判断的像素阈值，单位像素
    TRACKER_CONFIG trackerConfig_;
    uint32_t confirmAgeThreshold_;
};