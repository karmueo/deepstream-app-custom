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

class DeepTracker
{

public:
    DeepTracker(const std::string &engine_name);
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
    uint32_t list_size_;
};