#include "deepTracker.h"

// 计算两个矩形框的IOU
static float IOU(const cv::Rect &srcRect, const cv::Rect &dstRect)
{
    cv::Rect intersection;
    intersection = srcRect & dstRect;

    auto area_src = static_cast<float>(srcRect.area());
    auto area_dst = static_cast<float>(dstRect.area());
    auto area_intersection = static_cast<float>(intersection.area());
    float iou = area_intersection / (area_src + area_dst - area_intersection);
    return iou;
}

DeepTracker::DeepTracker(const std::string &engine_name)
{
    is_tracked_ = false;
    age_ = 0;
    trackId_ = 0;
    miss_ = 0;
    // FIXME:
    mixformer_ = std::make_shared<MixformerTRT>(engine_name);
    frameNum_ = 0;
    list_ = nullptr;
    list_size_ = 0;
}

DeepTracker::~DeepTracker()
{
    if (list_ != nullptr)
    {
        delete[] list_;
        list_ = nullptr;
    }
}

TrackInfo DeepTracker::update(const cv::Mat &img, const NvMOTObjToTrackList *detectObjList, const uint32_t frameNum)
{
    frameNum_ = frameNum;
    // 输出的结果存放在bbox中
    if (is_tracked_ == false)
    {
        if (img.empty() || detectObjList == nullptr || detectObjList->numFilled == 0)
        {
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        // 遍历检测到的目标
        // 当前画面的中心坐标
        auto image_cx = img.cols / 2.f;
        auto image_cy = img.rows / 2.f;
        float min_distance = 1000000.f;
        for (uint32_t numObjects = 0; numObjects < detectObjList->numFilled; numObjects++)
        {
            NvMOTObjToTrack obj = detectObjList->list[numObjects];
            // 计算哪个目标离中心最近
            auto obj_cx = obj.bbox.x + obj.bbox.width / 2.f;
            auto obj_cy = obj.bbox.y + obj.bbox.height / 2.f;
            auto distance = std::abs(obj_cx - image_cx) + std::abs(obj_cy - image_cy);
            if (distance < min_distance)
            {
                min_distance = distance;
                objectToTrack_ = &obj;
            }
        }

        trackInfo_.bbox.box.x0 = objectToTrack_->bbox.x;
        trackInfo_.bbox.box.x1 = objectToTrack_->bbox.x + objectToTrack_->bbox.width;
        trackInfo_.bbox.box.y0 = objectToTrack_->bbox.y;
        trackInfo_.bbox.box.y1 = objectToTrack_->bbox.y + objectToTrack_->bbox.height;
        trackInfo_.bbox.class_id = objectToTrack_->classId;
        mixformer_->init(img, trackInfo_.bbox);

        if (list_ != nullptr)
        {
            delete[] list_;
            list_ = nullptr;
        }
        list_size_ = 0;
        list_ = new NvDsTargetMiscDataFrame[30];

        is_tracked_ = true;
    }
    else
    {
        // 更新跟踪器并获取跟踪结果
        trackInfo_.bbox = mixformer_->track(img);
        bool is_track_match_detect = true;

        if (trackInfo_.bbox.score == 0 || trackInfo_.bbox.box.x0 == 0 || trackInfo_.bbox.box.y0 == 0)
        {
            miss_ = 100;
        }
        else
        {
            // 如果有检测结果，和检测结果对比来查看跟踪是否正确
            if (detectObjList->numFilled != 0)
            {
                is_track_match_detect = false;
                // 计算和所有检测frame->objectsIn.list结果的IOU
                cv::Rect trackRect = cv::Rect(trackInfo_.bbox.box.x0,
                                              trackInfo_.bbox.box.y0,
                                              trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0,
                                              trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0);
                for (uint i = 0; i < detectObjList->numFilled; i++)
                {
                    NvMOTObjToTrack obj = detectObjList->list[i];
                    // 计算IOU
                    cv::Rect detectionRect = cv::Rect(obj.bbox.x,
                                                      obj.bbox.y,
                                                      obj.bbox.width,
                                                      obj.bbox.height);

                    float iou = IOU(detectionRect, trackRect);
                    if (iou > 0.5)
                    {
                        // 如果IOU大于0.5，认为跟踪成功
                        is_track_match_detect = true;
                        break;
                    }
                }
            }
            // 如果跟踪置信度小于阈值或者前面和检测没有匹配的
            if (trackInfo_.bbox.score < 0.9 || !is_track_match_detect)
            {
                miss_++;
            }
        }

        if (miss_ > 20)
        {
            // 如果跟踪分数小于阈值，或者IOU小于0.5，认为跟踪失败
            is_tracked_ = false;
            age_ = 0;
            if (trackId_++ > 0xFFFFFFFF)
            {
                trackId_ = 0;
            }

            miss_ = 0;
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }
    }

    if (age_ <= 29)
    {
        list_[age_].age = age_;
        list_[age_].tBbox.left = trackInfo_.bbox.box.x0;
        list_[age_].tBbox.top = trackInfo_.bbox.box.y0;
        list_[age_].tBbox.width = trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0;
        list_[age_].tBbox.height = trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0;
        list_[age_].confidence = trackInfo_.bbox.score;
        list_[age_].trackerState = ACTIVE;
        list_[age_].visibility = 1.0;
        list_[age_].frameNum = frameNum_;
        list_size_ = age_ + 1;
    }
    else
    {
        // 过期的跟踪数据移除，更新为新的
        for (int i = 0; i < 29; i++)
        {
            list_[i].age = list_[i + 1].age;
            list_[i].tBbox.left = list_[i + 1].tBbox.left;
            list_[i].tBbox.top = list_[i + 1].tBbox.top;
            list_[i].tBbox.width = list_[i + 1].tBbox.width;
            list_[i].tBbox.height = list_[i + 1].tBbox.height;
            list_[i].confidence = list_[i + 1].confidence;
            list_[i].trackerState = ACTIVE;
            list_[i].visibility = 1.0;
            list_[i].frameNum = frameNum_;
        }
        list_[29].age = 30;
        list_[29].tBbox.left = trackInfo_.bbox.box.x0;
        list_[29].tBbox.top = trackInfo_.bbox.box.y0;
        list_[29].tBbox.width = trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0;
        list_[29].tBbox.height = trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0;
        list_[29].confidence = trackInfo_.bbox.score;
        list_[29].trackerState = ACTIVE;
        list_[29].visibility = 1.0;
        list_[29].frameNum = frameNum_;
        list_size_ = 30;
    }

    trackInfo_.age = age_++;
    trackInfo_.trackId = trackId_;
    trackInfo_.miss = miss_;

    return trackInfo_;
}

void DeepTracker::updatePastFrameObjBatch(NvDsTargetMiscDataBatch *pastFrameObjBatch)
{
    if (pastFrameObjBatch != nullptr)
    {
        // pastFrameObjBatch->list，一个流对应一个list
        if (pastFrameObjBatch->list != nullptr)
        {
            // 目前只用一个流所以pastFrameObjBatch_->list[0]
            // pastFrameObjBatch->list[0].list表示一个流对应的所有目标，默认分配了512个目标内存
            if (pastFrameObjBatch->list[0].list != nullptr)
            {
                // pastFrameObjBatch->list[0].list[0].list表示一个跟踪目标的历史帧数据
                pastFrameObjBatch->list[0].list[0].list = list_;
                pastFrameObjBatch->list[0].list[0].numObj = list_size_;
                pastFrameObjBatch->list[0].list[0].classId = objectToTrack_->classId;
                pastFrameObjBatch->list[0].list[0].uniqueId = trackId_;
                pastFrameObjBatch->list[0].list[0].numAllocated = 30;
            }

            pastFrameObjBatch->list[0].numFilled = 1;
            pastFrameObjBatch->list[0].streamID = 0;
            pastFrameObjBatch->list[0].surfaceStreamID = 0;
        }
        pastFrameObjBatch->numFilled = 1;
    }

    pastFrameObjBatch = pastFrameObjBatch;
}
