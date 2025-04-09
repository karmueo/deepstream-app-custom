#include "Tracker.h"
#include <fstream>
#include <cuda_runtime.h>

static bool isValidTrackedObjList(const NvMOTTrackedObjList *trackedObjList)
{
    // 检查指针是否为空
    if (trackedObjList == nullptr)
    {
        std::cerr << "Error: trackedObjList is nullptr!" << std::endl;
        return false;
    }

    if (trackedObjList->list == nullptr)
    {
        std::cerr << "Error: trackedObjList->list is nullptr!" << std::endl;
        return false;
    }

    // 所有检查通过，对象有效
    return true;
}

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

NvMOTContext::NvMOTContext(const NvMOTConfig &configIn, NvMOTConfigResponse &configResponse)
{
    configResponse.summaryStatus = NvMOTConfigStatus_OK;

    // FIXME:
    std::string engine_name = "/workspaces/Deepstream_template/deepstream/triton_model/Tracking/1/mixformer_v2.engine";
    mixformer_ = std::make_shared<MixformerTRT>(engine_name);
}

NvMOTContext::~NvMOTContext()
{
}

NvMOTStatus NvMOTContext::processFrame(const NvMOTProcessParams *params, NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
    cv::Mat in_mat;

    if (!params || params->numFrames <= 0)
    {
        return NvMOTStatus_OK;
    }

    for (uint streamIdx = 0; streamIdx < pTrackedObjectsBatch->numFilled; streamIdx++)
    {
        NvMOTTrackedObjList *trackedObjList = &pTrackedObjectsBatch->list[streamIdx];
        if (isValidTrackedObjList(trackedObjList) == false)
        {
            continue;
        }

        NvMOTFrame *frame = &params->frameList[streamIdx];

        if (frame->bufferList[0] == nullptr)
        {
            std::cout << "frame->bufferList[0] is nullptr" << std::endl;
            continue;
        }

        if (trackedObjList->numAllocated != MAX_TARGETS_PER_STREAM)
        {
            // Reallocate memory space
            delete trackedObjList->list;
            trackedObjList->list = new NvMOTTrackedObj[MAX_TARGETS_PER_STREAM];
        }

        NvBufSurfaceParams *bufferParams = frame->bufferList[0];
        cv::Mat bgraFrame(bufferParams->height, bufferParams->width, CV_8UC4, bufferParams->dataPtr);

        // 保存图片到本地查看
        // cv::imwrite("in_mat.jpeg", bgraFrame);

        DrOBB bbox;
        if (is_tracked_ == false)
        {
            if (frame->objectsIn.numFilled == 0)
            {
                continue;
            }

            // cv::imwrite("out.jpeg", bgraFrame);

            // 遍历检测到的目标
            // 当前画面的中心坐标
            auto image_cx = bgraFrame.cols / 2.f;
            auto image_cy = bgraFrame.rows / 2.f;
            float min_distance = 1000000.f;
            for (uint32_t numObjects = 0; numObjects < frame->objectsIn.numFilled; numObjects++)
            {
                NvMOTObjToTrack obj = frame->objectsIn.list[numObjects];
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

            bbox.box.x0 = objectToTrack_->bbox.x;
            bbox.box.x1 = objectToTrack_->bbox.x + objectToTrack_->bbox.width;
            bbox.box.y0 = objectToTrack_->bbox.y;
            bbox.box.y1 = objectToTrack_->bbox.y + objectToTrack_->bbox.height;
            bbox.class_id = objectToTrack_->classId;
            mixformer_->init(bgraFrame, bbox);

            is_tracked_ = true;
        }
        else
        {
            // 更新跟踪器并获取跟踪结果
            bbox = mixformer_->track(bgraFrame);
            bool is_track_match_detect = true;

            if (bbox.score == 0 || bbox.box.x0 == 0 || bbox.box.y0 == 0)
            {
                miss_ = 100;
            }
            else
            {
                // 如果有检测结果，和检测结果对比来查看跟踪是否正确
                if (frame->objectsIn.numFilled != 0)
                {
                    is_track_match_detect = false;
                    // 计算和所有检测frame->objectsIn.list结果的IOU
                    cv::Rect trackRect = cv::Rect(bbox.box.x0,
                                                  bbox.box.y0,
                                                  bbox.box.x1 - bbox.box.x0,
                                                  bbox.box.y1 - bbox.box.y0);
                    for (uint i = 0; i < frame->objectsIn.numFilled; i++)
                    {
                        NvMOTObjToTrack obj = frame->objectsIn.list[i];
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
                if (bbox.score < 0.9 || !is_track_match_detect)
                {
                    miss_++;
                }
            }

            if (miss_ > 20)
            {
                // 如果跟踪分数小于阈值，或者IOU小于0.5，认为跟踪失败
                is_tracked_ = false;
                age_ = 0;
                trackId_++;
                miss_ = 0;
            }
            else
            {
                // 如果正常跟踪上了,给showRect_赋值
                showRect_.x = bbox.box.x0;
                showRect_.y = bbox.box.y0;
                showRect_.width = bbox.box.x1 - bbox.box.x0;
                showRect_.height = bbox.box.y1 - bbox.box.y0;

                /* if (trackedObjList->numAllocated != 1 && trackedObjList->list == nullptr)
                {
                    // Allocate memory space
                    trackedObjList->list = new NvMOTTrackedObj[1];
                    trackedObjList->numAllocated = 1;
                } */
            }
        }

        NvMOTTrackedObj *trackedObjs = trackedObjList->list;
        // 单目标跟踪，所以只要第一个目标
        NvMOTTrackedObj *trackedObj = &trackedObjs[0];

        // 如果跟踪上了，给NvMOTTrackedObj赋值
        if (is_tracked_)
        {
            NvMOTRect motRect{
                static_cast<float>(bbox.box.x0),
                static_cast<float>(bbox.box.y0),
                static_cast<float>(bbox.box.x1 - bbox.box.x0),
                static_cast<float>(bbox.box.y1 - bbox.box.y0)};

            // 更新跟踪对象信息
            trackedObj->classId = bbox.class_id;
            trackedObj->trackingId = trackId_;
            trackedObj->bbox = motRect;
            trackedObj->confidence = bbox.score;
            trackedObj->age = age_++;
            // trackedObj->associatedObjectIn = objectToTrack_;
            // trackedObj->associatedObjectIn->doTracking = true;

            // trackedObjList->streamID = frame->streamID;
            trackedObjList->frameNum = frame->frameNum;
            trackedObjList->valid = true;
            trackedObjList->list = trackedObjs;
            trackedObjList->numFilled = 1;
            trackedObjList->numAllocated = 1;
        }
        else
        {
            // 取消跟踪，清空对象列表
            // trackedObjList->streamID = frame->streamID;
            trackedObjList->frameNum = frame->frameNum;
            trackedObjList->numFilled = 0;
            trackedObjList->valid = false;
            trackedObjList->numAllocated = 1;
        }
    }

    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::retrieveMiscData(const NvMOTProcessParams *params,
                                           NvMOTTrackerMiscData *pTrackerMiscData)
{
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
    return NvMOTStatus_OK;
}
