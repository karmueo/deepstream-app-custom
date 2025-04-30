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
    // mixformer_ = std::make_shared<MixformerTRT>(engine_name);
    tracker_ = std::make_shared<DeepTracker>("/workspace/deepstream-app-custom/src/Mixformer_plugin/models/sutrack_fp32.engine");
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

        TrackInfo trackInfo;
        trackInfo = tracker_->update(bgraFrame, &frame->objectsIn, frame->frameNum);

        NvMOTTrackedObj *trackedObjs = trackedObjList->list;
        // 单目标跟踪，所以只要第一个目标
        NvMOTTrackedObj *trackedObj = &trackedObjs[0];

        // 如果跟踪上了，给NvMOTTrackedObj赋值
        if (tracker_->isTracked())
        {
            NvMOTRect motRect{
                static_cast<float>(trackInfo.bbox.box.x0),
                static_cast<float>(trackInfo.bbox.box.y0),
                static_cast<float>(trackInfo.bbox.box.x1 - trackInfo.bbox.box.x0),
                static_cast<float>(trackInfo.bbox.box.y1 - trackInfo.bbox.box.y0)};

            // 更新跟踪对象信息
            trackedObj->classId = trackInfo.bbox.class_id;
            trackedObj->trackingId = trackInfo.trackId;
            trackedObj->bbox = motRect;
            trackedObj->confidence = trackInfo.bbox.score;
            trackedObj->age = trackInfo.age;
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
    /* std::set<NvMOTStreamId> videoStreamIdList;
    for (NvMOTStreamId streamInd = 0; streamInd < params->numFrames; streamInd++)
    {
        videoStreamIdList.insert(params->frameList[streamInd].streamID);
    }

    for (NvMOTStreamId streamInd = 0; streamInd < params->numFrames; streamInd++)
    {
        if (pTrackerMiscData && pTrackerMiscData->pPastFrameObjBatch)
        {
            tracker_->updatePastFrameObjBatch(pTrackerMiscData->pPastFrameObjBatch);
        }
    } */

    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
    return NvMOTStatus_OK;
}
