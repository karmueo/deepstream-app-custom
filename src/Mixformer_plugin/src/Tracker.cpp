#include "Tracker.h"
#include <iostream>

NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery)
{
    /**
     * 用户可以解析 pCustomConfigFilePath 中的低级配置文件来检查
     * 低级跟踪器的要求
     */
    /** An optional function queryParams(NvMOTQuery&) can be implemented in context handle to fill query params. */
    /*
    if (pQuery->contextHandle)
    {
         pQuery->contextHandle->queryParams(*pQuery);
    }
    */

    /**  所有自定义跟踪器的必需配置。 */
    pQuery->computeConfig = NVMOTCOMP_CPU; // among {NVMOTCOMP_GPU, NVMOTCOMP_CPU}
    pQuery->numTransforms = 1;             // 0 for IOU and NvSORT tracker, 1 for NvDCF or NvDeepSORT tracker as they require the video frames
    pQuery->memType = NVBUF_MEM_CUDA_UNIFIED;
    pQuery->batchMode = NvMOTBatchMode_Batch;          // batchMode must be set as NvMOTBatchMode_Batch
    pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA; // among {NVBUF_COLOR_FORMAT_NV12, NVBUF_COLOR_FORMAT_RGBA}

    pQuery->maxTargetsPerStream = MAX_TARGETS_PER_STREAM; // Max number of targets stored for each stream

    /** 可选配置以设置其他功能。 */
    pQuery->supportPastFrame = false;  // 仅当低级跟踪器支持过去帧数据时设置为 true
    pQuery->maxShadowTrackingAge = 30; // 如果 supportPastFrame 为 true，则需要跟踪阴影的最大长度
    pQuery->outputReidTensor = false;  // 仅当低级跟踪器支持输出 reid 特性时设置为 true
    pQuery->reidFeatureSize = 256;     // Re-ID特征的大小，如果outputReidTensor为true，则为必需

    std::cout << "[BYTETrack Initialized]" << std::endl;
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse)
{
    if (pContextHandle != nullptr)
    {
        NvMOT_DeInit(*pContextHandle);
    }

    /// 用户定义的上下文类
    NvMOTContext *pContext = nullptr;

    /// 实例化用户定义的上下文
    pContext = new NvMOTContext(*pConfigIn, *pConfigResponse);

    /// 将指针作为上下文句柄传递
    *pContextHandle = pContext;

    return NvMOTStatus_OK;
}

void NvMOT_DeInit(NvMOTContextHandle contextHandle)
{
    /// 销毁上下文句柄
    delete contextHandle;
}

NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
    /// 使用上下文中的用户定义方法处理给定的视频帧，并生成输出
    return contextHandle->processFrame(pParams, pTrackedObjectsBatch);
}

NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle contextHandle,
                                   NvMOTProcessParams *pParams,
                                   NvMOTTrackerMiscData *pTrackerMiscData)
{
    /// 如果有，请检索过去帧的数据
    return contextHandle->retrieveMiscData(pParams, pTrackerMiscData);
}

NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle,
                                NvMOTStreamId streamIdMask)
{
    /// 从低级跟踪器上下文中删除指定的视频流
    return contextHandle->removeStream(streamIdMask);
}
