#include "Tracker.h"
#include <iostream>

// 解析配置文件
int parseConfigFile(const char *pCustomConfigFilePath, TRACKER_CONFIG &trackerConfig)
{
    if (pCustomConfigFilePath == nullptr || strlen(pCustomConfigFilePath) == 0)
    {
        std::cerr << "Invalid custom config file path." << std::endl;
        return -1;
    }

    // 使用YAML库加载配置文件
    YAML::Node configyml = YAML::LoadFile(pCustomConfigFilePath);
    if (!configyml)
    {
        std::cerr << "Failed to load config file: " << pCustomConfigFilePath << std::endl;
        return -1;
    }

    std::string key;
    // 解析配置文件内容
    if (!configyml["BaseConfig"])
    {
        std::cerr << "BaseConfig section not found in config file." << std::endl;
        return -1;
    }
    for (YAML::const_iterator itr = configyml["BaseConfig"].begin();
         itr != configyml["BaseConfig"].end(); ++itr)
    {
        key = itr->first.as<std::string>();
        if (key == "modelName")
        {
            trackerConfig.modelName = static_cast<MODEL_NAME>(itr->second.as<int>());
            if (trackerConfig.modelName < MODEL_SUTRACK || trackerConfig.modelName > MODEL_MIXFORMERV2)
            {
                std::cerr << "Invalid modelName in config file, set to default SUTRACK" << std::endl;
                trackerConfig.modelName = MODEL_SUTRACK; // 默认设置为 SUTRACK
            }
        }
        else if (key == "modelType")
        {
            trackerConfig.modelType = itr->second.as<uint8_t>();
            if (trackerConfig.modelType != 0 && trackerConfig.modelType != 1)
            {
                // 强制要求模型类型为0或1
                trackerConfig.modelType = 0; // 默认设置为 FP32
                std::cerr << "Invalid modelType in config file, set to default FP32" << std::endl;
            }
        }
        else if (key == "modelRootPath")
        {
            // 这里可以添加对模型路径的处理
            trackerConfig.modelRootPath = itr->second.as<std::string>();
        }
        else
        {
            std::cerr << "Unknown key in config file: " << key << std::endl;
        }
    }

    if (!configyml["TargetManagement"])
    {
        std::cerr << "TargetManagement section not found in config file." << std::endl;
        return -1;
    }

    for (YAML::const_iterator itr = configyml["TargetManagement"].begin();
         itr != configyml["TargetManagement"].end(); ++itr)
    {
        key = itr->first.as<std::string>();
        if (key == "expandFactor")
        {
            trackerConfig.targetManagement.expandFactor = itr->second.as<float>();
            if (trackerConfig.targetManagement.expandFactor <= 0.0f)
            {
                // 强制要求膨胀因子大于等于1
                trackerConfig.targetManagement.expandFactor = 1.0f;
                std::cerr << "Invalid expandFactor in config file, set to default 1.0" << std::endl;
            }
        }
        else if (key == "probationAge")
        {
            trackerConfig.targetManagement.probationAge = itr->second.as<uint16_t>();
            if (trackerConfig.targetManagement.probationAge < 1)
            {
                // 强制要求 probationAge 大于等于1
                trackerConfig.targetManagement.probationAge = 1;
                std::cerr << "Invalid probationAge in config file, set to default 1" << std::endl;
            }
        }
        else if (key == "maxMiss")
        {
            trackerConfig.targetManagement.maxMiss = itr->second.as<uint16_t>();
            if (trackerConfig.targetManagement.maxMiss < 0)
            {
                trackerConfig.targetManagement.maxMiss = 0;
                std::cerr << "Invalid maxMiss in config file, set to default 0" << std::endl;
            }
        }
        else if (key == "scoreThreshold")
        {
            trackerConfig.targetManagement.scoreThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.scoreThreshold < 0.0f || trackerConfig.targetManagement.scoreThreshold > 1.0f)
            {
                trackerConfig.targetManagement.scoreThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid scoreThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "iouThreshold")
        {
            trackerConfig.targetManagement.iouThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.iouThreshold < 0.0f || trackerConfig.targetManagement.iouThreshold > 1.0f)
            {
                trackerConfig.targetManagement.iouThreshold = 0.5f; // 默认设置为0.5
                std::cerr << "Invalid iouThreshold in config file, set to default 0.5" << std::endl;
            }
        }
        else if (key == "trackBoxWidthThreshold")
        {
            trackerConfig.targetManagement.trackBoxWidthThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.trackBoxWidthThreshold < 0.0f || trackerConfig.targetManagement.trackBoxWidthThreshold > 1.0f)
            {
                trackerConfig.targetManagement.trackBoxWidthThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid trackBoxWidthThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "trackBoxHeightThreshold")
        {
            trackerConfig.targetManagement.trackBoxHeightThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.trackBoxHeightThreshold < 0.0f || trackerConfig.targetManagement.trackBoxHeightThreshold > 1.0f)
            {
                trackerConfig.targetManagement.trackBoxHeightThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid trackBoxHeightThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "maxTrackAge")
        {
            trackerConfig.targetManagement.maxTrackAge = itr->second.as<uint32_t>();
            if (trackerConfig.targetManagement.maxTrackAge < 1)
            {
                trackerConfig.targetManagement.maxTrackAge = 1; // 默认设置为1
                std::cerr << "Invalid maxTrackAge in config file, set to default 1" << std::endl;
            }
        }
        else
        {
            std::cerr << "Unknown key in TargetManagement: " << key << std::endl;
        }
    }

    return 0;
}

NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery)
{
    TRACKER_CONFIG trackerConfig;
    // 解析自定义配置文件
    if (parseConfigFile(pCustomConfigFilePath, trackerConfig) != 0)
    {
        std::cerr << "Failed to parse custom config file: " << pCustomConfigFilePath << std::endl;
        return NvMOTStatus_Error;
    }

    /**  所有自定义跟踪器的必需配置。 */
    pQuery->computeConfig = NVMOTCOMP_CPU; // among {NVMOTCOMP_GPU, NVMOTCOMP_CPU}
    pQuery->numTransforms = 1;             // 0 for IOU and NvSORT tracker, 1 for NvDCF or NvDeepSORT tracker as they require the video frames
    pQuery->memType = NVBUF_MEM_CUDA_UNIFIED;
    pQuery->batchMode = NvMOTBatchMode_Batch;          // batchMode must be set as NvMOTBatchMode_Batch
    pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA; // among {NVBUF_COLOR_FORMAT_NV12, NVBUF_COLOR_FORMAT_RGBA}
    pQuery->supportPastFrame = true;

    pQuery->maxTargetsPerStream = MAX_TARGETS_PER_STREAM; // Max number of targets stored for each stream

    /** 可选配置以设置其他功能。 */
    pQuery->maxShadowTrackingAge = trackerConfig.targetManagement.maxTrackAge; // 如果 supportPastFrame 为 true，则需要跟踪阴影的最大长度
    pQuery->outputReidTensor = false;  // 仅当低级跟踪器支持输出 reid 特性时设置为 true
    pQuery->reidFeatureSize = 256;     // Re-ID特征的大小，如果outputReidTensor为true，则为必需

    std::cout << "[Track Initialized]" << std::endl;
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
    if (contextHandle == nullptr)
    {
        return;
    }
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
