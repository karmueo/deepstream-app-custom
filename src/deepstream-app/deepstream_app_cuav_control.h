/**
 * @file deepstream_app_cuav_control.h
 * @brief C-UAV自动控制模块对外接口声明。
 *
 * 声明C-UAV控制发送元素创建、协议回调注册、启动测试报文发送、
 * 以及基于检测/跟踪结果的自动控制入口。
 */

#ifndef DEEPSTREAM_APP_CUAV_CONTROL_H
#define DEEPSTREAM_APP_CUAV_CONTROL_H

#include <gst/gst.h>
#include "deepstream_app.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建并初始化C-UAV控制发送元素。
 *
 * @param config DeepStream应用配置。
 * @param pipeline DeepStream流水线结构。
 * @return 创建成功或未配置C-UAV控制sink时返回TRUE，失败返回FALSE。
 */
gboolean create_cuav_control_element(NvDsConfig *config, NvDsPipeline *pipeline);

/**
 * @brief 为udpjsonmeta元素注册C-UAV协议解析回调。
 *
 * @param udpjsonmeta udpjsonmeta GStreamer元素。
 * @param appCtx 应用上下文，用作回调用户数据。
 */
void register_cuav_udpjson_callbacks(GstElement *udpjsonmeta, AppCtx *appCtx);

/**
 * @brief 根据配置发送C-UAV启动测试报文。
 *
 * @param appCtx 应用上下文。
 * @return 全部测试报文发送成功返回TRUE；未启用测试发送或发送失败返回FALSE。
 */
gboolean send_cuav_test_messages(AppCtx *appCtx);

/**
 * @brief 基于当前批量元数据执行C-UAV自动控制。
 *
 * @param appCtx 应用上下文。
 * @param batch_meta DeepStream批量元数据。
 */
void process_cuav_auto_control(AppCtx *appCtx, NvDsBatchMeta *batch_meta);

#ifdef __cplusplus
}
#endif

#endif /* DEEPSTREAM_APP_CUAV_CONTROL_H */
