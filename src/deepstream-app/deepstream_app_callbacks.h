#ifndef DEEPSTREAM_APP_CALLBACKS_H
#define DEEPSTREAM_APP_CALLBACKS_H

#include <gst/gst.h>
#include "deepstream_app.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief UDP 源 pad probe 回调，用于统计和打印缓冲区信息。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param user_data 用户数据。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn udpsrc_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

/**
 * @brief 控制通道 appsink 回调，用于接收并解析控制 JSON。
 *
 * @param sink appsink 元素。
 * @param appCtx 应用上下文。
 * @return GstFlowReturn 回调处理结果。
 */
GstFlowReturn on_control_data(GstElement *sink, AppCtx *appCtx);

/**
 * @brief 传感器流新增回调。
 *
 * @param appCtx 应用上下文。
 * @param sensorInfo 传感器信息。
 */
void s_sensor_info_callback_stream_added(AppCtx *appCtx, NvDsSensorInfo *sensorInfo);

/**
 * @brief 传感器流移除回调。
 *
 * @param appCtx 应用上下文。
 * @param sensorInfo 传感器信息。
 */
void s_sensor_info_callback_stream_removed(AppCtx *appCtx, NvDsSensorInfo *sensorInfo);

/**
 * @brief FPS 传感器流新增回调。
 *
 * @param appCtx 应用上下文。
 * @param sensorInfo FPS 传感器信息。
 */
void s_fps_sensor_info_callback_stream_added(AppCtx *appCtx, NvDsFPSSensorInfo *sensorInfo);

/**
 * @brief FPS 传感器流移除回调。
 *
 * @param appCtx 应用上下文。
 * @param sensorInfo FPS 传感器信息。
 */
void s_fps_sensor_info_callback_stream_removed(AppCtx *appCtx, NvDsFPSSensorInfo *sensorInfo);

/**
 * @brief 总线回调，用于处理 GStreamer 消息。
 *
 * @param bus 总线对象。
 * @param message 消息对象。
 * @param data 用户数据。
 * @return gboolean 是否继续监听总线。
 */
gboolean bus_callback(GstBus *bus, GstMessage *message, gpointer data);

/**
 * @brief 解析云端下发的 JSON 消息。
 *
 * @param appCtx 应用上下文。
 * @param data 消息数据。
 * @param size 消息大小。
 */
void parse_cloud_message(AppCtx *appCtx, gpointer data, guint size);

/**
 * @brief 推理完成回调，用于修改或统计元数据。
 *
 * @param appCtx 应用上下文。
 * @param buf 缓冲区。
 * @param batch_meta 批量元数据。
 * @param index 实例索引。
 */
void all_bbox_generated(AppCtx *appCtx, GstBuffer *buf,
                        NvDsBatchMeta *batch_meta, guint index);

/**
 * @brief 性能统计回调，用于打印每路 FPS。
 *
 * @param context 回调上下文，通常为 AppCtx。
 * @param str 性能统计结构体。
 */
void perf_cb(gpointer context, NvDsAppPerfStruct *str);

/**
 * @brief 消息订阅回调，用于处理云端下发消息。
 *
 * @param status 回调状态。
 * @param msg 消息指针。
 * @param msglen 消息长度。
 * @param topic 主题名称。
 * @param user_ptr 用户数据。
 */
void my_msg_broker_subscribe_cb(NvMsgBrokerErrorType status, void *msg,
                               int msglen, char *topic, void *user_ptr);

/**
 * @brief FPS 互斥锁，供性能回调与主线程共享。
 */
extern GMutex fps_lock;

/**
 * @brief 当前 FPS 数组，按源索引存放。
 */
extern gdouble fps[MAX_SOURCE_BINS];

/**
 * @brief 平均 FPS 数组，按源索引存放。
 */
extern gdouble fps_avg[MAX_SOURCE_BINS];

/**
 * @brief 实例数量（多路配置时使用）。
 */
extern guint num_instances;

#ifdef __cplusplus
}
#endif

#endif
