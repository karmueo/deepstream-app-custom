/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_app_callbacks.h"
#include <gst/app/gstappsink.h>
#include <json-glib/json-glib.h>
#include <string.h>
#include "nvdsmeta_schema.h"

GST_DEBUG_CATEGORY_EXTERN(NVDS_APP);

static NvDsSensorInfo *s_sensor_info_create(NvDsSensorInfo *sensor_info);
static void s_sensor_info_destroy(NvDsSensorInfo *sensor_info);
static NvDsFPSSensorInfo *s_fps_sensor_info_create(NvDsFPSSensorInfo *sensor_info);
static void s_fps_sensor_info_destroy(NvDsFPSSensorInfo *sensor_info);
static NvDsFPSSensorInfo *get_fps_sensor_info(AppCtx *appCtx, guint source_id);

/* 调试: udpsrc probe 回调，统计 buffer */
GstPadProbeReturn udpsrc_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    static guint64 cnt = 0;
    if (info->type & GST_PAD_PROBE_TYPE_BUFFER)
    {
        GstBuffer *b = GST_BUFFER(info->data);
        gsize      sz = 0;
        GstMapInfo map;
        if (gst_buffer_map(b, &map, GST_MAP_READ))
        {
            sz = map.size;
            gst_buffer_unmap(b, &map);
        }
        if (G_UNLIKELY(cnt < 20 || (cnt < 1000 && (cnt % 50) == 0) || (cnt % 500) == 0))
        {
            GstCaps *caps = gst_pad_get_current_caps(pad);
            gchar   *caps_str = caps ? gst_caps_to_string(caps) : g_strdup("(null)");
            g_print("[udpsrc-multicast][probe] #%" G_GUINT64_FORMAT " size=%zu caps=%s\n", ++cnt, sz, caps_str);
            g_free(caps_str);
            if (caps)
                gst_caps_unref(caps);
        }
        else
        {
            cnt++;
        }
    }
    return GST_PAD_PROBE_OK;
}

GstFlowReturn on_control_data(GstElement *sink, AppCtx *appCtx)
{
    /* // 控制速率,比如每秒处理30次
    static GTimer *timer = NULL;
    static gdouble last_time = 0;
    static int count = 0;
    if (!timer) timer = g_timer_new();
    gdouble now = g_timer_elapsed(timer, NULL);
    // 只处理每秒30次
    if (now - last_time < 1.0/30.0) {
        GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (sample) gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    last_time = now; */

    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;

    if (gst_buffer_map(buffer, &map, GST_MAP_READ))
    {
        // TODO:解析JSON,根据实际JSON进行解析
        JsonParser *parser = json_parser_new();
        if (json_parser_load_from_data(parser, (gchar *)map.data, -1, NULL))
        {
            JsonNode *root = json_parser_get_root(parser);
            if (JSON_NODE_HOLDS_OBJECT(root))
            {
                JsonObject *root_obj = json_node_get_object(root);

                // 检查是否为重连指令
                if (json_object_has_member(root_obj, "command"))
                {
                    const gchar *cmd = json_object_get_string_member(root_obj, "command");
                    if (g_strcmp0(cmd, "reconnect_rtsp") == 0)
                    {
                        // 构造重连消息并发送到总线
                        guint        source_id = json_object_get_int_member(root_obj, "source_id");
                        const gchar *new_uri =
                            json_object_has_member(root_obj, "new_uri") ?
                                json_object_get_string_member(root_obj, "new_uri") :
                                NULL;

                        GstStructure *s = gst_structure_new("reconnect-rtsp",
                                                            "source-id", G_TYPE_UINT, source_id,
                                                            "new-uri", G_TYPE_STRING, new_uri,
                                                            NULL);

                        GstBus     *bus = gst_pipeline_get_bus(GST_PIPELINE(appCtx->pipeline.pipeline));
                        GstMessage *msg = gst_message_new_application(GST_OBJECT(sink), s);
                        gst_bus_post(bus, msg);
                        gst_object_unref(bus);
                    }
                }

                // 解析timestamp
                if (json_object_has_member(root_obj, "timestamp"))
                {
                    guint64 timestamp = json_object_get_double_member(root_obj, "timestamp");
                    if (!appCtx->custom_msg_data)
                    {
                        appCtx->custom_msg_data = g_malloc0(sizeof(CustomMessageData));
                    }
                    appCtx->custom_msg_data->timestamp = timestamp;
                    // 这里可以打印或处理时间戳
                    // g_print("Timestamp: %.6f\n", timestamp);
                }

                // 解析message
                if (json_object_has_member(root_obj, "message"))
                {
                    const gchar *message = json_object_get_string_member(root_obj, "message");
                    // g_print("Message: %s\n", message);
                }

                // 解析sensor_data对象
                if (json_object_has_member(root_obj, "sensor_data"))
                {
                    JsonObject *sensor_data = json_object_get_object_member(root_obj, "sensor_data");

                    if (json_object_has_member(sensor_data, "temperature"))
                    {
                        gdouble temperature = json_object_get_double_member(sensor_data, "temperature");
                        // g_print("Temperature: %.1f\n", temperature);
                    }

                    if (json_object_has_member(sensor_data, "humidity"))
                    {
                        gint humidity = json_object_get_int_member(sensor_data, "humidity");
                        // g_print("Humidity: %d%%\n", humidity);
                    }
                }
            }
        }
        g_object_unref(parser);
        gst_buffer_unmap(buffer, &map);

        // NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
        // // 构造 user meta
        // NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(appCtx->last_batch_meta);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static NvDsSensorInfo *s_sensor_info_create(NvDsSensorInfo *sensor_info)
{
    NvDsSensorInfo *sensorInfoToHash = (NvDsSensorInfo *)g_malloc0(sizeof(NvDsSensorInfo));
    *sensorInfoToHash = *sensor_info;
    sensorInfoToHash->sensor_id = (gchar const *)g_strdup(sensor_info->sensor_id);
    sensorInfoToHash->sensor_name = (gchar const *)g_strdup(sensor_info->sensor_name);
    sensorInfoToHash->uri = (gchar const *)g_strdup(sensor_info->uri);
    return sensorInfoToHash;
}

static void s_sensor_info_destroy(NvDsSensorInfo *sensor_info)
{
    if (!sensor_info)
        return;
    if (sensor_info->sensor_id)
    {
        g_free((void *)sensor_info->sensor_id);
    }
    if (sensor_info->sensor_name)
    {
        g_free((void *)sensor_info->sensor_name);
    }

    g_free(sensor_info);
}

void s_sensor_info_callback_stream_added(AppCtx *appCtx, NvDsSensorInfo *sensorInfo)
{
    NvDsSensorInfo *sensorInfoToHash = s_sensor_info_create(sensorInfo);
    /** save the sensor info into the hash map */
    g_hash_table_insert(appCtx->sensorInfoHash, sensorInfo->source_id + (char *)NULL, sensorInfoToHash);
}

void s_sensor_info_callback_stream_removed(AppCtx *appCtx, NvDsSensorInfo *sensorInfo)
{
    NvDsSensorInfo *sensorInfoFromHash = get_sensor_info(appCtx, sensorInfo->source_id);
    /** remove the sensor info from the hash map */
    if (sensorInfoFromHash)
    {
        g_hash_table_remove(appCtx->sensorInfoHash, sensorInfo->source_id + (gchar *)NULL);
        s_sensor_info_destroy(sensorInfoFromHash);
    }
}

/*Note: Below callbacks/functions defined for FPS logging,
 *  when nvmultiurisrcbin is being used*/
static NvDsFPSSensorInfo *s_fps_sensor_info_create(NvDsFPSSensorInfo *sensor_info)
{
    NvDsFPSSensorInfo *fpssensorInfoToHash = (NvDsFPSSensorInfo *)g_malloc0(sizeof(NvDsFPSSensorInfo));
    *fpssensorInfoToHash = *sensor_info;
    fpssensorInfoToHash->uri = (gchar const *)g_strdup(sensor_info->uri);
    fpssensorInfoToHash->source_id = sensor_info->source_id;
    fpssensorInfoToHash->sensor_id = (gchar const *)g_strdup(sensor_info->sensor_id);
    fpssensorInfoToHash->sensor_name = (gchar const *)g_strdup(sensor_info->sensor_name);
    return fpssensorInfoToHash;
}

static void s_fps_sensor_info_destroy(NvDsFPSSensorInfo *sensor_info)
{
    if (!sensor_info)
        return;
    if (sensor_info->sensor_id)
    {
        g_free((void *)sensor_info->sensor_id);
    }
    if (sensor_info->sensor_name)
    {
        g_free((void *)sensor_info->sensor_name);
    }
    if (sensor_info->uri)
    {
        g_free((void *)sensor_info->uri);
    }

    g_free(sensor_info);
}

static NvDsFPSSensorInfo *get_fps_sensor_info(AppCtx *appCtx, guint source_id)
{
    NvDsFPSSensorInfo *sensorInfo = (NvDsFPSSensorInfo *)g_hash_table_lookup(appCtx->perf_struct.FPSInfoHash,
                                                                             GUINT_TO_POINTER(source_id));
    return sensorInfo;
}

void s_fps_sensor_info_callback_stream_added(AppCtx *appCtx, NvDsFPSSensorInfo *sensorInfo)
{
    NvDsFPSSensorInfo *fpssensorInfoToHash = s_fps_sensor_info_create(sensorInfo);
    /** save the sensor info into the hash map */
    g_hash_table_insert(appCtx->perf_struct.FPSInfoHash, GUINT_TO_POINTER(sensorInfo->source_id), fpssensorInfoToHash);
}

void s_fps_sensor_info_callback_stream_removed(AppCtx *appCtx, NvDsFPSSensorInfo *sensorInfo)
{
    NvDsFPSSensorInfo *fpsensorInfoFromHash = get_fps_sensor_info(appCtx, sensorInfo->source_id);
    /** remove the sensor info from the hash map */
    if (fpsensorInfoFromHash)
    {
        g_hash_table_remove(appCtx->perf_struct.FPSInfoHash, GUINT_TO_POINTER(sensorInfo->source_id));
        s_fps_sensor_info_destroy(fpsensorInfoFromHash);
    }
}

/**
 * callback function to receive messages from components
 * in the pipeline.
 */
gboolean bus_callback(GstBus *bus, GstMessage *message, gpointer data)
{
    AppCtx *appCtx = (AppCtx *)data;
    GST_CAT_DEBUG(NVDS_APP,
                  "Received message on bus: source %s, msg_type %s",
                  GST_MESSAGE_SRC_NAME(message), GST_MESSAGE_TYPE_NAME(message));
    switch (GST_MESSAGE_TYPE(message))
    {
    case GST_MESSAGE_INFO:
    {
        GError *error = NULL;
        gchar  *debuginfo = NULL;
        gst_message_parse_info(message, &error, &debuginfo);
        g_printerr("INFO from %s: %s\n",
                   GST_OBJECT_NAME(message->src), error->message);
        if (debuginfo)
        {
            g_printerr("Debug info: %s\n", debuginfo);
        }
        g_error_free(error);
        g_free(debuginfo);
        break;
    }
    case GST_MESSAGE_WARNING:
    {
        GError *error = NULL;
        gchar  *debuginfo = NULL;
        gst_message_parse_warning(message, &error, &debuginfo);
        g_printerr("WARNING from %s: %s\n",
                   GST_OBJECT_NAME(message->src), error->message);
        if (debuginfo)
        {
            g_printerr("Debug info: %s\n", debuginfo);
        }
        g_error_free(error);
        g_free(debuginfo);
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        GError      *error = NULL;
        gchar       *debuginfo = NULL;
        const gchar *attempts_error =
            "Reconnection attempts exceeded for all sources or EOS received.";
        guint        i = 0;
        gst_message_parse_error(message, &error, &debuginfo);

        if (strstr(error->message, attempts_error))
        {
            g_print("Reconnection attempt  exceeded or EOS received for all sources."
                    " Exiting.\n");
            g_error_free(error);
            g_free(debuginfo);
            appCtx->return_value = 0;
            appCtx->quit = TRUE;
            return TRUE;
        }

        g_printerr("ERROR from %s: %s\n",
                   GST_OBJECT_NAME(message->src), error->message);
        if (debuginfo)
        {
            g_printerr("Debug info: %s\n", debuginfo);
        }

        NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
        GstElement       *msg_src_elem = (GstElement *)GST_MESSAGE_SRC(message);
        gboolean          bin_found = FALSE;
        /* Find the source bin which generated the error. */
        while (msg_src_elem && !bin_found)
        {
            for (i = 0; i < bin->num_bins && !bin_found; i++)
            {
                if (bin->sub_bins[i].src_elem == msg_src_elem ||
                    bin->sub_bins[i].bin == msg_src_elem)
                {
                    bin_found = TRUE;
                    break;
                }
            }
            msg_src_elem = GST_ELEMENT_PARENT(msg_src_elem);
        }

        if ((i != bin->num_bins) &&
            (appCtx->config.multi_source_config[0].type == NV_DS_SOURCE_RTSP))
        {
            // Error from one of RTSP source.
            NvDsSrcBin *subBin = &bin->sub_bins[i];
            // 如果配置了无限重连（-1），则不要退出，直接触发重连
            if (subBin->config &&
                subBin->config->rtsp_reconnect_attempts == -1)
            {
                if (!subBin->reconfiguring)
                {
                    subBin->reconfiguring = TRUE;
                    g_timeout_add(0, reset_source_pipeline, subBin);
                }
                g_error_free(error);
                g_free(debuginfo);
                return TRUE;
            }

            // 默认行为：退出
            g_error_free(error);
            g_free(debuginfo);
            appCtx->return_value = 0;
            appCtx->quit = TRUE;
            return TRUE;
        }

        if (appCtx->config.multi_source_config[0].type ==
            NV_DS_SOURCE_CAMERA_V4L2)
        {
            if (g_strrstr(debuginfo, "reason not-negotiated (-4)"))
            {
                NVGSTDS_INFO_MSG_V("incorrect camera parameters provided, please provide supported resolution and frame rate\n");
            }

            if (g_strrstr(debuginfo, "Buffer pool activation failed"))
            {
                NVGSTDS_INFO_MSG_V("usb bandwidth might be saturated\n");
            }
        }

        g_error_free(error);
        g_free(debuginfo);
        appCtx->return_value = -1;
        appCtx->quit = TRUE;
        break;
    }
    case GST_MESSAGE_STATE_CHANGED:
    {
        GstState oldstate, newstate;
        gst_message_parse_state_changed(message, &oldstate, &newstate, NULL);
        if (GST_ELEMENT(GST_MESSAGE_SRC(message)) == appCtx->pipeline.pipeline)
        {
            switch (newstate)
            {
            case GST_STATE_PLAYING:
                NVGSTDS_INFO_MSG_V("Pipeline running\n");
                GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(appCtx->pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-playing");
                break;
            case GST_STATE_PAUSED:
                if (oldstate == GST_STATE_PLAYING)
                {
                    NVGSTDS_INFO_MSG_V("Pipeline paused\n");
                }
                break;
            case GST_STATE_READY:
                NVGSTDS_INFO_MSG_V("Pipeline ready\n");
                break;
            case GST_STATE_NULL:
                NVGSTDS_INFO_MSG_V("Pipeline stopped\n");
                break;
            default:
                break;
            }
        }
        break;
    }
    case GST_MESSAGE_EOS:
    {
        appCtx->eos_received = TRUE;
        if (appCtx->config.multi_source_config[0].type == NV_DS_SOURCE_RTSP)
        {
            NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
            for (guint i = 0; i < bin->num_bins; i++)
            {
                if (bin->sub_bins[i].config &&
                    bin->sub_bins[i].config->rtsp_reconnect_attempts == -1)
                {
                    if (!bin->sub_bins[i].reconfiguring)
                    {
                        bin->sub_bins[i].reconfiguring = TRUE;
                        g_timeout_add(0, reset_source_pipeline, &bin->sub_bins[i]);
                    }
                    return TRUE;
                }
            }
        }
        g_print("Received EOS. Exiting ...\n");
        appCtx->quit = TRUE;
        break;
    }
    case GST_MESSAGE_ELEMENT:
    {
        if (gst_nvmessage_is_stream_add(message))
        {
            g_mutex_lock(&(appCtx->perf_struct).struct_lock);

            appCtx->config.num_source_sub_bins++;
            NvDsSensorInfo sensorInfo = {0};
            gst_nvmessage_parse_stream_add(message, &sensorInfo);
            g_print("new stream added [%d:%s:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id, sensorInfo.sensor_name);
            /** Callback */
            s_sensor_info_callback_stream_added(appCtx, &sensorInfo);
            gboolean is_rtsp = g_str_has_prefix(sensorInfo.uri, "rtsp://");
            gboolean is_ipc = g_str_has_prefix(sensorInfo.uri, "ipc://");
            appCtx->config.multi_source_config[sensorInfo.source_id].uri = g_strdup(sensorInfo.uri);
            if (is_rtsp)
            {
                appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_RTSP;
            }
            else if (is_ipc)
            {
                appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_IPC;
            }
            else
            {
                appCtx->config.multi_source_config[sensorInfo.source_id].type = NV_DS_SOURCE_URI;
            }
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(appCtx->pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                                              "ds-app-added");
            NvDsFPSSensorInfo fpssensorInfo = {0};
            gst_nvmessage_parse_fps_stream_add(message, &fpssensorInfo);
            s_fps_sensor_info_callback_stream_added(appCtx, &fpssensorInfo);

            g_mutex_unlock(&(appCtx->perf_struct).struct_lock);
        }
        if (gst_nvmessage_is_stream_remove(message))
        {
            g_mutex_lock(&(appCtx->perf_struct).struct_lock);
            appCtx->config.num_source_sub_bins--;
            NvDsSensorInfo sensorInfo = {0};
            gst_nvmessage_parse_stream_remove(message, &sensorInfo);
            g_print("new stream removed [%d:%s]\n\n\n\n", sensorInfo.source_id, sensorInfo.sensor_id);
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(appCtx->pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                                              "ds-app-removed");
            /** Callback */
            s_sensor_info_callback_stream_removed(appCtx, &sensorInfo);
            NvDsFPSSensorInfo fpssensorInfo = {0};
            gst_nvmessage_parse_fps_stream_remove(message, &fpssensorInfo);
            s_fps_sensor_info_callback_stream_removed(appCtx, &fpssensorInfo);
            g_mutex_unlock(&(appCtx->perf_struct).struct_lock);
        }
        if (gst_nvmessage_is_reconnect_attempt_exceeded(message))
        {
            NvDsRtspAttemptsInfo rtsp_info = {0};
            gboolean             rec_attempt_exceeded_for_all = TRUE;
            if (gst_nvmessage_parse_reconnect_attempt_exceeded(message, &rtsp_info))
            {
                if (rtsp_info.attempt_exceeded)
                {
                    appCtx->config.multi_source_config[rtsp_info.source_id].rtsp_reconnect_attempt_exceeded =
                        rtsp_info.attempt_exceeded;
                    NVGSTDS_INFO_MSG_V("rtsp reconnect attempt exceeded for source_id : %d\n", rtsp_info.source_id);
                }

                if (appCtx->eos_received)
                {
                    for (int i = 0; i < MAX_SOURCE_BINS; i++)
                    {
                        if (appCtx->config.multi_source_config[i].type == NV_DS_SOURCE_RTSP)
                        {
                            if (appCtx->config.multi_source_config[i].rtsp_reconnect_attempt_exceeded != TRUE)
                            {
                                rec_attempt_exceeded_for_all = FALSE;
                            }
                        }
                    }

                    if (rec_attempt_exceeded_for_all)
                    {
                        NVGSTDS_INFO_MSG_V("Exiting ...\n");
                        appCtx->quit = TRUE;
                        return FALSE;
                    }
                }
            }
        }
        break;
    }
    // TODO: 根据解析的JSON报文进行实际的处理
    /* case GST_MESSAGE_APPLICATION:
    {
        const GstStructure *s = gst_message_get_structure(message);
        if (gst_structure_has_name(s, "reconnect-rtsp"))
        {
            guint source_id;
            const gchar *new_uri;
            gst_structure_get(s, "source-id", G_TYPE_UINT, &source_id,
                              "new-uri", G_TYPE_STRING, &new_uri, NULL);

            // 获取对应的RTSP源
            NvDsSrcBin *sub_bin = &appCtx->pipeline.multi_src_bin.sub_bins[source_id];

            // 更新URI（如果提供了新地址）
            if (new_uri && sub_bin->config->uri)
            {
                g_free(sub_bin->config->uri);
                sub_bin->config->uri = g_strdup(new_uri);
            }

            // 触发异步重连（避免阻塞总线线程）
            sub_bin->reconfiguring = TRUE;
            g_timeout_add(0, (GSourceFunc)reset_source_pipeline, sub_bin);
            appCtx->quit = TRUE;
        }
        break;
    } */
    default:
        break;
    }
    return TRUE;
}

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 * 所有推理（主要+次要）完成后调用的回调函数。
 * 在这是可以修改元数据内容。
 */
void all_bbox_generated(AppCtx *appCtx, GstBuffer *buf,
                        NvDsBatchMeta *batch_meta, guint index)
{
    // guint num_male = 0;
    // guint num_female = 0;
    // guint num_objects[128];
    // guint num_rects = 0; // 矩形数量

    // memset(num_objects, 0, sizeof(num_objects));
}

/**
 * callback function to print the performance numbers of each stream.
 */
void perf_cb(gpointer context, NvDsAppPerfStruct *str)
{
    static guint header_print_cnt = 0;
    guint        i;
    AppCtx      *appCtx = (AppCtx *)context;
    guint        numf = str->num_instances;

    g_mutex_lock(&fps_lock);
    for (i = 0; i < numf; i++)
    {
        fps[i] = str->fps[i];
        fps_avg[i] = str->fps_avg[i];
    }

    if (header_print_cnt % 20 == 0)
    {
        g_print("\n**PERF:  ");
        for (i = 0; i < numf; i++)
        {
            g_print("FPS %d (Avg)\t", i);
        }
        g_print("\n");
        header_print_cnt = 0;
    }
    header_print_cnt++;
    if (num_instances > 1)
        g_print("PERF(%d): ", appCtx->index);
    else
        g_print("**PERF:  ");

    for (i = 0; i < numf; i++)
    {
        g_print("%.2f (%.2f)\t", fps[i], fps_avg[i]);
    }
    g_print("\n");
    g_mutex_unlock(&fps_lock);
}

void my_msg_broker_subscribe_cb(NvMsgBrokerErrorType status, void *msg,
                               int msglen, char *topic, void *user_ptr)
{
    // 判断topic是否为事件消息主题
    if (strcmp(topic, "command") != 0)
    {
        status = NV_MSGBROKER_API_NOT_SUPPORTED;
        return;
    }

    if (msg && msglen > 0)
    {
        AppCtx           *appCtx = (AppCtx *)user_ptr;

        parse_cloud_message(appCtx, msg, msglen);
        status = NV_MSGBROKER_API_OK;
    }
    else
    {
        status = NV_MSGBROKER_API_ERR;
    }
}
