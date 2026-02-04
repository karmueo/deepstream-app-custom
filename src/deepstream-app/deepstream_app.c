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

#include <gst/gst.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "deepstream_app.h"
#include "deepstream_app_callbacks.h"
#include "deepstream_app_probes.h"
#include "nvds_obj_encode.h"
#include "gstudpjsonmeta.h"

GST_DEBUG_CATEGORY_EXTERN(NVDS_APP);

GQuark _dsmeta_quark;

#define CEIL(a, b) ((a + b - 1) / b)

/**
 * @brief  Add the (nvmsgconv->nvmsgbroker) sink-bin to the
 *         overall DS pipeline (if any configured) and link the same to
 *         common_elements.tee (This tee connects
 *         the common analytics path to Tiler/display-sink and
 *         to configured broker sink if any)
 *         NOTE: This API shall return TRUE if there are no
 *         broker sinks to add to pipeline
 *
 * @param  appCtx [IN]
 * @return TRUE if succussful; FALSE otherwise
 */
static gboolean add_and_link_broker_sink(AppCtx *appCtx);

/**
 * @brief  Checks if there are any [sink] groups
 *         configured for source_id=provided source_id
 *         NOTE: source_id key and this API is valid only when we
 *         disable [tiler] and thus use demuxer for individual
 *         stream out
 * @param  config [IN] The DS Pipeline configuration struct
 * @param  source_id [IN] Source ID for which a specific [sink]
 *         group is searched for
 */
static gboolean is_sink_available_for_source_id(NvDsConfig *config,
                                                guint source_id);

NvDsSensorInfo *get_sensor_info(AppCtx *appCtx, guint source_id)
{
    NvDsSensorInfo *sensorInfo = (NvDsSensorInfo *)g_hash_table_lookup(appCtx->sensorInfoHash,
                                                                       source_id + (gchar *)NULL);
    return sensorInfo;
}

static void on_cuav_guidance(const CUAVCommonHeader *header,
                             const CUAVGuidanceInfo *guidance,
                             gpointer user_data)
{
    (void)user_data;
    if (!header || !guidance)
        return;

    g_print("[cuav][guidance] msg_sn=%u time=%u-%02u-%02u %02u:%02u:%02u.%.0f "
            "tar_id=%u cat=%u stat=%u enu_a=%.2f enu_e=%.2f lon=%.6f lat=%.6f alt=%.2f\n",
            header->msg_sn,
            guidance->yr, guidance->mo, guidance->dy,
            guidance->h, guidance->min, guidance->sec, guidance->msec,
            guidance->tar_id, guidance->tar_category, guidance->guid_stat,
            guidance->enu_a, guidance->enu_e, guidance->lon, guidance->lat, guidance->alt);
}

/**
 * Function to dump object ReID embeddings to files when the tracker outputs
 * ReID embeddings into user meta. For this to work, property "reid-track-output-dir"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_reid_track_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    if (!appCtx->config.reid_track_dir_path)
        return;

    gchar reid_file[1024] = {0};
    FILE *reid_params_dump_file = NULL;

    /** Save the reid embedding for each frame. */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        /** Create dump file name. */
        guint stream_id = frame_meta->pad_index;
        g_snprintf(reid_file, sizeof(reid_file) - 1,
                   "%s/%02u_%03u_%06lu.txt", appCtx->config.reid_track_dir_path,
                   appCtx->index, stream_id, (gulong)frame_meta->frame_num);
        reid_params_dump_file = fopen(reid_file, "w");
        if (!reid_params_dump_file)
            continue;

        /** Save the reid embedding for each object. */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            guint64 id = obj->object_id;

            for (NvDsUserMetaList *l_obj_user = obj->obj_user_meta_list; l_obj_user != NULL;
                 l_obj_user = l_obj_user->next)
            {

                /** Find the object's reid embedding index in user meta. */
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_obj_user->data;
                if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_OBJ_REID_META && user_meta->user_meta_data)
                {

                    NvDsObjReid *pReidObj = (NvDsObjReid *)(user_meta->user_meta_data);
                    if (pReidObj != NULL && pReidObj->ptr_host != NULL && pReidObj->featureSize > 0)
                    {
                        fprintf(reid_params_dump_file, "%lu", id);
                        for (guint ele_i = 0; ele_i < pReidObj->featureSize; ele_i++)
                        {
                            fprintf(reid_params_dump_file, " %f", pReidObj->ptr_host[ele_i]);
                        }
                        fprintf(reid_params_dump_file, "\n");
                    }
                }
            }
        }
        fclose(reid_params_dump_file);
    }
}

/**
 * Function to dump terminated object information to files when the tracker outputs
 * terminated track info into user meta. For this to work, property "terminated_track_output_path"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_terminated_track_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    if (!appCtx->config.terminated_track_output_path)
        return;

    gchar term_file[1024] = {0};
    FILE *term_params_dump_file = NULL;
    /** Find batch terminted tensor in batch user meta. */
    GList *pTerminatedTrackList = NULL; // list of pointers to NvDsTargetMiscDataBatch
    NvDsTargetMiscDataBatch *pTerminatedTrackBatch = NULL;
    for (NvDsUserMetaList *l_batch_user = batch_meta->batch_user_meta_list;
         l_batch_user != NULL;
         l_batch_user = l_batch_user->next)
    {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_batch_user->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_TERMINATED_LIST_META)
        {
            pTerminatedTrackBatch = (NvDsTargetMiscDataBatch *)(user_meta->user_meta_data);
            pTerminatedTrackList = g_list_append(pTerminatedTrackList, pTerminatedTrackBatch);
        }
    }

    if (!pTerminatedTrackList)
        return;

    /** Save the Terminated data for each frame. */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        for (GList *l = pTerminatedTrackList; l != NULL; l = l->next)
        {
            pTerminatedTrackBatch = (NvDsTargetMiscDataBatch *)(l->data);

            for (uint si = 0; si < pTerminatedTrackBatch->numFilled; si++)
            {
                NvDsTargetMiscDataStream *objStream = (pTerminatedTrackBatch->list) + si;
                guint stream_id = (guint)(objStream->streamID);

                if (frame_meta->pad_index != stream_id)
                    continue;

                g_snprintf(term_file, sizeof(term_file) - 1,
                           "%s/%02u_%03u_%06lu.txt", appCtx->config.terminated_track_output_path,
                           appCtx->index, stream_id, (gulong)frame_meta->frame_num);

                term_params_dump_file = fopen(term_file, "w");

                if (!term_params_dump_file)
                    continue;

                for (uint li = 0; li < objStream->numFilled; li++)
                {

                    NvDsTargetMiscDataObject *objList = (objStream->list) + li;
                    fprintf(term_params_dump_file,
                            "Target: %ld,%d,%hu\n",
                            objList->uniqueId,
                            objList->classId,
                            stream_id);

                    for (uint oi = 0; oi < objList->numObj; oi++)
                    {
                        NvDsTargetMiscDataFrame *obj = (objList->list) + oi;

                        float left = obj->tBbox.left;
                        float right = left + obj->tBbox.width;
                        float top = obj->tBbox.top;
                        float bottom = top + obj->tBbox.height;

                        fprintf(term_params_dump_file,
                                "%u %lu %u 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %d %f\n",
                                obj->frameNum, objList->uniqueId, objList->classId, left, top, right, bottom,
                                obj->confidence, obj->trackerState, obj->visibility);
                    }
                }
                fprintf(term_params_dump_file, "\n");

                fclose(term_params_dump_file);
            }
        }
    }
    g_list_free(pTerminatedTrackList);
}

/**
 * Function to dump terminated object information to files when the tracker outputs
 * terminated track info into user meta. For this to work, property "terminated_track_output_path"
 * must be set in configuration file.
 * Data of different sources and frames is dumped in separate file.
 */
static void
write_shadow_track_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    if (!appCtx->config.shadow_track_output_path)
        return;

    gchar term_file[1024] = {0};
    FILE *shadow_dump_file = NULL;
    /** Find shadow tracked tensor in batch user meta. */
    GList *pShadowTrackList = NULL; // list of pointers to NvDsTargetMiscDataBatch
    NvDsTargetMiscDataBatch *pShadowTrackBatch = NULL;
    for (NvDsUserMetaList *l_batch_user = batch_meta->batch_user_meta_list;
         l_batch_user != NULL;
         l_batch_user = l_batch_user->next)
    {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_batch_user->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_SHADOW_LIST_META)
        {
            // std::cout << "Found Shadow Data" << std::endl;
            pShadowTrackBatch = (NvDsTargetMiscDataBatch *)(user_meta->user_meta_data);
            pShadowTrackList = g_list_append(pShadowTrackList, pShadowTrackBatch);
        }
    }

    if (!pShadowTrackList)
        return;

    /** Save the Terminated data for each frame. */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        for (GList *l = pShadowTrackList; l != NULL; l = l->next)
        {
            pShadowTrackBatch = (NvDsTargetMiscDataBatch *)(l->data);
            for (uint si = 0; si < pShadowTrackBatch->numFilled; si++)
            {
                NvDsTargetMiscDataStream *objStream = (pShadowTrackBatch->list) + si;
                guint stream_id = (guint)(objStream->streamID);

                if (frame_meta->pad_index != stream_id)
                    continue;

                g_snprintf(term_file, sizeof(term_file) - 1,
                           "%s/%02u_%03u_%06lu.txt", appCtx->config.shadow_track_output_path,
                           appCtx->index, stream_id, (gulong)frame_meta->frame_num);

                shadow_dump_file = fopen(term_file, "w");

                if (!shadow_dump_file)
                    continue;

                for (uint li = 0; li < objStream->numFilled; li++)
                {
                    NvDsTargetMiscDataObject *objList = (objStream->list) + li;

                    if (objList->numObj > 0)
                    {
                        NvDsTargetMiscDataFrame *obj = (objList->list); // get first element only

                        float left = obj->tBbox.left;
                        float right = left + obj->tBbox.width;
                        float top = obj->tBbox.top;
                        float bottom = top + obj->tBbox.height;

                        fprintf(shadow_dump_file,
                                "%u %lu %u 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %d %f\n",
                                obj->frameNum, objList->uniqueId, objList->classId, left, top, right, bottom,
                                obj->confidence, obj->trackerState, obj->visibility);
                    }
                }

                fclose(shadow_dump_file);
            }
        }
    }
    g_list_free(pShadowTrackList);
}

static gboolean
add_and_link_broker_sink(AppCtx *appCtx)
{
    NvDsConfig *config = &appCtx->config;
    /** Only first instance_bin broker sink
     * employed as there's only one analytics path for N sources
     * NOTE: There shall be only one [sink] group
     * with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
     * a) Multiple of them does not make sense as we have only
     * one analytics pipe generating the data for broker sink
     * b) If Multiple broker sinks are configured by the user
     * in config file, only the first in the order of
     * appearance will be considered
     * and others shall be ignored
     * c) Ideally it should be documented (or obvious) that:
     * multiple [sink] groups with type=6 (NV_DS_SINK_MSG_CONV_BROKER)
     * is invalid
     */
    NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[0];
    NvDsPipeline *pipeline = &appCtx->pipeline;

    for (guint i = 0; i < config->num_sink_sub_bins; i++)
    {
        if (config->sink_bin_sub_bin_config[i].type == NV_DS_SINK_MSG_CONV_BROKER)
        {
            if (!pipeline->common_elements.tee)
            {
                NVGSTDS_ERR_MSG_V("%s failed; broker added without analytics; check config file\n",
                                  __func__);
                return FALSE;
            }
            /** add the broker sink bin to pipeline */
            if (!gst_bin_add(GST_BIN(pipeline->pipeline),
                             instance_bin->sink_bin.sub_bins[i].bin))
            {
                return FALSE;
            }
            /** link the broker sink bin to the common_elements tee
             * (The tee after nvinfer -> tracker (optional) -> sgies (optional) block) */
            if (!link_element_to_tee_src_pad(pipeline->common_elements.tee,
                                             instance_bin->sink_bin.sub_bins[i].bin))
            {
                return FALSE;
            }
        }
    }
    return TRUE;
}

static gboolean
create_demux_pipeline(AppCtx *appCtx, guint index)
{
    gboolean ret = FALSE;
    NvDsConfig *config = &appCtx->config;
    NvDsInstanceBin *instance_bin = &appCtx->pipeline.demux_instance_bins[index];
    GstElement *last_elem;
    gchar elem_name[32];

    instance_bin->index = index;
    instance_bin->appCtx = appCtx;

    g_snprintf(elem_name, 32, "processing_demux_bin_%d", index);
    instance_bin->bin = gst_bin_new(elem_name);

    if (!create_demux_sink_bin(config->num_sink_sub_bins,
                               config->sink_bin_sub_bin_config, &instance_bin->demux_sink_bin,
                               config->sink_bin_sub_bin_config[index].source_id))
    {
        goto done;
    }

    gst_bin_add(GST_BIN(instance_bin->bin), instance_bin->demux_sink_bin.bin);
    last_elem = instance_bin->demux_sink_bin.bin;

    if (config->osd_config.enable)
    {
        if (!create_osd_bin(&config->osd_config, &instance_bin->osd_bin))
        {
            goto done;
        }

        gst_bin_add(GST_BIN(instance_bin->bin), instance_bin->osd_bin.bin);

        NVGSTDS_LINK_ELEMENT(instance_bin->osd_bin.bin, last_elem);

        last_elem = instance_bin->osd_bin.bin;
    }

    NVGSTDS_BIN_ADD_GHOST_PAD(instance_bin->bin, last_elem, "sink");
    if (config->osd_config.enable)
    {
        NVGSTDS_ELEM_ADD_PROBE(instance_bin->all_bbox_buffer_probe_id,
                               instance_bin->osd_bin.nvosd, "sink",
                               gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
    }
    else
    {
        NVGSTDS_ELEM_ADD_PROBE(instance_bin->all_bbox_buffer_probe_id,
                               instance_bin->demux_sink_bin.bin, "sink",
                               gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
    }

    ret = TRUE;
done:
    if (!ret)
    {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}

/**
 * Function to add components to pipeline which are dependent on number
 * of streams. These components work on single buffer. If tiling is being
 * used then single instance will be created otherwise < N > such instances
 * will be created for < N > streams
 * 函数用于向管道添加依赖于流数量的组件。
 * 这些组件在单个缓冲区上工作。如果使用平铺，则将创建单个实例，否则将为 <N> 个流创建 <N> 个此类实例。
 */
static gboolean
create_processing_instance(AppCtx *appCtx, guint index)
{
    gboolean ret = FALSE;
    NvDsConfig *config = &appCtx->config;
    NvDsInstanceBin *instance_bin = &appCtx->pipeline.instance_bins[index];
    GstElement *last_elem;
    gchar elem_name[32];

    instance_bin->index = index;
    instance_bin->appCtx = appCtx;

    g_snprintf(elem_name, 32, "processing_bin_%d", index);
    instance_bin->bin = gst_bin_new(elem_name);

    if (!create_sink_bin(config->num_sink_sub_bins,
                         config->sink_bin_sub_bin_config, &instance_bin->sink_bin, index))
    {
        goto done;
    }

    gst_bin_add(GST_BIN(instance_bin->bin), instance_bin->sink_bin.bin);
    last_elem = instance_bin->sink_bin.bin;

    if (config->osd_config.enable)
    {
        if (!create_osd_bin(&config->osd_config, &instance_bin->osd_bin))
        {
            goto done;
        }

        gst_bin_add(GST_BIN(instance_bin->bin), instance_bin->osd_bin.bin);

        NVGSTDS_LINK_ELEMENT(instance_bin->osd_bin.bin, last_elem);

        last_elem = instance_bin->osd_bin.bin;
    }

    NVGSTDS_BIN_ADD_GHOST_PAD(instance_bin->bin, last_elem, "sink");
    if (config->osd_config.enable)
    {
        // 在osd的输入pad上添加一个探针，此处已经得到了所有的bbox信息
        NVGSTDS_ELEM_ADD_PROBE(instance_bin->all_bbox_buffer_probe_id,
                               instance_bin->osd_bin.nvosd, "sink",
                               gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
    }
    else
    {
        NVGSTDS_ELEM_ADD_PROBE(instance_bin->all_bbox_buffer_probe_id,
                               instance_bin->sink_bin.bin, "sink",
                               gie_processing_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, instance_bin);
    }

    ret = TRUE;
done:
    if (!ret)
    {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}

/* 创建并插入自定义 udpmulticast 源到 pipeline (简单版：单独一个源 -> streammux) */
/* 独立的 udpmulticast 分支（不接入主视频推理链） */
static GstFlowReturn on_udpmulticast_sample(GstElement *sink, AppCtx *appCtx)
{
    GstSample *sample = NULL;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (!sample)
        return GST_FLOW_OK;
    /* 调试：每次收到 sample 都打印一次（可按需改为取模减少日志） */
    static guint frame_cnt = 0;
    GstBuffer *buf = gst_sample_get_buffer(sample);
    gsize size = 0;
    if (buf) {
        GstMapInfo map;
        if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
            size = map.size;
            gst_buffer_unmap(buf, &map);
        }
    }
    g_print("[udpmulticast] new-sample #%u buffer-size=%zu bytes\n", ++frame_cnt, size);
    /* TODO: 在这里解析真实业务数据（当前插件还未把 UDP 载荷填入 buffer，仅生成黑帧） */
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static gboolean
add_udpmulticast_source(AppCtx *appCtx)
{
    NvDsConfig *config = &appCtx->config;
    if (!config->udpmulticast_config.enable)
        return TRUE; /* 未启用直接返回 */

    /* --- 调试辅助：记录函数进入 --- */
    g_print("[udpsrc-multicast] add_udpmulticast_source() enter\n");

    /* 改为使用内置 udpsrc 直接加入组播，不再依赖自定义插件 */
    GstElement *udpsrc = gst_element_factory_make("udpsrc", "app_udpmulticast_src");
    GstElement *queue = gst_element_factory_make("queue", "udpmulti_queue");
    GstElement *sink = gst_element_factory_make("appsink", "udpmulti_sink");
    if (!udpsrc || !queue || !sink)
    {
        NVGSTDS_ERR_MSG_V("Failed to create udpmulticast branch elements");
        return FALSE;
    }

    if (config->udpmulticast_config.multicast_ip) {
        g_object_set(G_OBJECT(udpsrc), "multicast-group", config->udpmulticast_config.multicast_ip, NULL);
    }
    if (config->udpmulticast_config.port)
        g_object_set(G_OBJECT(udpsrc), "port", config->udpmulticast_config.port, NULL);
    /* iface: 若提供的是网卡名(如 eth0), 直接赋给 multicast-iface; 如果看起来像 IPv4 地址, 做提示 */
    if (config->udpmulticast_config.iface) {
        const gchar *iface = config->udpmulticast_config.iface;
        gboolean looks_ip = FALSE;
        int dot_cnt = 0; for (const char *p = iface; *p; ++p) if (*p=='.') dot_cnt++;
        if (dot_cnt == 3) looks_ip = TRUE; /* 粗略判断 */
        if (looks_ip) {
            g_print("[udpsrc-multicast][warn] iface='%s' 像是IP地址; udpsrc 的 multicast-iface 期望网卡名 (如 eth0). 建议改为设备名.\n", iface);
        }
        g_object_set(G_OBJECT(udpsrc), "multicast-iface", iface, NULL);
    }
    /* auto-multicast 让 udpsrc 自动 bind 与 setsockopt */
    g_object_set(G_OBJECT(udpsrc), "auto-multicast", TRUE, NULL);
    /* 允许地址重用 (多个监听 / 容器重启) */
    g_object_set(G_OBJECT(udpsrc), "reuse", TRUE, NULL);
    if (config->udpmulticast_config.recv_buf_size)
        g_object_set(G_OBJECT(udpsrc), "buffer-size", config->udpmulticast_config.recv_buf_size, NULL);

    /* 可选：设置超时时间（毫秒）-> 如果想让套接字更快产出数据或探测空闲，可用 timeout 属性 (GStreamer 1.22+ 支持) */
#ifdef GST_1_22
    g_object_set(G_OBJECT(udpsrc), "timeout", (guint64)5 * 1000 * 1000 * 1000ULL, NULL); /* 5s 无数据发送 GST_EVENT_EOS (调试可关闭) */
#endif

    /* queue 做节流，防止阻塞 (50Hz+ 可以适当调小缓冲) */
    // g_object_set(G_OBJECT(queue), "max-size-buffers", 100, "leaky", 2, NULL);

    /* appsink 设置 */
    g_object_set(G_OBJECT(sink), "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(sink, "new-sample", G_CALLBACK(on_udpmulticast_sample), appCtx);

    GstElement *pipeline = appCtx->pipeline.pipeline;
    gst_bin_add_many(GST_BIN(pipeline), udpsrc, queue, sink, NULL);
    if (!gst_element_link_many(udpsrc, queue, sink, NULL))
    {
        NVGSTDS_ERR_MSG_V("Failed to link udpmulticast branch elements");
        return FALSE;
    }

    /* 同步状态 */
    gst_element_sync_state_with_parent(udpsrc);
    gst_element_sync_state_with_parent(queue);
    gst_element_sync_state_with_parent(sink);
    /* 在源 pad 上加探针 */
    // GstPad *srcpad = gst_element_get_static_pad(udpsrc, "src");
    // if (srcpad) {
    //     gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, udpsrc_probe_cb, NULL, NULL);
    //     gst_object_unref(srcpad);
    // }

    /* 打印最终属性 */
    gchar *group = NULL; gchar *iface = NULL; gboolean auto_mc = FALSE; gboolean reuse = FALSE; gint port = 0; guint bufsize=0;
    g_object_get(udpsrc, "multicast-group", &group, "multicast-iface", &iface, "auto-multicast", &auto_mc, "reuse", &reuse, "port", &port, "buffer-size", &bufsize, NULL);
    g_print("[udpsrc-multicast] started group=%s port=%d iface=%s auto=%d reuse=%d bufsize=%u (NOT linked to streammux)\n",
            group?group:"(null)", port, iface?iface:"(null)", auto_mc, reuse, bufsize);
    if (group) g_free(group); if (iface) g_free(iface);

    return TRUE;
}

/**
 * Function to create common elements(Primary infer, tracker, secondary infer)
 * of the pipeline. These components operate on muxed data from all the
 * streams. So they are independent of number of streams in the pipeline.
 * 创建pipeline的常用元素（主推理、跟踪器、次推理）函数。这些组件在所有流的复用数据上操作。因此，它们独立于pipeline中的流数量。
 */
static gboolean
create_common_elements(NvDsConfig *config, NvDsPipeline *pipeline,
                       GstElement **sink_elem, GstElement **src_elem,
                       bbox_generated_callback bbox_generated_post_analytics_cb)
{
    gboolean ret = FALSE;
    *sink_elem = *src_elem = NULL;

    if (config->segvisual_config.enable)
    {
        if (!create_segvisual_bin(&config->segvisual_config,
                                  &pipeline->common_elements.segvisual_bin))
        {
            goto done;
        }

        gst_bin_add(GST_BIN(pipeline->pipeline),
                    pipeline->common_elements.segvisual_bin.bin);

        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.segvisual_bin.bin;
        }
        if (*sink_elem)
        {
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.segvisual_bin.bin,
                                 *sink_elem);
        }
        *sink_elem = pipeline->common_elements.segvisual_bin.bin;
    }

    if (config->primary_gie_config.enable)
    {
        if (config->num_secondary_gie_sub_bins > 0)
        {
            /** if using nvmultiurisrcbin, override batch-size config for sgie */
            if (config->use_nvmultiurisrcbin)
            {
                for (guint i = 0; i < config->num_secondary_gie_sub_bins; i++)
                {
                    config->secondary_gie_sub_bin_config[i].batch_size =
                        config->sgie_batch_size;
                }
            }
            if (!create_secondary_gie_bin(config->num_secondary_gie_sub_bins,
                                          config->primary_gie_config.unique_id,
                                          config->secondary_gie_sub_bin_config,
                                          &pipeline->common_elements.secondary_gie_bin))
            {
                goto done;
            }
            gst_bin_add(GST_BIN(pipeline->pipeline),
                        pipeline->common_elements.secondary_gie_bin.bin);
            if (!*src_elem)
            {
                *src_elem = pipeline->common_elements.secondary_gie_bin.bin;
            }
            if (*sink_elem)
            {
                NVGSTDS_LINK_ELEMENT(pipeline->common_elements.secondary_gie_bin.bin,
                                     *sink_elem);
            }
            *sink_elem = pipeline->common_elements.secondary_gie_bin.bin;
        }
    }

    if (config->primary_gie_config.enable)
    {
        if (config->num_secondary_preprocess_sub_bins > 0)
        {
            if (!create_secondary_preprocess_bin(config->num_secondary_preprocess_sub_bins,
                                                 config->primary_gie_config.unique_id,
                                                 config->secondary_preprocess_sub_bin_config,
                                                 &pipeline->common_elements.secondary_preprocess_bin))
            {
                g_print("creating secondary_preprocess bin failed\n");
                goto done;
            }
            gst_bin_add(GST_BIN(pipeline->pipeline),
                        pipeline->common_elements.secondary_preprocess_bin.bin);

            if (!*src_elem)
            {
                *src_elem = pipeline->common_elements.secondary_preprocess_bin.bin;
            }
            if (*sink_elem)
            {
                NVGSTDS_LINK_ELEMENT(pipeline->common_elements.secondary_preprocess_bin.bin, *sink_elem);
            }

            *sink_elem = pipeline->common_elements.secondary_preprocess_bin.bin;
        }
    }

    if (config->dsanalytics_config.enable)
    {
        if (!create_dsanalytics_bin(&config->dsanalytics_config,
                                    &pipeline->common_elements.dsanalytics_bin))
        {
            g_print("creating dsanalytics bin failed\n");
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline),
                    pipeline->common_elements.dsanalytics_bin.bin);

        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.dsanalytics_bin.bin;
        }
        if (*sink_elem)
        {
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.dsanalytics_bin.bin,
                                 *sink_elem);
        }
        *sink_elem = pipeline->common_elements.dsanalytics_bin.bin;
    }

    // 启用自定义的多帧目标识别插件
    if (config->videorecognition_config.enable)
    {
        if (!create_dsvideorecognition_bin(&config->videorecognition_config,
                                           &pipeline->common_elements.videorecognition_bin))
        {
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->common_elements.videorecognition_bin.bin);

        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.videorecognition_bin.bin;
        }
        if (*sink_elem)
        {
            // 把videorecognition的输出src pad连接到上一个元素的sink pad输入
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.videorecognition_bin.bin,
                                 *sink_elem);
        }

        // 也就说，如果启用该插件，该插件的输入应该要连接到跟踪的输出
        *sink_elem = pipeline->common_elements.videorecognition_bin.bin;
    }

    if (config->udpjsonmeta_config.enable)
    {
        GstElement *udpjsonmeta = gst_element_factory_make(NVDS_ELEM_UDPJSONMETA_ELEMENT, "udpjsonmeta"); /* UDP JSON 元数据插件 */
        if (!udpjsonmeta)
        {
            NVGSTDS_ERR_MSG_V("Failed to create element '%s'", NVDS_ELEM_UDPJSONMETA_ELEMENT);
            goto done;
        }

        if (config->udpjsonmeta_config.multicast_ip)
            g_object_set(G_OBJECT(udpjsonmeta), "multicast-ip", config->udpjsonmeta_config.multicast_ip, NULL);
        if (config->udpjsonmeta_config.iface)
            g_object_set(G_OBJECT(udpjsonmeta), "iface", config->udpjsonmeta_config.iface, NULL);
        if (config->udpjsonmeta_config.recv_buf_size)
            g_object_set(G_OBJECT(udpjsonmeta), "recv-buf-size", config->udpjsonmeta_config.recv_buf_size, NULL);
        if (config->udpjsonmeta_config.cache_ttl_ms)
            g_object_set(G_OBJECT(udpjsonmeta), "cache-ttl-ms", config->udpjsonmeta_config.cache_ttl_ms, NULL);
        if (config->udpjsonmeta_config.max_cache_size)
            g_object_set(G_OBJECT(udpjsonmeta), "max-cache-size", config->udpjsonmeta_config.max_cache_size, NULL);

        /* C-UAV 协议配置 */
        if (config->udpjsonmeta_config.enable_cuav_parser)
        {
            g_object_set(G_OBJECT(udpjsonmeta), "enable-cuav-parser", TRUE, NULL);
            g_object_set(G_OBJECT(udpjsonmeta), "cuav-port", config->udpjsonmeta_config.cuav_port, NULL);
            if (config->udpjsonmeta_config.cuav_ctrl_port)
            {
                g_object_set(G_OBJECT(udpjsonmeta), "cuav-ctrl-port",
                             config->udpjsonmeta_config.cuav_ctrl_port, NULL);
            }
            if (config->udpjsonmeta_config.enable_cuav_debug)
            {
                g_object_set(G_OBJECT(udpjsonmeta), "cuav-debug", TRUE, NULL);
            }
            gst_udpjson_meta_set_guidance_callback(GST_UDPJSON_META(udpjsonmeta),
                                                   on_cuav_guidance,
                                                   pipeline->common_elements.appCtx);
        }

        gst_bin_add(GST_BIN(pipeline->pipeline), udpjsonmeta);
        if (!*src_elem)
        {
            *src_elem = udpjsonmeta;
        }
        if (*sink_elem)
        {
            NVGSTDS_LINK_ELEMENT(udpjsonmeta, *sink_elem);
        }
        *sink_elem = udpjsonmeta;
        pipeline->common_elements.udpjsonmeta = udpjsonmeta;
    }

    if (config->tracker_config.enable)
    {
        if (!create_tracking_bin(&config->tracker_config,
                                 &pipeline->common_elements.tracker_bin))
        {
            g_print("creating tracker bin failed\n");
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline),
                    pipeline->common_elements.tracker_bin.bin);
        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.tracker_bin.bin;
        }
        if (*sink_elem)
        {
            // 也就说，如果启用了自定义多帧目标识别的插件，会进入这里，把跟踪的输出连接到上一个元素的sink pad输入
            // 也就是把跟踪的输出连接到videorecognition的输入
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.tracker_bin.bin,
                                 *sink_elem);
        }
        *sink_elem = pipeline->common_elements.tracker_bin.bin;
    }

    if (config->primary_gie_config.enable)
    {
        /** if using nvmultiurisrcbin, override batch-size config for pgie */
        if (config->use_nvmultiurisrcbin)
        {
            config->primary_gie_config.batch_size = config->max_batch_size;
        }
        if (!create_primary_gie_bin(&config->primary_gie_config,
                                    &pipeline->common_elements.primary_gie_bin))
        {
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline),
                    pipeline->common_elements.primary_gie_bin.bin);
        if (*sink_elem)
        {
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.primary_gie_bin.bin,
                                 *sink_elem);
        }
        *sink_elem = pipeline->common_elements.primary_gie_bin.bin;
        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.primary_gie_bin.bin;
        }

        pipeline->common_elements.appCtx->obj_ctx_handle =
            nvds_obj_enc_create_context(pipeline->common_elements.appCtx->config.primary_gie_config.gpu_id);

        if (!pipeline->common_elements.appCtx->obj_ctx_handle)
        {
            g_print("Unable to create context\n");
            goto done;
        }

        NVGSTDS_ELEM_ADD_PROBE(pipeline->common_elements.primary_bbox_buffer_probe_id,
                               pipeline->common_elements.primary_gie_bin.bin,
                               "src",
                               gie_primary_processing_done_buf_prob,
                               GST_PAD_PROBE_TYPE_BUFFER,
                               pipeline->common_elements.appCtx);
    }

    if (config->preprocess_config.enable)
    {
        if (!create_preprocess_bin(&config->preprocess_config,
                                   &pipeline->common_elements.preprocess_bin))
        {
            g_print("creating preprocess bin failed\n");
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline),
                    pipeline->common_elements.preprocess_bin.bin);

        if (!*src_elem)
        {
            *src_elem = pipeline->common_elements.preprocess_bin.bin;
        }
        if (*sink_elem)
        {
            NVGSTDS_LINK_ELEMENT(pipeline->common_elements.preprocess_bin.bin,
                                 *sink_elem);
        }

        *sink_elem = pipeline->common_elements.preprocess_bin.bin;
    }

    if (*src_elem)
    {
        NVGSTDS_ELEM_ADD_PROBE(pipeline->common_elements.primary_bbox_buffer_probe_id, *src_elem, "src",
                               analytics_done_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
                               &pipeline->common_elements);

        /* Add common message converter */
        if (config->msg_conv_config.enable)
        {
            NvDsSinkMsgConvBrokerConfig *convConfig = &config->msg_conv_config;
            pipeline->common_elements.msg_conv =
                gst_element_factory_make(NVDS_ELEM_MSG_CONV, "common_msg_conv");
            if (!pipeline->common_elements.msg_conv)
            {
                NVGSTDS_ERR_MSG_V("Failed to create element 'common_msg_conv'");
                goto done;
            }

            g_object_set(G_OBJECT(pipeline->common_elements.msg_conv),
                         "config", convConfig->config_file_path,
                         "msg2p-lib",
                         (convConfig->conv_msg2p_lib ? convConfig->conv_msg2p_lib : "null"),
                         "payload-type", convConfig->conv_payload_type, "comp-id",
                         convConfig->conv_comp_id, "debug-payload-dir",
                         convConfig->debug_payload_dir, "multiple-payloads",
                         convConfig->multiple_payloads, "msg2p-newapi", convConfig->conv_msg2p_new_api,
                         "frame-interval", convConfig->conv_frame_interval, NULL);

            gst_bin_add(GST_BIN(pipeline->pipeline),
                        pipeline->common_elements.msg_conv);

            NVGSTDS_LINK_ELEMENT(*src_elem, pipeline->common_elements.msg_conv);
            *src_elem = pipeline->common_elements.msg_conv;
        }
        pipeline->common_elements.tee =
            gst_element_factory_make(NVDS_ELEM_TEE, "common_analytics_tee");
        if (!pipeline->common_elements.tee)
        {
            NVGSTDS_ERR_MSG_V("Failed to create element 'common_analytics_tee'");
            goto done;
        }

        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->common_elements.tee);

        NVGSTDS_LINK_ELEMENT(*src_elem, pipeline->common_elements.tee);
        *src_elem = pipeline->common_elements.tee;
    }

    ret = TRUE;
done:
    return ret;
}

static gboolean
is_sink_available_for_source_id(NvDsConfig *config, guint source_id)
{
    for (guint j = 0; j < config->num_sink_sub_bins; j++)
    {
        if (config->sink_bin_sub_bin_config[j].enable &&
            config->sink_bin_sub_bin_config[j].source_id == source_id &&
            config->sink_bin_sub_bin_config[j].link_to_demux == FALSE)
        {
            return TRUE;
        }
    }
    return FALSE;
}

/**
 * Main function to create the pipeline.
 */
/**
 * @brief 创建管道
 *
 * @param appCtx 应用程序上下文
 * @param bbox_generated_post_analytics_cb 分析后生成的边界框回调
 * @param all_bbox_generated_cb 所有生成的边界框回调
 * @param perf_cb 性能回调
 * @param overlay_graphics_cb 覆盖图形回调
 * @return gboolean 返回TRUE表示成功，FALSE表示失败
 */
gboolean
create_pipeline(AppCtx *appCtx,
                bbox_generated_callback bbox_generated_post_analytics_cb,
                bbox_generated_callback all_bbox_generated_cb,
                perf_callback perf_cb,
                overlay_graphics_callback overlay_graphics_cb,
                nv_msgbroker_subscribe_cb_t msg_broker_subscribe_cb)
{
    gboolean ret = FALSE;
    NvDsPipeline *pipeline = &appCtx->pipeline;
    NvDsConfig *config = &appCtx->config;
    GstBus *bus;
    GstElement *last_elem;
    GstElement *tmp_elem1;
    GstElement *tmp_elem2;
    guint i;
    GstPad *fps_pad = NULL;
    gulong latency_probe_id;

    _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);

    appCtx->all_bbox_generated_cb = all_bbox_generated_cb;
    appCtx->bbox_generated_post_analytics_cb = bbox_generated_post_analytics_cb;
    appCtx->overlay_graphics_cb = overlay_graphics_cb;
    appCtx->sensorInfoHash = g_hash_table_new(NULL, NULL);
    appCtx->perf_struct.FPSInfoHash = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, NULL);

    // 初始化 ROI-based NMS 配置（只读取一次，避免每帧重复解析）
    appCtx->roi_nms_enabled = FALSE;
    appCtx->roi_centers = NULL;
    
    if (config->preprocess_config.config_file_path) {
        GKeyFile *key_file = g_key_file_new();
        GError *error = NULL;
        
        if (g_key_file_load_from_file(key_file, config->preprocess_config.config_file_path, 
                                       G_KEY_FILE_NONE, &error)) {
            // 检查 [group-0] 中的 process-on-roi
            if (g_key_file_has_group(key_file, "group-0") && 
                g_key_file_has_key(key_file, "group-0", "process-on-roi", NULL)) {
                gint process_on_roi = g_key_file_get_integer(key_file, "group-0", "process-on-roi", NULL);
                
                if (process_on_roi == 1 && g_key_file_has_key(key_file, "group-0", "roi-params-src-0", NULL)) {
                    gchar *roi_params_str = g_key_file_get_string(key_file, "group-0", "roi-params-src-0", NULL);
                    
                    if (roi_params_str) {
                        // 解析 ROI 参数: left;top;width;height;left;top;width;height;...
                        gchar **roi_tokens = g_strsplit(roi_params_str, ";", -1);
                        guint num_tokens = g_strv_length(roi_tokens);
                        
                        // 移除末尾的空字符串（如果配置末尾有分号）
                        while (num_tokens > 0 && roi_tokens[num_tokens - 1][0] == '\0') {
                            num_tokens--;
                        }
                        
                        if (num_tokens >= 4 && num_tokens % 4 == 0) {
                            appCtx->roi_nms_enabled = TRUE;
                            appCtx->roi_centers = g_array_new(FALSE, FALSE, sizeof(gfloat));
                            
                            g_print("[ROI-NMS] ROI-based NMS enabled. ROI count: %u\n", num_tokens / 4);
                            
                            for (guint i = 0; i < num_tokens; i += 4) {
                                gfloat left = g_ascii_strtod(roi_tokens[i], NULL);
                                gfloat top = g_ascii_strtod(roi_tokens[i+1], NULL);
                                gfloat width = g_ascii_strtod(roi_tokens[i+2], NULL);
                                gfloat height = g_ascii_strtod(roi_tokens[i+3], NULL);
                                
                                gfloat center[2];
                                center[0] = left + width / 2.0f;   // center_x
                                center[1] = top + height / 2.0f;   // center_y
                                g_array_append_vals(appCtx->roi_centers, center, 2);
                                
                                g_print("[ROI-NMS]   ROI %u: left=%.0f top=%.0f width=%.0f height=%.0f center=(%.1f, %.1f)\n",
                                        i/4, left, top, width, height, center[0], center[1]);
                            }
                        }
                        
                        g_strfreev(roi_tokens);
                        g_free(roi_params_str);
                    }
                }
            }
        }
        
        if (error) {
            g_error_free(error);
        }
        g_key_file_free(key_file);
    }

    if (config->osd_config.num_out_buffers < 8)
    {
        config->osd_config.num_out_buffers = 8;
    }

    pipeline->pipeline = gst_pipeline_new("pipeline");
    if (!pipeline->pipeline)
    {
        NVGSTDS_ERR_MSG_V("Failed to create pipeline");
        goto done;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline->pipeline));
    pipeline->bus_id = gst_bus_add_watch(bus, bus_callback, appCtx);
    gst_object_unref(bus);

    if (config->file_loop)
    {
        /* Let each source bin know it needs to loop. */
        guint i;
        for (i = 0; i < config->num_source_sub_bins; i++)
            config->multi_source_config[i].loop = TRUE;
    }

    /* --- BEGIN: UDP JSON 控制报文接收分支 --- */
    // {
    //     GstElement *udp_ctrl_src =
    //         gst_element_factory_make("udpsrc", "udp_ctrl_src");
    //     GstElement *queue = gst_element_factory_make("queue", "ctrl_queue");
    //     GstElement *caps_json =
    //         gst_element_factory_make("capsfilter", "caps_json");
    //     GstElement *json_sink =
    //         gst_element_factory_make("appsink", "json_ctrl_sink");

    //     if (!udp_ctrl_src || !caps_json || !json_sink || !queue)
    //     {
    //         NVGSTDS_ERR_MSG_V("Failed to create UDP control elements");
    //         goto done;
    //     }

    //     // leaky=2 表示丢弃最新数据（downstream），leaky=1
    //     // 表示丢弃最旧数据（upstream）。
    //     g_object_set(G_OBJECT(queue), "max-size-buffers", 10, "leaky", 1, NULL);

    //     /* 设置 UDP 接收端口或组播组 */
    //     g_object_set(G_OBJECT(udp_ctrl_src), "multicast-group",
    //                  "239.255.255.250", "port", 5000, "auto-multicast", TRUE,
    //                  NULL);

    //     GstCaps *caps = gst_caps_new_empty_simple("application/x-json");
    //     g_object_set(caps_json, "caps", caps, NULL);
    //     gst_caps_unref(caps);

    //     /* appsink 配置：异步接收并回调处理 */
    //     g_object_set(G_OBJECT(json_sink), "emit-signals", TRUE, "sync", FALSE,
    //                  NULL);
    //     g_signal_connect(json_sink, "new-sample", G_CALLBACK(on_control_data),
    //                      appCtx);

    //     /* 加入 pipeline 并链接 */
    //     gst_bin_add_many(GST_BIN(pipeline->pipeline), udp_ctrl_src, queue,
    //                      caps_json, json_sink, NULL);
    //     if (!gst_element_link_many(udp_ctrl_src, queue, caps_json, json_sink,
    //                                NULL))
    //     {
    //         NVGSTDS_ERR_MSG_V("Failed to link UDP JSON control elements");
    //         goto done;
    //     }
    // }
    /* --- END: UDP JSON 控制报文接收分支 --- */

    for (guint i = 0; i < config->num_sink_sub_bins; i++)
    {
        NvDsSinkSubBinConfig *sink_config = &config->sink_bin_sub_bin_config[i];
        switch (sink_config->type)
        {
        case NV_DS_SINK_FAKE:
#ifndef IS_TEGRA
        case NV_DS_SINK_RENDER_EGL:
#else
        case NV_DS_SINK_RENDER_3D:
#endif
        case NV_DS_SINK_RENDER_DRM:
            /* Set the "qos" property of sink, if not explicitly specified in the
               config. */
            if (!sink_config->render_config.qos_value_specified)
            {
                sink_config->render_config.qos = FALSE;
            }
        default:
            break;
        }
    }
    /*
     * Add muxer and < N > source components to the pipeline based
     * on the settings in configuration file.
     */
    if (config->use_nvmultiurisrcbin)
    {
        if (config->num_source_sub_bins > 0)
        {
            if (!create_nvmultiurisrcbin_bin(config->num_source_sub_bins,
                                             config->multi_source_config, &pipeline->multi_src_bin))
                goto done;
        }
        else
        {
            if (!config->source_attr_all_parsed)
            {
                NVGSTDS_ERR_MSG_V("[source-attr-all] config group not set, needs to be configured");
                goto done;
            }
            if (!create_nvmultiurisrcbin_bin(config->num_source_sub_bins,
                                             &config->source_attr_all_config, &pipeline->multi_src_bin))
                goto done;
            //[source-list] added with num-source-bins=0; This means source-bin
            // will be created and be waiting for source adds over REST API
            // mark num-source-bins=1 as one source-bin is indeed created
            config->num_source_sub_bins = 1;
        }
        /** set properties for nvmultiurisrcbin */
        if (config->uri_list)
        {
            gchar *uri_list_comma_sep = g_strjoinv(",", config->uri_list);
            g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "uri-list",
                         uri_list_comma_sep, NULL);
            g_free(uri_list_comma_sep);
        }
        if (config->sensor_id_list)
        {
            gchar *uri_list_comma_sep = g_strjoinv(",", config->sensor_id_list);
            g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "sensor-id-list",
                         uri_list_comma_sep, NULL);
            g_free(uri_list_comma_sep);
        }
        if (config->sensor_name_list)
        {
            gchar *uri_list_comma_sep = g_strjoinv(",", config->sensor_name_list);
            g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "sensor-name-list",
                         uri_list_comma_sep, NULL);
            g_free(uri_list_comma_sep);
        }
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "max-batch-size",
                     config->max_batch_size, NULL);
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "ip-address",
                     config->http_ip, NULL);
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "port",
                     config->http_port, NULL);
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "extract-sei-type5-data-dec",
                     config->extract_sei_type5_data, NULL);
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "low-latency-mode",
                     config->low_latency_mode, NULL);
        g_object_set(pipeline->multi_src_bin.nvmultiurisrcbin, "sei-uuid",
                     config->sei_uuid, NULL);
    }
    else
    {
        if (!create_multi_source_bin(config->num_source_sub_bins,
                                     config->multi_source_config, &pipeline->multi_src_bin))
            goto done;
    }
    gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->multi_src_bin.bin);

    /* 如果启用了[udpmulticast]，在streammux之后添加 */
    if (!add_udpmulticast_source(appCtx))
    {
        NVGSTDS_ERR_MSG_V("add_udpmulticast_source failed");
        goto done;
    }

    if (config->streammux_config.is_parsed)
    {
        if (config->use_nvmultiurisrcbin)
        {
            config->streammux_config.use_nvmultiurisrcbin = TRUE;
            /** overriding mux_config.batch_size to max_batch_size */
            config->streammux_config.batch_size = config->max_batch_size;
        }

        if (!set_streammux_properties(&config->streammux_config,
                                      pipeline->multi_src_bin.streammux))
        {
            NVGSTDS_WARN_MSG_V("Failed to set streammux properties");
        }
    }

    if (appCtx->latency_info == NULL)
    {
        appCtx->latency_info = (NvDsFrameLatencyInfo *)
            calloc(1, config->streammux_config.batch_size *
                          sizeof(NvDsFrameLatencyInfo));
    }

    /** a tee after the tiler which shall be connected to sink(s) */
    pipeline->tiler_tee = gst_element_factory_make(NVDS_ELEM_TEE, "tiler_tee");
    if (!pipeline->tiler_tee)
    {
        NVGSTDS_ERR_MSG_V("Failed to create element 'tiler_tee'");
        goto done;
    }
    gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->tiler_tee);

    /** Tiler + Demux in Parallel Use-Case */
    if (config->tiled_display_config.enable ==
        NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX)
    {
        pipeline->demuxer =
            gst_element_factory_make(NVDS_ELEM_STREAM_DEMUX, "demuxer");
        if (!pipeline->demuxer)
        {
            NVGSTDS_ERR_MSG_V("Failed to create element 'demuxer'");
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->demuxer);

        /** NOTE:
         * demux output is supported for only one source
         * If multiple [sink] groups are configured with
         * link_to_demux=1, only the first [sink]
         * shall be constructed for all occurences of
         * [sink] groups with link_to_demux=1
         */
        {
            gchar pad_name[16];
            GstPad *demux_src_pad;

            i = 0;
            if (!create_demux_pipeline(appCtx, i))
            {
                goto done;
            }

            for (i = 0; i < config->num_sink_sub_bins; i++)
            {
                if (config->sink_bin_sub_bin_config[i].link_to_demux == TRUE)
                {
                    g_snprintf(pad_name, 16, "src_%02d",
                               config->sink_bin_sub_bin_config[i].source_id);
                    break;
                }
            }

            if (i >= config->num_sink_sub_bins)
            {
                g_print("\n\nError : sink for demux (use link-to-demux-only property) is not provided in the config file\n\n");
                goto done;
            }

            i = 0;

            gst_bin_add(GST_BIN(pipeline->pipeline),
                        pipeline->demux_instance_bins[i].bin);

            demux_src_pad = gst_element_request_pad_simple(pipeline->demuxer, pad_name);
            NVGSTDS_LINK_ELEMENT_FULL(pipeline->demuxer, pad_name,
                                      pipeline->demux_instance_bins[i].bin, "sink");
            gst_object_unref(demux_src_pad);

            NVGSTDS_ELEM_ADD_PROBE(latency_probe_id,
                                   appCtx->pipeline.demux_instance_bins[i].demux_sink_bin.bin,
                                   "sink",
                                   demux_latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER,
                                   appCtx);
            latency_probe_id = latency_probe_id;
        }

        last_elem = pipeline->demuxer;
        link_element_to_tee_src_pad(pipeline->tiler_tee, last_elem);
        last_elem = pipeline->tiler_tee;
    }

    if (config->tiled_display_config.enable)
    {

        /* Tiler will generate a single composited buffer for all sources. So need
         * to create only one processing instance. */
        if (!create_processing_instance(appCtx, 0))
        {
            goto done;
        }
        // create and add tiling component to pipeline.
        if (config->tiled_display_config.columns *
                config->tiled_display_config.rows <
            config->num_source_sub_bins)
        {
            if (config->tiled_display_config.columns == 0)
            {
                config->tiled_display_config.columns =
                    (guint)(sqrt(config->num_source_sub_bins) + 0.5);
            }
            config->tiled_display_config.rows =
                (guint)ceil(1.0 * config->num_source_sub_bins /
                            config->tiled_display_config.columns);
            NVGSTDS_WARN_MSG_V("Num of Tiles less than number of sources, readjusting to "
                               "%u rows, %u columns",
                               config->tiled_display_config.rows,
                               config->tiled_display_config.columns);
        }

        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->instance_bins[0].bin);
        last_elem = pipeline->instance_bins[0].bin;

        if (!create_tiled_display_bin(&config->tiled_display_config,
                                      &pipeline->tiled_display_bin))
        {
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->tiled_display_bin.bin);
        NVGSTDS_LINK_ELEMENT(pipeline->tiled_display_bin.bin, last_elem);
        last_elem = pipeline->tiled_display_bin.bin;

        link_element_to_tee_src_pad(pipeline->tiler_tee,
                                    pipeline->tiled_display_bin.bin);
        last_elem = pipeline->tiler_tee;

        NVGSTDS_ELEM_ADD_PROBE(latency_probe_id,
                               pipeline->instance_bins->sink_bin.sub_bins[0].sink, "sink",
                               latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, appCtx);
        latency_probe_id = latency_probe_id;
    }
    else
    {
        /*
         * Create demuxer only if tiled display is disabled.
         */
        pipeline->demuxer =
            gst_element_factory_make(NVDS_ELEM_STREAM_DEMUX, "demuxer");
        if (!pipeline->demuxer)
        {
            NVGSTDS_ERR_MSG_V("Failed to create element 'demuxer'");
            goto done;
        }
        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->demuxer);

        for (i = 0; i < config->num_source_sub_bins; i++)
        {
            gchar pad_name[16];
            GstPad *demux_src_pad;

            /* Check if any sink has been configured to render/encode output for
             * source index `i`. The processing instance for that source will be
             * created only if atleast one sink has been configured as such.
             */
            if (!is_sink_available_for_source_id(config, i))
                continue;

            if (!create_processing_instance(appCtx, i))
            {
                goto done;
            }
            gst_bin_add(GST_BIN(pipeline->pipeline),
                        pipeline->instance_bins[i].bin);

            g_snprintf(pad_name, 16, "src_%02d", i);
            demux_src_pad = gst_element_request_pad_simple(pipeline->demuxer, pad_name);
            NVGSTDS_LINK_ELEMENT_FULL(pipeline->demuxer, pad_name,
                                      pipeline->instance_bins[i].bin, "sink");
            gst_object_unref(demux_src_pad);

            for (int k = 0; k < MAX_SINK_BINS; k++)
            {
                if (pipeline->instance_bins[i].sink_bin.sub_bins[k].sink)
                {
                    NVGSTDS_ELEM_ADD_PROBE(latency_probe_id,
                                           pipeline->instance_bins[i].sink_bin.sub_bins[k].sink, "sink",
                                           latency_measurement_buf_prob, GST_PAD_PROBE_TYPE_BUFFER, appCtx);
                    break;
                }
            }

            latency_probe_id = latency_probe_id;
        }
        last_elem = pipeline->demuxer;
    }

    if (config->tiled_display_config.enable == NV_DS_TILED_DISPLAY_DISABLE)
    {
        fps_pad = gst_element_get_static_pad(pipeline->demuxer, "sink");
    }
    else
    {
        fps_pad =
            gst_element_get_static_pad(pipeline->tiled_display_bin.bin, "sink");
    }

    pipeline->common_elements.appCtx = appCtx;
    // Decide where in the pipeline the element should be added and add only if
    // enabled
    if (config->dsexample_config.enable)
    {
        // Create dsexample element bin and set properties
        if (!create_dsexample_bin(&config->dsexample_config,
                                  &pipeline->dsexample_bin))
        {
            goto done;
        }
        // Add dsexample bin to instance bin
        gst_bin_add(GST_BIN(pipeline->pipeline), pipeline->dsexample_bin.bin);

        // Link this bin to the last element in the bin
        NVGSTDS_LINK_ELEMENT(pipeline->dsexample_bin.bin, last_elem);

        // Set this bin as the last element
        last_elem = pipeline->dsexample_bin.bin;
    }

    // create and add common components to pipeline.
    if (!create_common_elements(config, pipeline, &tmp_elem1, &tmp_elem2,
                                bbox_generated_post_analytics_cb))
    {
        goto done;
    }

    if (!add_and_link_broker_sink(appCtx))
    {
        goto done;
    }

    if (tmp_elem2)
    {
        NVGSTDS_LINK_ELEMENT(tmp_elem2, last_elem);
        last_elem = tmp_elem1;
    }

    NVGSTDS_LINK_ELEMENT(pipeline->multi_src_bin.bin, last_elem);

    // enable performance measurement and add call back function to receive
    // performance data.
    if (config->enable_perf_measurement)
    {
        appCtx->perf_struct.context = appCtx;
        if (config->use_nvmultiurisrcbin)
        {
            appCtx->perf_struct.stream_name_display = config->stream_name_display;
            appCtx->perf_struct.use_nvmultiurisrcbin = config->use_nvmultiurisrcbin;
            enable_perf_measurement(&appCtx->perf_struct, fps_pad,
                                    config->max_batch_size,
                                    config->perf_measurement_interval_sec,
                                    config->multi_source_config[0].dewarper_config.num_surfaces_per_frame,
                                    perf_cb);
        }
        else
        {
            enable_perf_measurement(&appCtx->perf_struct, fps_pad,
                                    pipeline->multi_src_bin.num_bins,
                                    config->perf_measurement_interval_sec,
                                    config->multi_source_config[0].dewarper_config.num_surfaces_per_frame,
                                    perf_cb);
        }
    }

    latency_probe_id = latency_probe_id;

    if (config->num_message_consumers)
    {
        for (i = 0; i < config->num_message_consumers; i++)
        {
            /* Pass AppCtx as user data so the subscribe callback can
             * toggle instance-level state correctly. */
            appCtx->c2d_ctx[i] =
                start_cloud_to_device_messaging(&config->message_consumer_config[i],
                                                msg_broker_subscribe_cb, appCtx);
            if (appCtx->c2d_ctx[i] == NULL)
            {
                NVGSTDS_ERR_MSG_V("Failed to create message consumer");
                goto done;
            }
        }
    }

    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(appCtx->pipeline.pipeline),
                                      GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-null");

    g_mutex_init(&appCtx->app_lock);
    g_cond_init(&appCtx->app_cond);
    g_mutex_init(&appCtx->latency_lock);

    ret = TRUE;
done:
    if (fps_pad)
        gst_object_unref(fps_pad);

    if (!ret)
    {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }
    return ret;
}

/**
 * Function to destroy pipeline and release the resources, probes etc.
 */
void destroy_pipeline(AppCtx *appCtx)
{
    gint64 end_time;
    NvDsConfig *config = &appCtx->config;
    guint i;
    GstBus *bus = NULL;

    end_time = g_get_monotonic_time() + G_TIME_SPAN_SECOND;

    if (!appCtx)
        return;

    gst_element_send_event(appCtx->pipeline.pipeline, gst_event_new_eos());
    sleep(1);

    g_mutex_lock(&appCtx->app_lock);
    if (appCtx->pipeline.pipeline)
    {
        destroy_smart_record_bin(&appCtx->pipeline.multi_src_bin);
        bus = gst_pipeline_get_bus(GST_PIPELINE(appCtx->pipeline.pipeline));

        while (TRUE)
        {
            GstMessage *message = gst_bus_pop(bus);
            if (message == NULL || GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS)
                break;
            else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR)
                bus_callback(bus, message, appCtx);
            else
                gst_message_unref(message);
        }
        gst_object_unref(bus);
        gst_element_set_state(appCtx->pipeline.pipeline, GST_STATE_NULL);
    }
    g_cond_wait_until(&appCtx->app_cond, &appCtx->app_lock, end_time);
    g_mutex_unlock(&appCtx->app_lock);

    for (i = 0; i < appCtx->config.num_source_sub_bins; i++)
    {
        NvDsInstanceBin *bin = &appCtx->pipeline.instance_bins[i];
        if (config->osd_config.enable)
        {
            NVGSTDS_ELEM_REMOVE_PROBE(bin->all_bbox_buffer_probe_id,
                                      bin->osd_bin.nvosd, "sink");
        }
        else
        {
            NVGSTDS_ELEM_REMOVE_PROBE(bin->all_bbox_buffer_probe_id,
                                      bin->sink_bin.bin, "sink");
        }

        if (config->primary_gie_config.enable)
        {
            NVGSTDS_ELEM_REMOVE_PROBE(bin->primary_bbox_buffer_probe_id,
                                      bin->primary_gie_bin.bin, "src");
        }
    }
    if (appCtx->latency_info == NULL)
    {
        free(appCtx->latency_info);
        appCtx->latency_info = NULL;
    }
    if (appCtx->sensorInfoHash)
    {
        g_hash_table_destroy(appCtx->sensorInfoHash);
    }
    if (appCtx->perf_struct.FPSInfoHash)
    {
        g_hash_table_destroy(appCtx->perf_struct.FPSInfoHash);
    }
    if (appCtx->custom_msg_data)
    {
        g_free(appCtx->custom_msg_data);
        appCtx->custom_msg_data = NULL;
    }
    
    // 清理 ROI NMS 配置缓存
    if (appCtx->roi_centers)
    {
        g_array_free(appCtx->roi_centers, TRUE);
        appCtx->roi_centers = NULL;
    }
    appCtx->roi_nms_enabled = FALSE;

    destroy_sink_bin();
    g_mutex_clear(&appCtx->latency_lock);

    if (appCtx->pipeline.pipeline)
    {
        bus = gst_pipeline_get_bus(GST_PIPELINE(appCtx->pipeline.pipeline));
        gst_bus_remove_watch(bus);
        gst_object_unref(bus);
        gst_object_unref(appCtx->pipeline.pipeline);
        appCtx->pipeline.pipeline = NULL;
        pause_perf_measurement(&appCtx->perf_struct);

        // for pipeline-recreate, reset rtsp srouce's depay, such as rtph264depay.
        NvDsSrcParentBin *pbin = &appCtx->pipeline.multi_src_bin;
        if (pbin)
        {
            NvDsSrcBin *src_bin;
            for (i = 0; i < MAX_SOURCE_BINS; i++)
            {
                src_bin = &pbin->sub_bins[i];
                if (src_bin && src_bin->config && src_bin->config->type == NV_DS_SOURCE_RTSP)
                {
                    src_bin->depay = NULL;
                }
            }
        }
    }

    if (config->num_message_consumers)
    {
        for (i = 0; i < config->num_message_consumers; i++)
        {
            if (appCtx->c2d_ctx[i])
                stop_cloud_to_device_messaging(appCtx->c2d_ctx[i]);
        }
    }
}

gboolean
pause_pipeline(AppCtx *appCtx)
{
    GstState cur;
    GstState pending;
    GstStateChangeReturn ret;
    GstClockTime timeout = 5 * GST_SECOND / 1000;

    ret =
        gst_element_get_state(appCtx->pipeline.pipeline, &cur, &pending,
                              timeout);

    if (ret == GST_STATE_CHANGE_ASYNC)
    {
        return FALSE;
    }

    if (cur == GST_STATE_PAUSED)
    {
        return TRUE;
    }
    else if (cur == GST_STATE_PLAYING)
    {
        gst_element_set_state(appCtx->pipeline.pipeline, GST_STATE_PAUSED);
        gst_element_get_state(appCtx->pipeline.pipeline, &cur, &pending,
                              GST_CLOCK_TIME_NONE);
        pause_perf_measurement(&appCtx->perf_struct);
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

gboolean
resume_pipeline(AppCtx *appCtx)
{
    GstState cur;
    GstState pending;
    GstStateChangeReturn ret;
    GstClockTime timeout = 5 * GST_SECOND / 1000;

    ret =
        gst_element_get_state(appCtx->pipeline.pipeline, &cur, &pending,
                              timeout);

    if (ret == GST_STATE_CHANGE_ASYNC)
    {
        return FALSE;
    }

    if (cur == GST_STATE_PLAYING)
    {
        return TRUE;
    }
    else if (cur == GST_STATE_PAUSED)
    {
        gst_element_set_state(appCtx->pipeline.pipeline, GST_STATE_PLAYING);
        gst_element_get_state(appCtx->pipeline.pipeline, &cur, &pending,
                              GST_CLOCK_TIME_NONE);
        resume_perf_measurement(&appCtx->perf_struct);
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}
