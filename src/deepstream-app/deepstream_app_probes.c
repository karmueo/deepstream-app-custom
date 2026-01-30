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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "deepstream_app.h"
#include "deepstream_app_callbacks.h"
#include "deepstream_app_probes.h"
#include "nvds_obj_encode.h"

static guint demux_batch_num = 0;  // demux 延迟统计批次计数
static guint64 last_pts = 0;       // 上一次保存对象的时间戳

// 获取目标框中心点与尺寸信息。
//
// Args:
//   obj: 目标元数据指针。
//   cx: 输出中心点 X。
//   cy: 输出中心点 Y。
//   w: 输出宽度。
//   h: 输出高度。
static void get_obj_bbox_center(const NvDsObjectMeta *obj,
                                gfloat *cx,
                                gfloat *cy,
                                gfloat *w,
                                gfloat *h)
{
    gfloat left = obj->rect_params.left;
    gfloat top = obj->rect_params.top;
    gfloat width = obj->rect_params.width;
    gfloat height = obj->rect_params.height;

    *cx = left + width * 0.5f;
    *cy = top + height * 0.5f;
    *w = width;
    *h = height;
}

// 判断目标尺寸是否稳定。
//
// Args:
//   last_size: 上一帧尺寸。
//   curr_size: 当前帧尺寸。
//   ratio_thresh: 尺寸变化比例阈值。
//
// Returns:
//   gboolean: 尺寸变化是否在阈值内。
static gboolean is_size_stable(gfloat last_size, gfloat curr_size, gfloat ratio_thresh)
{
    gfloat base = (last_size > 1.0f) ? last_size : 1.0f;
    return (fabsf(curr_size - last_size) / base) <= ratio_thresh;
}

// 判断目标中心点是否稳定。
//
// Args:
//   last_cx: 上一帧中心点 X。
//   last_cy: 上一帧中心点 Y。
//   cx: 当前帧中心点 X。
//   cy: 当前帧中心点 Y。
//   center_thresh: 中心点变化阈值。
//
// Returns:
//   gboolean: 中心点变化是否在阈值内。
static gboolean is_center_stable(gfloat last_cx,
                                 gfloat last_cy,
                                 gfloat cx,
                                 gfloat cy,
                                 gfloat center_thresh)
{
    return (fabsf(cx - last_cx) <= center_thresh) &&
           (fabsf(cy - last_cy) <= center_thresh);
}

// 重置静态目标过滤状态。
//
// Args:
//   state: 静态目标过滤状态指针。
static void reset_static_target_state(StaticTargetFilterState *state)
{
    state->active = FALSE;
    state->has_last = FALSE;
    state->last_object_id = 0;
    state->static_object_id = 0;
    state->consecutive_count = 0;
}

// 判断目标是否处于静态区域内。
//
// Args:
//   obj: 目标元数据指针。
//   state: 静态目标过滤状态指针。
//   center_thresh: 中心点变化阈值。
//   size_thresh: 尺寸变化阈值。
//
// Returns:
//   gboolean: 是否处于静态区域内。
static gboolean is_obj_in_static_region(const NvDsObjectMeta *obj,
                                        const StaticTargetFilterState *state,
                                        gfloat center_thresh,
                                        gfloat size_thresh)
{
    if (state->static_w <= 0.0f || state->static_h <= 0.0f)
    {
        return FALSE;
    }

    gfloat cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f;
    get_obj_bbox_center(obj, &cx, &cy, &w, &h);

    if (!is_center_stable(state->static_cx, state->static_cy, cx, cy, center_thresh))
    {
        return FALSE;
    }

    if (!is_size_stable(state->static_w, w, size_thresh) ||
        !is_size_stable(state->static_h, h, size_thresh))
    {
        return FALSE;
    }

    return TRUE;
}

// 按 KITTI 格式输出检测框数据。
//
// Args:
//   appCtx: 应用上下文。
//   batch_meta: 批量元数据。
static void write_kitti_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    gchar bbox_file[1024] = {0};
    FILE *bbox_params_dump_file = NULL;

    if (!appCtx->config.bbox_dir_path)
        return;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        guint stream_id = frame_meta->pad_index;
        g_snprintf(bbox_file, sizeof(bbox_file) - 1,
                   "%s/%02u_%03u_%06lu.txt", appCtx->config.bbox_dir_path,
                   appCtx->index, stream_id, (gulong)frame_meta->frame_num);
        bbox_params_dump_file = fopen(bbox_file, "w");
        if (!bbox_params_dump_file)
            continue;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float right = left + obj->rect_params.width;
            float bottom = top + obj->rect_params.height;
            // Here confidence stores detection confidence, since dump gie output
            // is before tracker plugin
            float confidence = obj->confidence;
            fprintf(bbox_params_dump_file,
                    "%s 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                    obj->obj_label, left, top, right, bottom, confidence);
        }
        fclose(bbox_params_dump_file);
    }
}

// 主要检测后对检测框做自定义变换（示例：扩大一倍）。
//
// Args:
//   appCtx: 应用上下文。
//   batch_meta: 批量元数据。
static void change_gieoutput(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    // 遍历每一帧
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // 获取当前帧的宽度和高度（用于边界检查）
        const gint frame_width = frame_meta->source_frame_width;
        const gint frame_height = frame_meta->source_frame_height;

        // 遍历每个检测到的对象
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            // 2. 获取原始矩形参数
            NvOSD_RectParams *rect = &obj_meta->rect_params;
            float original_width = rect->width;
            float original_height = rect->height;

            // 3. 计算原矩形中心点
            float center_x = rect->left + original_width / 2.0f;
            float center_y = rect->top + original_height / 2.0f;

            // 4. 膨胀一倍后的尺寸
            float new_width = original_width * 1.2f;
            float new_height = original_height * 1.2f;

            // 5. 调整左上角坐标（保持中心点不变）
            float new_left = center_x - new_width / 2.0f;
            float new_top = center_y - new_height / 2.0f;

            // 6. 边界检查（防止超出图像范围）
            // new_left = CLAMP(new_left, 0.0f, frame_width - 1.0f);
            // new_top = CLAMP(new_top, 0.0f, frame_height - 1.0f);
            // new_width = CLAMP(new_width, 0.0f, frame_width - new_left);
            // new_height = CLAMP(new_height, 0.0f, frame_height - new_top);

            // 7. 更新矩形参数
            rect->left = new_left;
            rect->top = new_top;
            rect->width = new_width;
            rect->height = new_height;
        }
    }
}

// 输出过去轨迹数据到文件。
//
// Args:
//   appCtx: 应用上下文。
//   batch_meta: 批量元数据。
static void write_kitti_past_track_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    if (!appCtx->config.kitti_track_dir_path)
        return;

    // dump past frame tracked objects appending current frame objects
    gchar bbox_file[1024] = {0};
    FILE *bbox_params_dump_file = NULL;

    NvDsTargetMiscDataBatch *pPastFrameObjBatch = NULL;
    NvDsUserMetaList *bmeta_list = NULL;
    NvDsUserMeta *user_meta = NULL;
    for (bmeta_list = batch_meta->batch_user_meta_list; bmeta_list != NULL;
         bmeta_list = bmeta_list->next)
    {
        user_meta = (NvDsUserMeta *)bmeta_list->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_PAST_FRAME_META)
        {
            pPastFrameObjBatch =
                (NvDsTargetMiscDataBatch *)(user_meta->user_meta_data);
            for (uint si = 0; si < pPastFrameObjBatch->numFilled; si++)
            {
                NvDsTargetMiscDataStream *objStream = (pPastFrameObjBatch->list) + si;
                guint stream_id = (guint)(objStream->streamID);
                for (uint li = 0; li < objStream->numFilled; li++)
                {
                    NvDsTargetMiscDataObject *objList = (objStream->list) + li;
                    for (uint oi = 0; oi < objList->numObj; oi++)
                    {
                        NvDsTargetMiscDataFrame *obj = (objList->list) + oi;
                        g_snprintf(bbox_file, sizeof(bbox_file) - 1,
                                   "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
                                   appCtx->index, stream_id, (gulong)obj->frameNum);

                        float left = obj->tBbox.left;
                        float right = left + obj->tBbox.width;
                        float top = obj->tBbox.top;
                        float bottom = top + obj->tBbox.height;
                        // Past frame object confidence given by tracker
                        float confidence = obj->confidence;
                        bbox_params_dump_file = fopen(bbox_file, "a");
                        if (!bbox_params_dump_file)
                        {
                            continue;
                        }
                        fprintf(bbox_params_dump_file,
                                "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                                objList->objLabel, objList->uniqueId, left, top, right, bottom,
                                confidence);
                        fclose(bbox_params_dump_file);
                    }
                }
            }
        }
    }
}

// 输出包含跟踪 ID 的 KITTI 标签。
//
// Args:
//   appCtx: 应用上下文。
//   batch_meta: 批量元数据。
static void write_kitti_track_output(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    gchar bbox_file[1024] = {0};
    FILE *bbox_params_dump_file = NULL;

    // 给config.kitti_track_dir_path赋个临时的值
    appCtx->config.kitti_track_dir_path = "output";

    if (!appCtx->config.kitti_track_dir_path)
        return;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        guint stream_id = frame_meta->pad_index;
        g_snprintf(bbox_file, sizeof(bbox_file) - 1,
                   "%s/%02u_%03u_%06lu.txt", appCtx->config.kitti_track_dir_path,
                   appCtx->index, stream_id, (gulong)frame_meta->frame_num);
        bbox_params_dump_file = fopen(bbox_file, "w");
        if (!bbox_params_dump_file)
            continue;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            float left = obj->tracker_bbox_info.org_bbox_coords.left;
            float top = obj->tracker_bbox_info.org_bbox_coords.top;
            float right = left + obj->tracker_bbox_info.org_bbox_coords.width;
            float bottom = top + obj->tracker_bbox_info.org_bbox_coords.height;
            // Here confidence stores tracker confidence value for tracker output
            float confidence = obj->tracker_confidence;
            guint64 id = obj->object_id;
            bool write_proj_info = false;
            float visibility = -1.0, x_img_foot = -1.0, y_img_foot = -1.0;
            // Attach projected object info if stored in user meta
            for (NvDsUserMetaList *l_obj_user = obj->obj_user_meta_list; l_obj_user != NULL;
                 l_obj_user = l_obj_user->next)
            {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_obj_user->data;
                if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_VISIBILITY && user_meta->user_meta_data)
                {
                    write_proj_info = true;
                    visibility = *((float *)(user_meta->user_meta_data));
                }
                else if (user_meta && user_meta->base_meta.meta_type == NVDS_OBJ_IMAGE_FOOT_LOCATION && user_meta->user_meta_data)
                {
                    write_proj_info = true;
                    x_img_foot = ((float *)(user_meta->user_meta_data))[0];
                    y_img_foot = ((float *)(user_meta->user_meta_data))[1];
                }
            }

            if (write_proj_info)
            {
                fprintf(bbox_params_dump_file,
                        "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f %f %f %f\n",
                        obj->obj_label, id, left, top, right, bottom, confidence, visibility, x_img_foot, y_img_foot);
            }
            else
            {
                fprintf(bbox_params_dump_file,
                        "%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n",
                        obj->obj_label, id, left, top, right, bottom, confidence);
            }
        }
        fclose(bbox_params_dump_file);
    }
}

// 分类器组件排序比较函数。
//
// Args:
//   a: 第一个比较元素。
//   b: 第二个比较元素。
//
// Returns:
//   gint: 比较结果。
static gint component_id_compare_func(gconstpointer a, gconstpointer b)
{
    NvDsClassifierMeta *cmetaa = (NvDsClassifierMeta *)a;
    NvDsClassifierMeta *cmetab = (NvDsClassifierMeta *)b;

    if (cmetaa->unique_component_id < cmetab->unique_component_id)
        return -1;
    if (cmetaa->unique_component_id > cmetab->unique_component_id)
        return 1;
    return 0;
}

// 处理附加元数据（显示文字、颜色等）。
//
// Args:
//   appCtx: 应用上下文。
//   batch_meta: 批量元数据。
static void process_meta(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    // For single source always display text either with demuxer or with tiler
    if (!appCtx->config.tiled_display_config.enable ||
        appCtx->config.num_source_sub_bins == 1)
    {
        appCtx->show_bbox_text = 1;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
            gint class_index = obj->class_id;
            NvDsGieConfig *gie_config = NULL;
            gchar *str_ins_pos = NULL;

            if (obj->unique_component_id ==
                (gint)appCtx->config.primary_gie_config.unique_id)
            {
                gie_config = &appCtx->config.primary_gie_config;
            }
            else
            {
                for (gint i = 0; i < (gint)appCtx->config.num_secondary_gie_sub_bins;
                     i++)
                {
                    gie_config = &appCtx->config.secondary_gie_sub_bin_config[i];
                    if (obj->unique_component_id == (gint)gie_config->unique_id)
                    {
                        break;
                    }
                    gie_config = NULL;
                }
            }
            g_free(obj->text_params.display_text);
            obj->text_params.display_text = NULL;

            if (gie_config != NULL)
            {
                if (g_hash_table_contains(gie_config->bbox_border_color_table,
                                          class_index + (gchar *)NULL))
                {
                    obj->rect_params.border_color = *((NvOSD_ColorParams *)
                                                          g_hash_table_lookup(gie_config->bbox_border_color_table,
                                                                              class_index + (gchar *)NULL));
                }
                else
                {
                    obj->rect_params.border_color = gie_config->bbox_border_color;
                }
                obj->rect_params.border_width = appCtx->config.osd_config.border_width;

                if (g_hash_table_contains(gie_config->bbox_bg_color_table,
                                          class_index + (gchar *)NULL))
                {
                    obj->rect_params.has_bg_color = 1;
                    obj->rect_params.bg_color = *((NvOSD_ColorParams *)
                                                      g_hash_table_lookup(gie_config->bbox_bg_color_table,
                                                                          class_index + (gchar *)NULL));
                }
                else
                {
                    obj->rect_params.has_bg_color = 0;
                }
            }

            if (!appCtx->show_bbox_text)
                continue;

            obj->text_params.x_offset = obj->rect_params.left;
            obj->text_params.y_offset = obj->rect_params.top - 30;
            obj->text_params.font_params.font_color =
                appCtx->config.osd_config.text_color;
            obj->text_params.font_params.font_size =
                appCtx->config.osd_config.text_size;
            obj->text_params.font_params.font_name = appCtx->config.osd_config.font;
            if (appCtx->config.osd_config.text_has_bg)
            {
                obj->text_params.set_bg_clr = 1;
                obj->text_params.text_bg_clr = appCtx->config.osd_config.text_bg_color;
            }

            obj->text_params.display_text = (char *)g_malloc(128);
            obj->text_params.display_text[0] = '\0';
            str_ins_pos = obj->text_params.display_text;

            if (obj->obj_label[0] != '\0')
                sprintf(str_ins_pos, "%s", obj->obj_label);
            str_ins_pos += strlen(str_ins_pos);

            if (obj->object_id != UNTRACKED_OBJECT_ID)
            {
                /** object_id is a 64-bit sequential value;
                 * but considering the display aesthetic,
                 * trimming to lower 32-bits */
                if (appCtx->config.tracker_config.display_tracking_id)
                {
                    guint64 const LOW_32_MASK = 0x00000000FFFFFFFF;
                    sprintf(str_ins_pos, " %lu", (obj->object_id & LOW_32_MASK));
                    str_ins_pos += strlen(str_ins_pos);
                }
            }

            obj->classifier_meta_list =
                g_list_sort(obj->classifier_meta_list, component_id_compare_func);
            for (NvDsMetaList *l_class = obj->classifier_meta_list; l_class != NULL;
                 l_class = l_class->next)
            {
                NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *)l_class->data;
                for (NvDsMetaList *l_label = cmeta->label_info_list; l_label != NULL;
                     l_label = l_label->next)
                {
                    NvDsLabelInfo *label = (NvDsLabelInfo *)l_label->data;
                    if (label->pResult_label)
                    {
                        sprintf(str_ins_pos, " %s", label->pResult_label);
                    }
                    else if (label->result_label[0] != '\0')
                    {
                        sprintf(str_ins_pos, " %s", label->result_label);
                    }
                    str_ins_pos += strlen(str_ins_pos);
                }
            }
        }
    }
}

// 处理推理缓冲区与元数据，并触发应用层回调。
//
// Args:
//   buf: GstBuffer。
//   appCtx: 应用上下文。
//   index: 源索引。
static void process_buffer(GstBuffer *buf, AppCtx *appCtx, guint index)
{
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
    {
        NVGSTDS_WARN_MSG_V("Batch meta not found for buffer %p", buf);
        return;
    }
    process_meta(appCtx, batch_meta);
    // NvDsInstanceData *data = &appCtx->instance_data[index];
    // guint i;

    //  data->frame_num++;

    /* Opportunity to modify the processed metadata or do analytics based on
     * type of object e.g. maintaining count of particular type of car.
     * 这里会调用all_bbox_generated_cb回调函数，这个函数的实现在deepstream_app_main.c中
     * 在这个回调函数中可以修改处理后的元数据或基于检测结果进行分析
     */
    if (appCtx->all_bbox_generated_cb)
    {
        appCtx->all_bbox_generated_cb(appCtx, buf, batch_meta, index);
    }
    // data->bbox_list_size = 0;

    /*
     * 回调函数以附加特定于应用程序的附加元数据。
     */
    if (appCtx->overlay_graphics_cb)
    {
        appCtx->overlay_graphics_cb(appCtx, buf, batch_meta, index);
    }
}

// 主推理完成后的探针回调，提供 NMS、框修正与可选保存。
//
// Args:
//   pad: 触发探针的 pad。
//   info: 探针信息，包含 GstBuffer 等数据。
//   u_data: 用户数据，通常为 AppCtx 指针。
//
// Returns:
//   GstPadProbeReturn: 继续或丢弃 buffer 的处理结果。
GstPadProbeReturn gie_primary_processing_done_buf_prob(GstPad *pad,
                                                       GstPadProbeInfo *info,
                                                       gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    AppCtx *appCtx = (AppCtx *)u_data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_obj = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
    {
        NVGSTDS_WARN_MSG_V("Batch meta not found for buffer %p", buf);
        return GST_PAD_PROBE_OK;
    }

    // ==== 可选：在修改框尺寸/写出之前执行 NMS 去重 ====
    // 使用缓存的 ROI 配置（已在初始化时读取）
    gboolean use_roi_nms = appCtx->roi_nms_enabled;
    GArray *roi_centers = appCtx->roi_centers;

    // TODO: 简单 NMS: 按置信度排序，IoU>阈值则删除低置信度框（不再按类别区分，跨类别也抑制）
    // 如果启用了 ROI，则在 IoU 相同情况下优先保留离 ROI 中心点更近的框
    const gfloat IOU_THRESH = 0.01f;  // 可调，>0 且 <1

    // 统计 & 处理每个帧
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        // 收集对象指针
        GArray *objs = g_array_new(FALSE, FALSE, sizeof(NvDsObjectMeta *));
        for (NvDsMetaList *l = frame_meta->obj_meta_list; l; l = l->next)
        {
            NvDsObjectMeta *o = (NvDsObjectMeta *)l->data;
            g_array_append_val(objs, o);
        }
        guint n = objs->len;
        if (n > 1)
        {
            // 简单冒泡/插入排序，按 confidence 降序 (对象数量通常较少; 若多可改为快速排序)
            for (guint i = 1; i < n; i++)
            {
                NvDsObjectMeta *key = g_array_index(objs, NvDsObjectMeta *, i);
                gfloat kc = key->confidence;
                gint j = i - 1;
                while (j >= 0)
                {
                    NvDsObjectMeta *pj = g_array_index(objs, NvDsObjectMeta *, j);
                    if (pj->confidence >= kc)
                        break;
                    g_array_index(objs, NvDsObjectMeta *, j + 1) = pj;
                    j--;
                }
                g_array_index(objs, NvDsObjectMeta *, j + 1) = key;
            }
            // 标记保留（安全初始化）
            GByteArray *keep = g_byte_array_sized_new(n);
            g_byte_array_set_size(keep, n);
            memset(keep->data, 1, n);
            for (guint i = 0; i < n; i++)
            {
                if (!keep->data[i])
                    continue;
                NvDsObjectMeta *a = g_array_index(objs, NvDsObjectMeta *, i);
                float ax1 = a->rect_params.left;
                float ay1 = a->rect_params.top;
                float aw = a->rect_params.width;
                float ah = a->rect_params.height;
                float ax2 = ax1 + aw;
                float ay2 = ay1 + ah;
                if (aw <= 0 || ah <= 0)
                {
                    keep->data[i] = 0;
                    continue;
                }

                // 计算 a 的中心点
                float acx = ax1 + aw / 2.0f;
                float acy = ay1 + ah / 2.0f;

                for (guint j = i + 1; j < n; j++)
                {
                    if (!keep->data[j])
                        continue;
                    NvDsObjectMeta *b = g_array_index(objs, NvDsObjectMeta *, j);
                    // 跨类别也进行抑制（去掉按类别过滤）
                    float bx1 = b->rect_params.left;
                    float by1 = b->rect_params.top;
                    float bw = b->rect_params.width;
                    float bh = b->rect_params.height;
                    if (bw <= 0 || bh <= 0)
                    {
                        keep->data[j] = 0;
                        continue;
                    }
                    float bx2 = bx1 + bw;
                    float by2 = by1 + bh;
                    float ix1 = ax1 > bx1 ? ax1 : bx1;
                    float iy1 = ay1 > by1 ? ay1 : by1;
                    float ix2 = ax2 < bx2 ? ax2 : bx2;
                    float iy2 = ay2 < by2 ? ay2 : by2;
                    float iw = ix2 - ix1;
                    float ih = iy2 - iy1;
                    if (iw <= 0 || ih <= 0)
                        continue;
                    float inter = iw * ih;
                    float uni = aw * ah + bw * bh - inter;
                    float iou = (uni <= 0.f) ? 0.f : inter / uni;

                    if (iou > IOU_THRESH)
                    {
                        // 如果启用了 ROI-based NMS，需要比较哪个框离 ROI 中心点更近
                        gboolean suppress_j = TRUE;

                        if (use_roi_nms && roi_centers && roi_centers->len >= 2)
                        {
                            // 计算 b 的中心点
                            float bcx = bx1 + bw / 2.0f;
                            float bcy = by1 + bh / 2.0f;

                            // 找到离 a 和 b 最近的 ROI 中心点
                            gfloat min_dist_a = G_MAXFLOAT;
                            gfloat min_dist_b = G_MAXFLOAT;

                            for (guint k = 0; k < roi_centers->len / 2; k++)
                            {
                                gfloat roi_cx = g_array_index(roi_centers, gfloat, k * 2);
                                gfloat roi_cy = g_array_index(roi_centers, gfloat, k * 2 + 1);

                                gfloat dist_a =
                                    sqrtf((acx - roi_cx) * (acx - roi_cx) +
                                          (acy - roi_cy) * (acy - roi_cy));
                                gfloat dist_b =
                                    sqrtf((bcx - roi_cx) * (bcx - roi_cx) +
                                          (bcy - roi_cy) * (bcy - roi_cy));

                                if (dist_a < min_dist_a)
                                    min_dist_a = dist_a;
                                if (dist_b < min_dist_b)
                                    min_dist_b = dist_b;
                            }

                            // 如果 b 离 ROI 中心点更近，则抑制 a 而不是 b
                            if (min_dist_b < min_dist_a)
                            {
                                suppress_j = FALSE;
                                keep->data[i] = 0;  // 抑制 a
                                break;              // a 已被抑制，退出内层循环
                            }
                        }

                        if (suppress_j)
                        {
                            keep->data[j] = 0;  // 抑制低置信度的或离 ROI 中心点更远的
                        }
                    }
                }
            }
            // 删除被抑制的 meta
            for (guint i = 0; i < n; i++)
            {
                if (!keep->data[i])
                {
                    NvDsObjectMeta *o = g_array_index(objs, NvDsObjectMeta *, i);
                    nvds_remove_obj_meta_from_frame(frame_meta, o);
                }
            }
            g_byte_array_unref(keep);
        }
        g_array_free(objs, TRUE);
    }

    // 注意：不要在这里释放 roi_centers，它是 AppCtx 的缓存数据

    // 边界 clamp + 最小尺寸过滤 (>=4) 确保后续 SGIE/OSD 安全
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        gint frame_w = frame_meta->source_frame_width;
        gint frame_h = frame_meta->source_frame_height;
        if (frame_w <= 0 || frame_h <= 0)
            continue;
        for (NvDsMetaList *l_obj_it = frame_meta->obj_meta_list; l_obj_it != NULL;)
        {
            NvDsMetaList *l_next = l_obj_it->next;
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj_it->data;
            NvOSD_RectParams *r = &obj->rect_params;
            if (r->left < 0)
                r->left = 0;
            if (r->top < 0)
                r->top = 0;
            if (r->width < 0)
                r->width = 0;
            if (r->height < 0)
                r->height = 0;
            if (r->left + r->width > frame_w)
            {
                r->width = frame_w - r->left;
                if (r->width < 0)
                    r->width = 0;
            }
            if (r->top + r->height > frame_h)
            {
                r->height = frame_h - r->top;
                if (r->height < 0)
                    r->height = 0;
            }
            if (r->width < 4 || r->height < 4)
            {
                nvds_remove_obj_meta_from_frame(frame_meta, obj);
            }
            l_obj_it = l_next;
        }
    }

    // 后续自定义变换（示例：把检测框扩大一倍）
    // change_gieoutput(appCtx, batch_meta);

#ifdef ENABLE_OBJ_SAVE
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ))
    {
        GST_ERROR("input buffer mapinfo failed");
        return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *)inmap.data;
    gst_buffer_unmap(buf, &inmap);

    // 后面要保存jpg图片需要用到
    // obj_ctx_handle = appCtx->obj_ctx_handle;

    NvDsMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        if ((frame_meta->buf_pts - last_pts) > 1 * GST_SECOND || last_pts == 0)
        {
            guint num_rects = 0;
            for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
            {
                num_rects++;
                obj_meta = (NvDsObjectMeta *)(l_obj->data);
                /* Conditions that user needs to set to encode the detected objects of
                 * interest. Here, by default all the detected objects are encoded.
                 * For demonstration, we will encode the first object in the frame. */
                // 如果检测到目标，并且和上一次保存的时间相差超过3秒，则保存图片
                if ((obj_meta->class_id >= 0))
                {
                    NvDsObjEncUsrArgs frameData = {0};
                    // frameData.isFrame = 1;
                    /* To be set by user */
                    frameData.saveImg = 1;
                    frameData.attachUsrMeta = TRUE;
                    /* Set if Image scaling Required */
                    frameData.scaleImg = FALSE;
                    frameData.scaledWidth = 0;
                    frameData.scaledHeight = 0;
                    /* Quality */
                    frameData.quality = 100;
                    frameData.objNum = num_rects;
                    /* Set to calculate time taken to encode JPG image. */
                    frameData.calcEncodeTime = 0;
                    /*Main Function Call */
                    nvds_obj_enc_process((gpointer)appCtx->obj_ctx_handle,
                                         &frameData,
                                         ip_surf,
                                         obj_meta,
                                         frame_meta);
                    last_pts = frame_meta->buf_pts;
                    break;
                }
            }
            nvds_obj_enc_finish((gpointer)appCtx->obj_ctx_handle);
            break;
        }
    }
#endif

    return GST_PAD_PROBE_OK;
}

// 全部推理完成后的探针回调，进入 OSD 或 sink 前执行。
//
// Args:
//   pad: 触发探针的 pad。
//   info: 探针信息，包含 GstBuffer 等数据。
//   u_data: 用户数据，通常为 NvDsInstanceBin 指针。
//
// Returns:
//   GstPadProbeReturn: 继续或丢弃 buffer 的处理结果。
GstPadProbeReturn gie_processing_done_buf_prob(GstPad *pad,
                                               GstPadProbeInfo *info,
                                               gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsInstanceBin *bin = (NvDsInstanceBin *)u_data;
    guint index = bin->index;
    AppCtx *appCtx = bin->appCtx;

    if (gst_buffer_is_writable(buf))
        process_buffer(buf, appCtx, index);
    return GST_PAD_PROBE_OK;
}

// 跟踪之后的 buffer 探针回调。
//
// Args:
//   pad: 触发探针的 pad。
//   info: 探针信息，包含 GstBuffer 等数据。
//   u_data: 用户数据，通常为 NvDsInstanceBin 指针。
//
// Returns:
//   GstPadProbeReturn: 继续或丢弃 buffer 的处理结果。
GstPadProbeReturn analytics_done_buf_prob(GstPad *pad,
                                          GstPadProbeInfo *info,
                                          gpointer u_data)
{
    NvDsInstanceBin *bin = (NvDsInstanceBin *)u_data;
    guint index = bin->index;
    AppCtx *appCtx = bin->appCtx;
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
    {
        NVGSTDS_WARN_MSG_V("Batch meta not found for buffer %p", buf);
        return GST_PAD_PROBE_OK;
    }

    if (appCtx->config.tracker_config.enable &&
        appCtx->config.tracker_config.enable_static_target_filter)
    {
        NvDsTrackerConfig *tracker_cfg = &appCtx->config.tracker_config;
        guint frames_needed = tracker_cfg->static_target_filter_frames;
        gfloat center_thresh = tracker_cfg->static_target_filter_center_thresh;
        gfloat size_thresh = tracker_cfg->static_target_filter_size_thresh;

        if (frames_needed == 0)
        {
            frames_needed = 1;
        }
        if (center_thresh < 0.0f)
        {
            center_thresh = 0.0f;
        }
        if (size_thresh < 0.0f)
        {
            size_thresh = 0.0f;
        }

        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
             l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
            if (frame_meta->source_id >= MAX_SOURCE_BINS)
            {
                continue;
            }

            StaticTargetFilterState *state =
                &appCtx->static_target_filter_states[frame_meta->source_id];

            NvDsObjectMeta *tracked_obj = NULL;
            for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                 l_obj = l_obj->next)
            {
                NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
                if (obj->object_id != UNTRACKED_OBJECT_ID)
                {
                    tracked_obj = obj;
                    break;
                }
            }

            gboolean new_target_detected = FALSE;
            if (state->active && tracked_obj &&
                tracked_obj->object_id != state->static_object_id)
            {
                new_target_detected = TRUE;
                reset_static_target_state(state);
            }

            if (!state->active)
            {
                if (tracked_obj)
                {
                    gfloat cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f;
                    get_obj_bbox_center(tracked_obj, &cx, &cy, &w, &h);

                    if (state->has_last &&
                        tracked_obj->object_id == state->last_object_id &&
                        is_center_stable(state->last_cx, state->last_cy, cx, cy, center_thresh) &&
                        is_size_stable(state->last_w, w, size_thresh) &&
                        is_size_stable(state->last_h, h, size_thresh))
                    {
                        state->consecutive_count++;
                    }
                    else
                    {
                        state->consecutive_count = 1;
                    }

                    state->has_last = TRUE;
                    state->last_object_id = tracked_obj->object_id;
                    state->last_cx = cx;
                    state->last_cy = cy;
                    state->last_w = w;
                    state->last_h = h;

                    if (state->consecutive_count >= frames_needed)
                    {
                        state->active = TRUE;
                        state->static_object_id = tracked_obj->object_id;
                        state->static_cx = cx;
                        state->static_cy = cy;
                        state->static_w = w;
                        state->static_h = h;
                    }
                }
                else
                {
                    state->has_last = FALSE;
                    state->consecutive_count = 0;
                    state->last_object_id = 0;
                }
            }

            if (state->active && !new_target_detected)
            {
                for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;)
                {
                    NvDsMetaList *l_next = l_obj->next;
                    NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
                    if (is_obj_in_static_region(obj, state, center_thresh, size_thresh))
                    {
                        nvds_remove_obj_meta_from_frame(frame_meta, obj);
                    }
                    l_obj = l_next;
                }
            }
        }
    }

    // 仅在启用了 tracker 时，过滤掉未被跟踪的检测目标
    // if (appCtx->config.tracker_config.enable)
    // {
    //     // 注意：UNTRACKED_OBJECT_ID 表示尚未关联到跟踪轨迹的检测框
    //     for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    //     {
    //         NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
    //         for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;)
    //         {
    //             NvDsMetaList *l_next = l_obj->next; // 删除节点前先保存 next
    //             NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;
    //             if (obj->object_id == UNTRACKED_OBJECT_ID)
    //             {
    //                 // 从当前帧移除未跟踪目标
    //                 nvds_remove_obj_meta_from_frame(frame_meta, obj);
    //             }
    //             l_obj = l_next;
    //         }
    //     }
    // }

    /*
     * Output KITTI labels with tracking ID if configured to do so.
     */
    write_kitti_track_output(appCtx, batch_meta);
    write_kitti_past_track_output(appCtx, batch_meta);
    // write_reid_track_output(appCtx, batch_meta);
    // write_terminated_track_output(appCtx, batch_meta);
    // write_shadow_track_output(appCtx, batch_meta);

    if (appCtx->bbox_generated_post_analytics_cb)
    {
        appCtx->bbox_generated_post_analytics_cb(appCtx, buf, batch_meta, index);
    }
    return GST_PAD_PROBE_OK;
}

// 主 pipeline 延迟统计的探针回调。
//
// Args:
//   pad: 触发探针的 pad。
//   info: 探针信息，包含 GstBuffer 等数据。
//   u_data: 用户数据，通常为 AppCtx 指针。
//
// Returns:
//   GstPadProbeReturn: 继续或丢弃 buffer 的处理结果。
GstPadProbeReturn latency_measurement_buf_prob(GstPad *pad,
                                               GstPadProbeInfo *info,
                                               gpointer u_data)
{
    AppCtx *appCtx = (AppCtx *)u_data;
    guint i = 0, num_sources_in_batch = 0;
    if (nvds_enable_latency_measurement)
    {
        GstBuffer *buf = (GstBuffer *)info->data;
        NvDsFrameLatencyInfo *latency_info = NULL;
        g_mutex_lock(&appCtx->latency_lock);
        latency_info = appCtx->latency_info;
        guint64 batch_num = GPOINTER_TO_SIZE(g_object_get_data(G_OBJECT(pad), "latency-batch-num"));
        g_print("\n************BATCH-NUM = %lu**************\n", batch_num);

        num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

        for (i = 0; i < num_sources_in_batch; i++)
        {
            g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
                    latency_info[i].source_id,
                    latency_info[i].frame_num, latency_info[i].latency);
        }
        g_mutex_unlock(&appCtx->latency_lock);
        g_object_set_data(G_OBJECT(pad), "latency-batch-num", GSIZE_TO_POINTER(batch_num + 1));
    }

    return GST_PAD_PROBE_OK;
}

// demux 分支延迟统计的探针回调。
//
// Args:
//   pad: 触发探针的 pad。
//   info: 探针信息，包含 GstBuffer 等数据。
//   u_data: 用户数据，通常为 AppCtx 指针。
//
// Returns:
//   GstPadProbeReturn: 继续或丢弃 buffer 的处理结果。
GstPadProbeReturn demux_latency_measurement_buf_prob(GstPad *pad,
                                                     GstPadProbeInfo *info,
                                                     gpointer u_data)
{
    AppCtx *appCtx = (AppCtx *)u_data;
    guint i = 0, num_sources_in_batch = 0;
    if (nvds_enable_latency_measurement)
    {
        GstBuffer *buf = (GstBuffer *)info->data;
        NvDsFrameLatencyInfo *latency_info = NULL;
        g_mutex_lock(&appCtx->latency_lock);
        latency_info = appCtx->latency_info;
        g_print("\n************DEMUX BATCH-NUM = %d**************\n",
                demux_batch_num);
        num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

        for (i = 0; i < num_sources_in_batch; i++)
        {
            g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
                    latency_info[i].source_id,
                    latency_info[i].frame_num, latency_info[i].latency);
        }
        g_mutex_unlock(&appCtx->latency_lock);
        demux_batch_num++;
    }

    return GST_PAD_PROBE_OK;
}
