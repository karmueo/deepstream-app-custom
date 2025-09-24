/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"
#include "gst-nvdssr.h"
#include "nvds_version.h"
#include "nvdsmeta_schema.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <cuda_runtime_api.h>
#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#ifdef EN_DEBUG
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

#define MAX_INSTANCES 128
#define MAX_TIME_STAMP_LEN (64)
#define APP_TITLE "DeepStream"

#define DEFAULT_X_WINDOW_WIDTH 1920
#define DEFAULT_X_WINDOW_HEIGHT 1080

AppCtx           *appCtx[MAX_INSTANCES];
static guint      cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar    **cfg_files = NULL;
static gchar    **input_uris = NULL;
static gboolean   print_version = FALSE;
static gboolean   show_bbox_text = FALSE;
static gboolean   print_dependencies_version = FALSE;
static gboolean   quit = FALSE;
static gint       return_value = 0;
static guint      num_instances;
static guint      num_input_uris;
static GMutex     fps_lock;
static gdouble    fps[MAX_SOURCE_BINS];
static gdouble    fps_avg[MAX_SOURCE_BINS];

static Display *display = NULL;
static Window   windows[MAX_INSTANCES] = {0};

static GThread *x_event_thread = NULL;
static GMutex   disp_lock;

static guint    rrow, rcol, rcfg;
static gboolean rrowsel = FALSE, selecting = FALSE;

static gint frame_number = 0;
static gint last_save_frame_number = 0;

GST_DEBUG_CATEGORY(NVDS_APP);

GOptionEntry entries[] = {
    {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
     "Print DeepStreamSDK version", NULL},
    {"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
     "Display Bounding box labels in tiled mode", NULL},
    {"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
     "Print DeepStreamSDK and dependencies version", NULL},
    {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
     "Set the config file", NULL},
    {"input-uri", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_uris,
     "Set the input uri (file://stream or rtsp://stream)", NULL},
    {NULL},
};

static gboolean g_pending_request = FALSE; // 是否有待处理的请求

// NTP时间戳转Unix时间戳（秒.小数）
static double ntp_to_unix(uint64_t ntp_timestamp)
{
    const uint32_t NTP_TO_UNIX = 2208988800U; // 1900~1970的秒差

    // 提取高32位（秒）和低32位（小数）
    uint32_t seconds = (uint32_t)(ntp_timestamp >> 32);
    uint32_t fraction = (uint32_t)(ntp_timestamp & 0xFFFFFFFF);

    // 关键步骤：确保无符号运算避免溢出
    double unix_seconds = (double)(seconds - NTP_TO_UNIX) // 先转double再减
                          + (double)fraction / (1ULL << 32); // 小数部分

    return unix_seconds;
}

// 冷却结束回调
static gboolean on_cooldown_end()
{
    if (g_pending_request)
    {
        g_pending_request = FALSE;
    }
    return G_SOURCE_REMOVE;
}

static gboolean smart_record_event_generator(NvDsSrcBin *src_bin)
{
    if (g_pending_request)
    {
        return FALSE;
    }

    NvDsSRSessionId sessId = 0;
    guint           startTime = 7;
    guint           duration = 8;

    if (src_bin->config->smart_rec_duration >= 0)
        duration = src_bin->config->smart_rec_duration;

    if (src_bin->config->smart_rec_start_time >= 0)
        startTime = src_bin->config->smart_rec_start_time;

    if (src_bin->recordCtx && !src_bin->reconfiguring)
    {
        NvDsSRContext *ctx = (NvDsSRContext *)src_bin->recordCtx;
        if (ctx->recordOn)
        {
            NvDsSRStop(ctx, 0);
        }
        NvDsSRStart(ctx, &sessId, startTime, duration, NULL);
        g_pending_request = TRUE;
        g_timeout_add(30000, on_cooldown_end, NULL);
    }

    return TRUE;
}

static gpointer meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta     *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta *)g_memdup2(srcMeta, sizeof(NvDsEventMsgMeta));

    if (srcMeta->ts)
        dstMeta->ts = g_strdup(srcMeta->ts);

    if (srcMeta->objSignature.size > 0)
    {
        dstMeta->objSignature.signature = (gdouble *)g_memdup2(
            srcMeta->objSignature.signature, srcMeta->objSignature.size);
        dstMeta->objSignature.size = srcMeta->objSignature.size;
    }

    if (srcMeta->objectId)
    {
        dstMeta->objectId = g_strdup(srcMeta->objectId);
    }

    if (srcMeta->sensorStr)
    {
        dstMeta->sensorStr = g_strdup(srcMeta->sensorStr);
    }

    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE)
        {
            NvDsVehicleObject *srcObj = (NvDsVehicleObject *)srcMeta->extMsg;
            NvDsVehicleObject *obj =
                (NvDsVehicleObject *)g_malloc0(sizeof(NvDsVehicleObject));
            if (srcObj->type)
                obj->type = g_strdup(srcObj->type);
            if (srcObj->make)
                obj->make = g_strdup(srcObj->make);
            if (srcObj->model)
                obj->model = g_strdup(srcObj->model);
            if (srcObj->color)
                obj->color = g_strdup(srcObj->color);
            if (srcObj->license)
                obj->license = g_strdup(srcObj->license);
            if (srcObj->region)
                obj->region = g_strdup(srcObj->region);

            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(NvDsVehicleObject);
        }
        else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON)
        {
            NvDsPersonObject *srcObj = (NvDsPersonObject *)srcMeta->extMsg;
            NvDsPersonObject *obj =
                (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));

            obj->age = srcObj->age;

            if (srcObj->gender)
                obj->gender = g_strdup(srcObj->gender);
            if (srcObj->cap)
                obj->cap = g_strdup(srcObj->cap);
            if (srcObj->hair)
                obj->hair = g_strdup(srcObj->hair);
            if (srcObj->apparel)
                obj->apparel = g_strdup(srcObj->apparel);

            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(NvDsPersonObject);
        }
    }

    return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta     *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    user_meta->user_meta_data = NULL;

    if (srcMeta->ts)
    {
        g_free(srcMeta->ts);
    }

    if (srcMeta->objSignature.size > 0)
    {
        g_free(srcMeta->objSignature.signature);
        srcMeta->objSignature.size = 0;
    }

    if (srcMeta->objectId)
    {
        g_free(srcMeta->objectId);
    }

    if (srcMeta->sensorStr)
    {
        g_free(srcMeta->sensorStr);
    }

    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE)
        {
            NvDsVehicleObject *obj = (NvDsVehicleObject *)srcMeta->extMsg;
            if (obj->type)
                g_free(obj->type);
            if (obj->color)
                g_free(obj->color);
            if (obj->make)
                g_free(obj->make);
            if (obj->model)
                g_free(obj->model);
            if (obj->license)
                g_free(obj->license);
            if (obj->region)
                g_free(obj->region);
        }
        else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON)
        {
            NvDsPersonObject *obj = (NvDsPersonObject *)srcMeta->extMsg;

            if (obj->gender)
                g_free(obj->gender);
            if (obj->cap)
                g_free(obj->cap);
            if (obj->hair)
                g_free(obj->hair);
            if (obj->apparel)
                g_free(obj->apparel);
        }
        g_free(srcMeta->extMsg);
        srcMeta->extMsg = NULL;
        srcMeta->extMsgSize = 0;
    }
    g_free(srcMeta);
}

/** TODO: 待实现
 * @brief 解析接收的消息消息
 *
 * @param data 消息数据
 * @param size 消息大小
 */
static void parse_cloud_message(gpointer data, guint size)
{
    JsonNode *rootNode = NULL;
    GError   *error = NULL;
    gchar    *sensorStr = NULL;
    gint      start, duration;
    gboolean  startRec, ret;

    /**
     * Following minimum json message is expected to trigger the start / stop
     * of smart record.
     * {
     *   command: string   // <start-recording / stop-recording>
     *   start: string     // "2020-05-18T20:02:00.051Z"
     *   end: string       // "2020-05-18T20:02:02.851Z",
     *   sensor: {
     *     id: string
     *   }
     * }
     */
    JsonParser *parser = json_parser_new();
    ret = json_parser_load_from_data(parser, data, size, &error);
    if (!ret)
    {
        NVGSTDS_ERR_MSG_V("Error in parsing json message %s", error->message);
        g_error_free(error);
        g_object_unref(parser);
        return;
    }

    rootNode = json_parser_get_root(parser);
    if (JSON_NODE_HOLDS_OBJECT(rootNode))
    {
        JsonObject *object;

        object = json_node_get_object(rootNode);
        if (json_object_has_member(object, "command"))
        {
            const gchar *type =
                json_object_get_string_member(object, "command");
            if (!g_strcmp0(type, "start-recording"))
                startRec = TRUE;
            else if (!g_strcmp0(type, "stop-recording"))
                startRec = FALSE;
            else
            {
                NVGSTDS_WARN_MSG_V("wrong command %s", type);
                goto error;
            }
        }
        else
        {
            // 'command' field not provided, assume it to be start-recording.
            startRec = TRUE;
        }

        if (json_object_has_member(object, "sensor"))
        {
            JsonObject *tempObj =
                json_object_get_object_member(object, "sensor");
            if (json_object_has_member(tempObj, "id"))
            {
                sensorStr =
                    g_strdup(json_object_get_string_member(tempObj, "id"));
                if (!sensorStr)
                {
                    NVGSTDS_WARN_MSG_V("wrong sensor.id value");
                    goto error;
                }

                g_strstrip(sensorStr);
                if (!g_strcmp0(sensorStr, ""))
                {
                    NVGSTDS_WARN_MSG_V("empty sensor.id value");
                    goto error;
                }
            }
            else
            {
                NVGSTDS_WARN_MSG_V(
                    "wrong message format, missing 'sensor.id' field.");
                goto error;
            }
        }
        else
        {
            NVGSTDS_WARN_MSG_V(
                "wrong message format, missing 'sensor.id' field.");
            goto error;
        }
    }

    g_object_unref(parser);

error:
    g_object_unref(parser);
    g_free(sensorStr);
}

/**
 * @brief 生成符合RFC3339标准的时间戳。
 *
 * @param buf 用于存储生成的时间戳的缓冲区。
 * @param buf_size 缓冲区的大小。
 */
static void generate_ts_rfc3339(char *buf, int buf_size)
{
    time_t          tloc;
    struct tm       tm_log;
    struct timespec ts;
    char            strmsec[6]; // .nnnZ\0

    clock_gettime(CLOCK_REALTIME, &ts);
    tloc = ts.tv_sec;            // 直接赋值即可，无需 memcpy
    localtime_r(&tloc, &tm_log); // 使用本地时区
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S%z",
             &tm_log); // %z 输出时区偏移（如 +0800）
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3d", ms);
    strncat(buf, strmsec, buf_size);
}

/**
 * @brief 生成事件消息元数据
 *
 * @param appCtx 应用程序上下文
 * @param data 传递的NvDsEventMsgMeta指针,输出
 * @param class_id 类别ID
 * @param useTs 是否使用时间戳
 * @param ts 时间戳
 * @param src_uri 源URI
 * @param stream_id 流ID
 * @param sensor_id 传感器ID
 * @param obj_params 对象元数据参数
 * @param scaleW 宽度缩放比例
 * @param scaleH 高度缩放比例
 * @param frame_meta 帧元数据
 */
static void generate_event_msg_meta(AppCtx *appCtx, gpointer data,
                                    gint class_id, gboolean useTs,
                                    GstClockTime ts, gchar *src_uri,
                                    gint stream_id, guint sensor_id,
                                    NvDsObjectMeta *obj_params, float scaleW,
                                    float scaleH, NvDsFrameMeta *frame_meta)
{
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
    GstClockTime      ts_generated = 0;

    meta->objType = NVDS_OBJECT_TYPE_UNKNOWN; /**< object unknown */
    /* The sensor_id is parsed from the source group name which has the format
     * [source<sensor-id>]. */
    meta->sensorId = sensor_id;
    meta->placeId = sensor_id;
    meta->moduleId = sensor_id;
    meta->frameId = frame_meta->frame_num;
    meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);

    meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);

    // strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);
    meta->videoPath = (gchar *)g_malloc0(256);
    strncpy(meta->videoPath, appCtx->config.multi_source_config[stream_id].uri,
            256);

    for (NvDsClassifierMetaList *cl = obj_params->classifier_meta_list; cl;
         cl = cl->next)
    {
        NvDsClassifierMeta *cl_meta = (NvDsClassifierMeta *)cl->data;
        for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll;
             ll = ll->next)
        {
            NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
            if (ll_meta->result_label[0] != '\0')
            {
                if (ll_meta->result_prob > 0.5)
                {
                    strncpy(meta->objectId, ll_meta->result_label,
                            MAX_LABEL_SIZE);
                    meta->confidence = ll_meta->result_prob;
                    break;
                }
            }
        }
    }

    /** INFO: This API is called once for every 30 frames (now) */
    // 生成符合RFC3339标准的时间戳。
    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);

    /**
     * Valid attributes in the metadata sent over nvmsgbroker:
     * a) Sensor ID (shall be configured in nvmsgconv config file)
     * b) bbox info (meta->bbox) <- obj_params->rect_params (attr_info have sgie
     * info) c) tracking ID (meta->trackingId) <- obj_params->object_id 通过
     * nvmsgbroker 发送的元数据中的有效属性： a) 传感器 ID（应在 nvmsgconv
     * 配置文件中配置） b) bbox 信息（meta->bbox）<-
     * obj_params->rect_params（attr_info 有 sgie 信息） c) 跟踪
     * ID（meta->trackingId）<- obj_params->object_id
     */

    /** bbox - 分辨率由 nvinfer 缩放到了 streammux 提供的分辨率
     * 因此必须将其缩放回原始流分辨率
     */

    meta->bbox.left = obj_params->rect_params.left * scaleW;
    meta->bbox.top = obj_params->rect_params.top * scaleH;
    meta->bbox.width = obj_params->rect_params.width * scaleW;
    meta->bbox.height = obj_params->rect_params.height * scaleH;

    /** tracking ID */
    meta->trackingId = obj_params->object_id;

    /** 使用 nvmultiurisrcbin REST API 添加流时的传感器 ID */
    NvDsSensorInfo *sensorInfo = get_sensor_info(appCtx, stream_id);
    if (sensorInfo)
    {
        /** 此数据流是使用REST API添加的；我们有传感器信息！ */
        LOGD("this stream [%d:%s] was added using REST API; we have Sensor "
             "Info\n",
             sensorInfo->source_id, sensorInfo->sensor_id);
        meta->sensorStr = g_strdup(sensorInfo->sensor_id);
    }

    (void)ts_generated;

    meta->type = NVDS_EVENT_MOVING;
    meta->objType = NVDS_OBJECT_TYPE_UNKNOWN;
    meta->objClassId = class_id;
}

/**
 * 所有推理（主要+次要）完成后调用的回调函数。
 * 在这里可以修改元数据。
 */
static void bbox_generated_probe_after_analytics(AppCtx *appCtx, GstBuffer *buf,
                                                 NvDsBatchMeta *batch_meta,
                                                 guint          index)
{
    NvDsObjectMeta   *obj_meta = NULL;
    GstClockTime      buffer_pts = 0;
    guint32           stream_id = 0;
    NvDsSRSessionId   sessId = 0;
    NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
    NvDsSrcBin       *src_bin = &bin->sub_bins[index];
    guint             startTime = 7;
    guint             duration = 8;

    NvBufSurface *ip_surf = NULL;
    if (appCtx->config.enable_jpeg_save) {
        GstMapInfo inmap = GST_MAP_INFO_INIT;
        if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
            GST_ERROR("input buffer mapinfo failed");
            return;
        }
        ip_surf = (NvBufSurface *)inmap.data;
        gst_buffer_unmap(buf, &inmap);
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        stream_id = frame_meta->source_id;

        /* if (appCtx->custom_msg_data)
        {
            guint64 custom_timest = appCtx->custom_msg_data->timestamp;
            // rtsp时间戳获取
            guint64 ntp = frame_meta->ntp_timestamp;
            // 转成可读的时间格式
            char time_str[64];
            if (custom_timest > 0 && ntp > 0)
            {
                // TODO:时间戳更具需要修改
                gdouble ntp_unix = ntp_to_unix(ntp);
                time_t  t = (time_t)ntp_unix;
            }
        } */

        GList *l;
        for (l = frame_meta->obj_meta_list; l != NULL; l = l->next)
        {
            obj_meta = (NvDsObjectMeta *)(l->data);

            // HACK: 测试接收自定义的分类结果数据
            bool isTrueTarget = false;
            if (g_list_length(obj_meta->classifier_meta_list) > 0)
            {
                for (NvDsClassifierMetaList *cl =
                         obj_meta->classifier_meta_list;
                     cl; cl = cl->next)
                {
                    NvDsClassifierMeta *cl_meta =
                        (NvDsClassifierMeta *)cl->data;
                    for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll;
                         ll = ll->next)
                    {
                        NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
                        if (ll_meta->result_label[0] != '\0')
                        {
                            // FIXME: 不要写死
                            if (ll_meta->result_prob > 0.5)
                            {
                                isTrueTarget = true;
                                if (src_bin->config->smart_record == 3)
                                {
                                    // 启用智能视频记录
                                    // g_timeout_add(30000,
                                    // smart_record_event_generator, src_bin);
                                    smart_record_event_generator(src_bin);

                                    if (appCtx->config.enable_jpeg_save && ip_surf) {
                                        /* 5 秒节流：同一 source 在 5 秒内只执行一次保存 */
                                        static GstClockTime last_save_pts_per_src[MAX_SOURCE_BINS] = {0};
                                        guint sid = frame_meta->source_id;
                                        if (sid < MAX_SOURCE_BINS) {
                                            GstClockTime now_pts = frame_meta->buf_pts; /* 纳秒 */
                                            if (last_save_pts_per_src[sid] == 0 ||
                                                now_pts - last_save_pts_per_src[sid] >= 5 * GST_SECOND) {
                                                last_save_pts_per_src[sid] = now_pts;
                                                NvDsObjEncUsrArgs frameData = {0};
                                                frameData.isFrame = 1;
                                                frameData.saveImg = 1;
                                                frameData.attachUsrMeta = FALSE;
                                                frameData.scaleImg = FALSE;
                                                frameData.scaledWidth = 0;
                                                frameData.scaledHeight = 0;
                                                frameData.quality = 100;
                                                frameData.objNum = (int)obj_meta->object_id; /* 用跟踪ID/对象ID 作为 objNum */
                                                frameData.calcEncodeTime = 0;
                                                /* 自定义文件名: <sanitized-uri>_<pts_ms>_f<frame>_o<obj>.jpg */
                                                const char *raw_uri = appCtx->config.multi_source_config[frame_meta->source_id].uri ? appCtx->config.multi_source_config[frame_meta->source_id].uri : "unknown";
                                                char uri_sanitized[256];
                                                size_t rlen = strlen(raw_uri);
                                                if (rlen >= sizeof(uri_sanitized)) rlen = sizeof(uri_sanitized)-1;
                                                for (size_t si=0; si<rlen; ++si) {
                                                    char c = raw_uri[si];
                                                    if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')) uri_sanitized[si]=c; else uri_sanitized[si]='_';
                                                }
                                                uri_sanitized[rlen]='\0';
                                                unsigned long long pts_ms = (unsigned long long)(frame_meta->buf_pts/1000000ULL);
                                                snprintf(frameData.fileNameImg, sizeof(frameData.fileNameImg),
                                                         "%s_%llu_f%u_o%d.jpg", uri_sanitized, pts_ms, frame_meta->frame_num, frameData.objNum);
                                                nvds_obj_enc_process((gpointer)appCtx->obj_ctx_handle,
                                                                    &frameData, ip_surf, obj_meta, frame_meta);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /* NOTE: 分类显示逻辑已移至 overlay_graphics
             * 中统一处理，这里不再拼接文本。 */

            /* 分类结果历史加权平滑：仅在 tracker 与 secondary-gie 开启时启用 */
            gboolean tracker_on = appCtx->config.tracker_config.enable;
            gboolean sgie_on = FALSE;
            for (guint si = 0; si < appCtx->config.num_secondary_gie_sub_bins;
                 ++si)
            {
                if (appCtx->config.secondary_gie_sub_bin_config[si].enable)
                {
                    sgie_on = TRUE;
                    break;
                }
            }

            // FIXME:
            sgie_on = FALSE;
            if (tracker_on && sgie_on &&
                obj_meta->object_id != UNTRACKED_OBJECT_ID)
            {
                /* 定义对象聚合结构 */
                typedef struct _ObjClsAgg
                {
                    GHashTable *label_scores; /* key: gchar*, value: gdouble* */
                    guint       last_seen_frame;
                } ObjClsAgg;

                /* 懒初始化全局缓存表 */
                if (!appCtx->cls_agg_map)
                {
                    appCtx->cls_agg_map = g_hash_table_new_full(
                        g_int64_hash, g_int64_equal, g_free, (GDestroyNotify)NULL /* value freed below when destroying map at exit */);
                }

                guint64    oid = obj_meta->object_id;
                ObjClsAgg *agg =
                    (ObjClsAgg *)g_hash_table_lookup(appCtx->cls_agg_map, &oid);
                if (!agg)
                {
                    guint64 *key = (guint64 *)g_malloc(sizeof(guint64));
                    *key = oid;
                    agg = g_new0(ObjClsAgg, 1);
                    agg->label_scores = g_hash_table_new_full(
                        g_str_hash, g_str_equal, g_free, g_free);
                    agg->last_seen_frame = frame_meta->frame_num;
                    g_hash_table_insert(appCtx->cls_agg_map, key, agg);
                }

                /* 计算归一化目标面积作为权重之一，并进行可调平衡 */
                gdouble area = (gdouble)(obj_meta->rect_params.width *
                                         obj_meta->rect_params.height);
                gdouble pW =
                    appCtx->config.streammux_config.pipeline_width
                        ? appCtx->config.streammux_config.pipeline_width
                        : frame_meta->pipeline_width;
                gdouble pH =
                    appCtx->config.streammux_config.pipeline_height
                        ? appCtx->config.streammux_config.pipeline_height
                        : frame_meta->pipeline_height;
                gdouble norm_area =
                    (pW > 0 && pH > 0) ? (area / (pW * pH)) : 1.0;
                if (norm_area < 0.0)
                    norm_area = 0.0;
                /* clip 到 [eps, 1]，避免极端缩小目标权重为0 */
                const gdouble eps = 1e-6;
                if (norm_area < eps)
                    norm_area = eps;

                /* 平衡参数：alpha 控制对分类置信度的偏好，beta 控制对面积的偏好
                    ref_area 用来拉升小目标的面积贡献（例如 0.02 表示 2% 画面）
                 */
                const gdouble alpha = 0.8;     /* 推荐范围 [0.6, 0.95] */
                const gdouble beta = 0.2;      /* 推荐范围 [0.05, 0.4] */
                const gdouble ref_area = 0.02; /* 推荐范围 [0.01, 0.05] */

                /* 面积项映射：使用归一化后除以参考面积，限制上限，避免过大 */
                gdouble area_term =
                    norm_area / ref_area; /* 小目标 <1，大目标 >1 */
                if (area_term > 1.0)
                    area_term = 1.0;

                /* 累加每个分类标签的加权得分：score += prob * norm_area */
                for (NvDsClassifierMetaList *cl =
                         obj_meta->classifier_meta_list;
                     cl; cl = cl->next)
                {
                    NvDsClassifierMeta *cl_meta =
                        (NvDsClassifierMeta *)cl->data;
                    for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll;
                         ll = ll->next)
                    {
                        NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
                        if (ll_meta->result_label[0] == '\0')
                            continue;
                        if (ll_meta->result_prob <= 0.0)
                            continue;

                        gchar   *label_key = g_strdup(ll_meta->result_label);
                        gdouble *sum_ptr = (gdouble *)g_hash_table_lookup(
                            agg->label_scores, label_key);
                        if (!sum_ptr)
                        {
                            sum_ptr = g_new0(gdouble, 1);
                            g_hash_table_insert(agg->label_scores, label_key,
                                                sum_ptr);
                        }
                        else
                        {
                            g_free(label_key); /* key already exists */
                        }
                        /* 可调融合：置信度^alpha 与 面积项^beta 的加权几何式 */
                        gdouble prob_term = ll_meta->result_prob;
                        if (prob_term < eps)
                            prob_term = eps;
                        gdouble fused =
                            pow(prob_term, alpha) * pow(area_term, beta);
                        *sum_ptr += fused;
                    }
                }
                agg->last_seen_frame = frame_meta->frame_num;

                /* 选择累计得分最高的标签，并用归一化后概率替换当前结果 */
                GHashTableIter it;
                gpointer       k, v;
                gdouble        best_score = -1.0, total_score = 0.0;
                gchar         *best_label = NULL;
                g_hash_table_iter_init(&it, agg->label_scores);
                while (g_hash_table_iter_next(&it, &k, &v))
                {
                    gdouble s = *((gdouble *)v);
                    total_score += s;
                    if (s > best_score)
                    {
                        best_score = s;
                        best_label = (gchar *)k;
                    }
                }
                if (best_label && total_score > 0.0)
                {
                    gdouble best_prob = best_score / total_score; /* 归一化 */
                    /* 将该对象的分类列表中的概率重写：仅保留最佳标签为
                     * best_prob，其它置0 */
                    for (NvDsClassifierMetaList *cl =
                             obj_meta->classifier_meta_list;
                         cl; cl = cl->next)
                    {
                        NvDsClassifierMeta *cl_meta =
                            (NvDsClassifierMeta *)cl->data;
                        for (NvDsLabelInfoList *ll = cl_meta->label_info_list;
                             ll; ll = ll->next)
                        {
                            NvDsLabelInfo *ll_meta = (NvDsLabelInfo *)ll->data;
                            if (ll_meta->result_label[0] == '\0')
                                continue;
                            if (g_strcmp0(ll_meta->result_label, best_label) ==
                                0)
                            {
                                ll_meta->result_prob = best_prob;
                            }
                            else
                            {
                                ll_meta->result_prob = 0.0;
                            }
                        }
                    }
                }
            }

            /**
             * 仅在此回调在 tiler 之后启用
             * 注意：缩放回代码注释
             * 现在 bbox_generated_probe_after_analytics() 是在分析之后
             * （例如 pgie、tracker 或 sgie）
             * 并且在 tiler 之前，没有插件会缩放元数据，并且将
             * 对应于 nvstreammux 分辨率
             */
            float scaleW = 0;
            float scaleH = 0;
            /* 消息发送频率
             * 此处消息每30帧发送一次给第一个对象。
             */
            buffer_pts = frame_meta->buf_pts;
            if (!appCtx->config.streammux_config.pipeline_width ||
                !appCtx->config.streammux_config.pipeline_height)
            {
                g_print("invalid pipeline params\n");
                return;
            }
            LOGD("stream %d==source_frame_width:%d [%d X %d]\n",
                 frame_meta->source_id, frame_meta->pad_index,
                 frame_meta->source_frame_width,
                 frame_meta->source_frame_height);
            scaleW = (float)frame_meta->source_frame_width /
                     appCtx->config.streammux_config.pipeline_width;
            scaleH = (float)frame_meta->source_frame_height /
                     appCtx->config.streammux_config.pipeline_height;

            /** 为每个检测对象生成 NvDsEventMsgMeta */
            NvDsEventMsgMeta *msg_meta =
                (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
            generate_event_msg_meta(
                appCtx, msg_meta, obj_meta->class_id, TRUE,
                /**< useTs NOTE: Pass FALSE for files without base-timestamp in
                   URI */
                buffer_pts, appCtx->config.multi_source_config[stream_id].uri,
                stream_id,
                appCtx->config.multi_source_config[stream_id].camera_id,
                obj_meta, scaleW, scaleH, frame_meta);
            NvDsUserMeta *user_event_meta =
                nvds_acquire_user_meta_from_pool(batch_meta);
            if (user_event_meta)
            {
                /*
                 * Since generated event metadata has custom objects for
                 * Vehicle / Person which are allocated dynamically, we are
                 * setting copy and free function to handle those fields when
                 * metadata copy happens between two components.
                 * 由于生成的事件元数据具有动态分配的车辆/人员自定义对象，我们
                 * 设置了复制和释放函数来处理这些字段，以便在两个组件之间进行元数据复制时使用
                 */
                user_event_meta->user_meta_data = (void *)msg_meta;
                user_event_meta->base_meta.batch_meta = batch_meta;
                user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                user_event_meta->base_meta.copy_func =
                    (NvDsMetaCopyFunc)meta_copy_func;
                user_event_meta->base_meta.release_func =
                    (NvDsMetaReleaseFunc)meta_free_func;
                nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
            }
            else
            {
                g_print("Error in attaching event meta to buffer\n");
            }
        }
    }

    if (appCtx->config.enable_jpeg_save) {
        nvds_obj_enc_finish((gpointer)appCtx->obj_ctx_handle);
    }

    /* NvDsMetaList *l_user_meta = NULL;
    NvDsUserMeta *user_meta = NULL;
    for (l_user_meta = batch_meta->batch_user_meta_list; l_user_meta != NULL;
         l_user_meta = l_user_meta->next)
    {
        user_meta = (NvDsUserMeta *)(l_user_meta->data);
        // TODO:
        if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META)
        {
            g_print("Preprocess batch meta found\n");
        }
    } */
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
static void all_bbox_generated(AppCtx *appCtx, GstBuffer *buf,
                               NvDsBatchMeta *batch_meta, guint index)
{
    // guint num_male = 0;
    // guint num_female = 0;
    // guint num_objects[128];
    // guint num_rects = 0; // 矩形数量

    // memset(num_objects, 0, sizeof(num_objects));
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void _intr_handler(int signum)
{
    struct sigaction action;

    NVGSTDS_ERR_MSG_V("User Interrupted.. \n");

    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;

    sigaction(SIGINT, &action, NULL);

    cintr = TRUE;
}

/**
 * callback function to print the performance numbers of each stream.
 */
static void perf_cb(gpointer context, NvDsAppPerfStruct *str)
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

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean check_for_interrupt(gpointer data)
{
    if (quit)
    {
        return FALSE;
    }

    if (cintr)
    {
        cintr = FALSE;

        quit = TRUE;
        g_main_loop_quit(main_loop);

        return FALSE;
    }
    return TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void _intr_setup(void)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = _intr_handler;

    sigaction(SIGINT, &action, NULL);
}

static gboolean kbhit(void)
{
    struct timeval tv;
    fd_set         rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO(&rdfs);
    FD_SET(STDIN_FILENO, &rdfs);

    select(STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
static void changemode(int dir)
{
    static struct termios oldt, newt;

    if (dir == 1)
    {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    }
    else
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

static void print_runtime_commands(void)
{
    g_print("\nRuntime commands:\n"
            "\th: Print this help\n"
            "\tq: Quit\n\n"
            "\tp: Pause\n"
            "\tr: Resume\n\n");

    if (appCtx[0]->config.tiled_display_config.enable)
    {
        g_print("NOTE: To expand a source in the 2D tiled display and view "
                "object details,"
                " left-click on the source.\n"
                "      To go back to the tiled display, right-click anywhere "
                "on the window.\n\n");
    }
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
static gboolean event_thread_func(gpointer arg)
{
    guint    i;
    gboolean ret = TRUE;

    // Check if all instances have quit
    for (i = 0; i < num_instances; i++)
    {
        if (!appCtx[i]->quit)
            break;
    }

    if (i == num_instances)
    {
        quit = TRUE;
        g_main_loop_quit(main_loop);
        return FALSE;
    }
    // Check for keyboard input
    if (!kbhit())
    {
        // continue;
        return TRUE;
    }
    int c = fgetc(stdin);
    g_print("\n");

    gint        source_id;
    GstElement *tiler = appCtx[rcfg]->pipeline.tiled_display_bin.tiler;
    if (appCtx[rcfg]->config.tiled_display_config.enable)
    {
        g_object_get(G_OBJECT(tiler), "show-source", &source_id, NULL);

        if (selecting)
        {
            if (rrowsel == FALSE)
            {
                if (c >= '0' && c <= '9')
                {
                    rrow = c - '0';
                    if (rrow < appCtx[rcfg]->config.tiled_display_config.rows)
                    {
                        g_print("--selecting source  row %d--\n", rrow);
                        rrowsel = TRUE;
                    }
                    else
                    {
                        g_print(
                            "--selected source  row %d out of bound, reenter\n",
                            rrow);
                    }
                }
            }
            else
            {
                if (c >= '0' && c <= '9')
                {
                    unsigned int tile_num_columns =
                        appCtx[rcfg]->config.tiled_display_config.columns;
                    rcol = c - '0';
                    if (rcol < tile_num_columns)
                    {
                        selecting = FALSE;
                        rrowsel = FALSE;
                        source_id = tile_num_columns * rrow + rcol;
                        g_print("--selecting source  col %d sou=%d--\n", rcol,
                                source_id);
                        if (source_id >=
                            (gint)appCtx[rcfg]->config.num_source_sub_bins)
                        {
                            source_id = -1;
                        }
                        else
                        {
                            appCtx[rcfg]->show_bbox_text = TRUE;
                            appCtx[rcfg]->active_source_index = source_id;
                            g_object_set(G_OBJECT(tiler), "show-source",
                                         source_id, NULL);
                        }
                    }
                    else
                    {
                        g_print(
                            "--selected source  col %d out of bound, reenter\n",
                            rcol);
                    }
                }
            }
        }
    }
    switch (c)
    {
    case 'h':
        print_runtime_commands();
        break;
    case 'p':
        for (i = 0; i < num_instances; i++)
            pause_pipeline(appCtx[i]);
        break;
    case 'r':
        for (i = 0; i < num_instances; i++)
            resume_pipeline(appCtx[i]);
        break;
    case 'q':
        quit = TRUE;
        g_main_loop_quit(main_loop);
        ret = FALSE;
        break;
    case 'c':
        if (appCtx[rcfg]->config.tiled_display_config.enable &&
            selecting == FALSE && source_id == -1)
        {
            g_print("--selecting config file --\n");
            c = fgetc(stdin);
            if (c >= '0' && c <= '9')
            {
                rcfg = c - '0';
                if (rcfg < num_instances)
                {
                    g_print("--selecting config  %d--\n", rcfg);
                }
                else
                {
                    g_print("--selected config file %d out of bound, reenter\n",
                            rcfg);
                    rcfg = 0;
                }
            }
        }
        break;
    case 'z':
        if (appCtx[rcfg]->config.tiled_display_config.enable &&
            source_id == -1 && selecting == FALSE)
        {
            g_print("--selecting source --\n");
            selecting = TRUE;
        }
        else
        {
            if (!show_bbox_text)
                appCtx[rcfg]->show_bbox_text = FALSE;
            g_object_set(G_OBJECT(tiler), "show-source", -1, NULL);
            appCtx[rcfg]->active_source_index = -1;
            selecting = FALSE;
            rcfg = 0;
            g_print("--tiled mode --\n");
        }
        break;
    default:
        break;
    }
    return ret;
}

static int get_source_id_from_coordinates(float x_rel, float y_rel,
                                          AppCtx *appCtx)
{
    int tile_num_rows = appCtx->config.tiled_display_config.rows;
    int tile_num_columns = appCtx->config.tiled_display_config.columns;

    int source_id = (int)(x_rel * tile_num_columns);
    source_id += ((int)(y_rel * tile_num_rows)) * tile_num_columns;

    /* Don't allow clicks on empty tiles. */
    if (source_id >= (gint)appCtx->config.num_source_sub_bins)
        source_id = -1;

    return source_id;
}

/**
 * Thread to monitor X window events.
 */
static gpointer nvds_x_event_thread(gpointer data)
{
    g_mutex_lock(&disp_lock);
    while (display)
    {
        XEvent e;
        guint  index;
        memset(&e, 0, sizeof(XEvent));
        while (XPending(display))
        {
            XNextEvent(display, &e);
            switch (e.type)
            {
            case ButtonPress:
            {
                XWindowAttributes win_attr;
                XButtonEvent      ev = e.xbutton;
                gint              source_id;
                GstElement       *tiler;
                memset(&win_attr, 0, sizeof(XWindowAttributes));

                XGetWindowAttributes(display, ev.window, &win_attr);

                for (index = 0; index < MAX_INSTANCES; index++)
                    if (ev.window == windows[index])
                        break;

                tiler = appCtx[index]->pipeline.tiled_display_bin.tiler;
                g_object_get(G_OBJECT(tiler), "show-source", &source_id, NULL);

                if (ev.button == Button1 && source_id == -1 &&
                    (index >= 0 && index < MAX_INSTANCES))
                {
                    source_id = get_source_id_from_coordinates(
                        ev.x * 1.0 / win_attr.width,
                        ev.y * 1.0 / win_attr.height, appCtx[index]);
                    if (source_id > -1)
                    {
                        g_object_set(G_OBJECT(tiler), "show-source", source_id,
                                     NULL);
                        appCtx[index]->active_source_index = source_id;
                        appCtx[index]->show_bbox_text = TRUE;
                    }
                }
                else if (ev.button == Button3)
                {
                    g_object_set(G_OBJECT(tiler), "show-source", -1, NULL);
                    appCtx[index]->active_source_index = -1;
                    if (!show_bbox_text)
                        appCtx[index]->show_bbox_text = FALSE;
                }
            }
            break;
            case KeyRelease:
            case KeyPress:
            {
                KeySym p, r, q;
                guint  i;
                p = XKeysymToKeycode(display, XK_P);
                r = XKeysymToKeycode(display, XK_R);
                q = XKeysymToKeycode(display, XK_Q);
                if (e.xkey.keycode == p)
                {
                    for (i = 0; i < num_instances; i++)
                        pause_pipeline(appCtx[i]);
                    break;
                }
                if (e.xkey.keycode == r)
                {
                    for (i = 0; i < num_instances; i++)
                        resume_pipeline(appCtx[i]);
                    break;
                }
                if (e.xkey.keycode == q)
                {
                    quit = TRUE;
                    g_main_loop_quit(main_loop);
                }
            }
            break;
            case ClientMessage:
            {
                Atom wm_delete;
                for (index = 0; index < MAX_INSTANCES; index++)
                    if (e.xclient.window == windows[index])
                        break;

                wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", 1);
                if (wm_delete != None && wm_delete == (Atom)e.xclient.data.l[0])
                {
                    quit = TRUE;
                    g_main_loop_quit(main_loop);
                }
            }
            break;
            }
        }
        g_mutex_unlock(&disp_lock);
        g_usleep(G_USEC_PER_SEC / 20);
        g_mutex_lock(&disp_lock);
    }
    g_mutex_unlock(&disp_lock);
    return NULL;
}

// TODO:暂时没用
static void msg_broker_subscribe_cb(NvMsgBrokerErrorType status, void *msg,
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
        NvDsEventMsgMeta *event_msg_meta = (NvDsEventMsgMeta *)msg;
        AppCtx           *appCtx = (AppCtx *)user_ptr;

        parse_cloud_message(msg, msglen);
        status = NV_MSGBROKER_API_OK;
    }
    else
    {
        status = NV_MSGBROKER_API_ERR;
    }
}

/**
 * callback function to add application specific metadata.
 * Here it demonstrates how to display the URI of source in addition to
 * the text generated after inference.
 * 该回调函数用于添加应用程序特定的元数据。
 * 这里演示了如何显示源的URI以及推理后生成的文本。
 */
static gboolean overlay_graphics(AppCtx *appCtx, GstBuffer *buf,
                                 NvDsBatchMeta *batch_meta, guint index)
{
    int srcIndex = appCtx->active_source_index;
    // if (srcIndex == -1)
    //     return TRUE;

    /* 为每个对象生成完整的检测+分类标签(概率)文本，覆盖原来的
     * bbox_generated_probe_after_analytics 中逻辑 */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            /* 构建完整标签：检测类别(置信度) + 分类结果(概率) */
            GString *gstr = g_string_new(NULL);

            /* 首先添加检测结果 */
            if (obj_meta->obj_label[0] != '\0')
            {
                g_string_append_printf(gstr, "%s(%.2f)", obj_meta->obj_label,
                                       obj_meta->confidence);
            }
            else
            {
                /* 如果没有标签，使用类别ID */
                g_string_append_printf(gstr, "Class_%d(%.2f)",
                                       obj_meta->class_id,
                                       obj_meta->confidence);
            }

            /* 如果有分类结果，追加分类信息 */
            if (obj_meta->classifier_meta_list)
            {
                g_string_append_printf(gstr, " ");
                for (NvDsClassifierMetaList *cl =
                         obj_meta->classifier_meta_list;
                     cl; cl = cl->next)
                {
                    NvDsClassifierMeta *cl_meta =
                        (NvDsClassifierMeta *)cl->data;
                    for (NvDsLabelInfoList *ll = cl_meta->label_info_list; ll;
                         ll = ll->next)
                    {
                        NvDsLabelInfo *li = (NvDsLabelInfo *)ll->data;
                        if (li->result_label[0] == '\0')
                            continue;
                        g_string_append_printf(gstr, "%s(%.2f) ",
                                               li->result_label,
                                               li->result_prob);
                    }
                }
            }

            /* 设置显示文本 - 所有对象都会有标签 */
            if (gstr->len > 0)
            {
                /* 改进：
                 * 1) 根据目标尺寸自适应缩小字体，避免小框被整块文字遮住。
                 * 2) 小目标时尝试把标签放在框外（先上方, 若不足则下方）。
                 * 3) 背景半透明以减少遮挡感。 */
                if (obj_meta->text_params.display_text)
                    g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = g_strdup(gstr->str);

                int frame_w = frame_meta->source_frame_width > 0
                                  ? frame_meta->source_frame_width
                                  : 1920;
                int frame_h = frame_meta->source_frame_height > 0
                                  ? frame_meta->source_frame_height
                                  : 1080;

                float bw = obj_meta->rect_params.width;
                float bh = obj_meta->rect_params.height;
                float min_side = bw < bh ? bw : bh;
                int   base_font = appCtx->config.osd_config.text_size > 0
                                      ? appCtx->config.osd_config.text_size
                                      : 12;
                float scale = 1.0f;
                if (min_side < 40)
                    scale = 0.6f;
                else if (min_side < 80)
                    scale = 0.8f; /* 可根据需要再细化 */
                int font_size = (int)(base_font * scale);
                if (font_size < 8)
                    font_size = 8; /* 最小字号 */

                obj_meta->text_params.font_params.font_color =
                    (NvOSD_ColorParams){1.0f, 1.0f, 1.0f, 1.0f};
                obj_meta->text_params.font_params.font_size = font_size;
                obj_meta->text_params.font_params.font_name = "Serif";
                obj_meta->text_params.set_bg_clr = 1;
                /* 透明度稍低，减轻遮挡 (A=0.4) */
                obj_meta->text_params.text_bg_clr =
                    (NvOSD_ColorParams){0.f, 0.f, 0.f, 0.4f};

                /* 估算文本高度：字号 + 顶/底边距(简单) */
                int text_h = font_size + 4;
                int x = (int)obj_meta->rect_params.left;
                if (x < 0)
                    x = 0;
                if (x > frame_w - 4)
                    x = frame_w - 4;

                /* 判断小目标比例阈值（例如 <1% 认为小） */
                float area_ratio =
                    (bw * bh) / ((float)frame_w * (float)frame_h + 1e-3f);
                int y;
                if (area_ratio < 0.01f)
                {
                    /* 优先放在框外上方 */
                    int y_above = (int)obj_meta->rect_params.top - text_h - 2;
                    if (y_above >= 0)
                    {
                        y = y_above;
                    }
                    else
                    {
                        /* 上面放不下，放框下方 */
                        y = (int)(obj_meta->rect_params.top +
                                  obj_meta->rect_params.height + 2);
                        if (y > frame_h - text_h)
                            y = frame_h - text_h;
                    }
                }
                else
                {
                    /* 大一点的框，仍尝试放到上方内部或外侧 */
                    int y_try = (int)obj_meta->rect_params.top - text_h - 2;
                    if (y_try < 0)
                    {
                        y = (int)obj_meta->rect_params.top +
                            2; /* 放到框内靠上 */
                        if (y + text_h > frame_h)
                            y = frame_h - text_h;
                    }
                    else
                    {
                        y = y_try;
                    }
                }
                obj_meta->text_params.x_offset = x;
                obj_meta->text_params.y_offset = y;

                /* 如果目标极窄而文本会超出右边界，可左移 */
                int estimated_text_w =
                    (int)(strlen(obj_meta->text_params.display_text) *
                          font_size * 0.55f);
                if (estimated_text_w > 0 && x + estimated_text_w > frame_w)
                {
                    int new_x = frame_w - estimated_text_w - 2;
                    if (new_x < 0)
                        new_x = 0;
                    obj_meta->text_params.x_offset = new_x;
                }
            }
            g_string_free(gstr, TRUE);
        }
    }

    if (srcIndex == -1)
        return TRUE;

    NvDsFrameLatencyInfo *latency_info = NULL;
    NvDsDisplayMeta      *display_meta =
        nvds_acquire_display_meta_from_pool(batch_meta);

    display_meta->num_labels = 1;
    display_meta->text_params[0].display_text = g_strdup_printf(
        "Source: %s", appCtx->config.multi_source_config[srcIndex].uri);

    display_meta->text_params[0].y_offset = 20;
    display_meta->text_params[0].x_offset = 20;
    display_meta->text_params[0].font_params.font_color =
        (NvOSD_ColorParams){0, 1, 0, 1};
    display_meta->text_params[0].font_params.font_size =
        appCtx->config.osd_config.text_size * 1.5;
    display_meta->text_params[0].font_params.font_name = "Serif";
    display_meta->text_params[0].set_bg_clr = 1;
    display_meta->text_params[0].text_bg_clr =
        (NvOSD_ColorParams){0, 0, 0, 1.0};

    if (nvds_enable_latency_measurement)
    {
        g_mutex_lock(&appCtx->latency_lock);
        latency_info = &appCtx->latency_info[index];
        display_meta->num_labels++;
        display_meta->text_params[1].display_text =
            g_strdup_printf("Latency: %lf", latency_info->latency);
        g_mutex_unlock(&appCtx->latency_lock);

        display_meta->text_params[1].y_offset =
            (display_meta->text_params[0].y_offset * 2) +
            display_meta->text_params[0].font_params.font_size;
        display_meta->text_params[1].x_offset = 20;
        display_meta->text_params[1].font_params.font_color =
            (NvOSD_ColorParams){0, 1, 0, 1};
        display_meta->text_params[1].font_params.font_size =
            appCtx->config.osd_config.text_size * 1.5;
        display_meta->text_params[1].font_params.font_name = "Arial";
        display_meta->text_params[1].set_bg_clr = 1;
        display_meta->text_params[1].text_bg_clr =
            (NvOSD_ColorParams){0, 0, 0, 1.0};
    }

    nvds_add_display_meta_to_frame(
        nvds_get_nth_frame_meta(batch_meta->frame_meta_list, 0), display_meta);
    return TRUE;
}

static gboolean recreate_pipeline_thread_func(gpointer arg)
{
    guint    i;
    gboolean ret = TRUE;
    AppCtx  *appCtx = (AppCtx *)arg;

    g_print("Destroy pipeline\n");
    destroy_pipeline(appCtx);

    g_print("Recreate pipeline\n");
    if (!create_pipeline(appCtx, bbox_generated_probe_after_analytics,
                         all_bbox_generated, perf_cb, overlay_graphics,
                         NULL))
    {
        NVGSTDS_ERR_MSG_V("Failed to create pipeline");
        return_value = -1;
        return FALSE;
    }

    if (gst_element_set_state(appCtx->pipeline.pipeline, GST_STATE_PAUSED) ==
        GST_STATE_CHANGE_FAILURE)
    {
        NVGSTDS_ERR_MSG_V("Failed to set pipeline to PAUSED");
        return_value = -1;
        return FALSE;
    }

    for (i = 0; i < appCtx->config.num_sink_sub_bins; i++)
    {
        if (!GST_IS_VIDEO_OVERLAY(
                appCtx->pipeline.instance_bins[0].sink_bin.sub_bins[i].sink))
        {
            continue;
        }

        gst_video_overlay_set_window_handle(
            GST_VIDEO_OVERLAY(
                appCtx->pipeline.instance_bins[0].sink_bin.sub_bins[i].sink),
            (gulong)windows[appCtx->index]);
        gst_video_overlay_expose(GST_VIDEO_OVERLAY(
            appCtx->pipeline.instance_bins[0].sink_bin.sub_bins[i].sink));
    }

    if (gst_element_set_state(appCtx->pipeline.pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE)
    {

        g_print("\ncan't set pipeline to playing state.\n");
        return_value = -1;
        return FALSE;
    }

    return ret;
}

int main(int argc, char *argv[])
{
    GOptionContext *ctx = NULL;
    GOptionGroup   *group = NULL;
    GError         *error = NULL;
    guint           i;

    ctx = g_option_context_new("Nvidia DeepStream Demo");
    group = g_option_group_new("abc", NULL, NULL, NULL, NULL);
    g_option_group_add_entries(group, entries);

    g_option_context_set_main_group(ctx, group);
    g_option_context_add_group(ctx, gst_init_get_option_group());

    GST_DEBUG_CATEGORY_INIT(NVDS_APP, "NVDS_APP", 0, NULL);

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    if (!g_option_context_parse(ctx, &argc, &argv, &error))
    {
        NVGSTDS_ERR_MSG_V("%s", error->message);
        return -1;
    }

    if (print_version)
    {
        g_print("deepstream-app version %d.%d.%d\n", NVDS_APP_VERSION_MAJOR,
                NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
        nvds_version_print();
        return 0;
    }

    if (print_dependencies_version)
    {
        g_print("deepstream-app version %d.%d.%d\n", NVDS_APP_VERSION_MAJOR,
                NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
        nvds_version_print();
        nvds_dependencies_version_print();
        return 0;
    }

    if (cfg_files)
    {
        num_instances = g_strv_length(cfg_files);
    }
    if (input_uris)
    {
        num_input_uris = g_strv_length(input_uris);
    }

    if (!cfg_files || num_instances == 0)
    {
        NVGSTDS_ERR_MSG_V("Specify config file with -c option");
        return_value = -1;
        goto done;
    }

    for (i = 0; i < num_instances; i++)
    {
        appCtx[i] = g_malloc0(sizeof(AppCtx));
        appCtx[i]->person_class_id = -1;
        appCtx[i]->car_class_id = -1;
        appCtx[i]->index = i;
        appCtx[i]->active_source_index = -1;
        if (show_bbox_text)
        {
            appCtx[i]->show_bbox_text = TRUE;
        }

        /* init classification aggregator map */
        appCtx[i]->cls_agg_map = NULL; /* lazy init in probe */

        if (input_uris && input_uris[i])
        {
            appCtx[i]->config.multi_source_config[0].uri =
                g_strdup_printf("%s", input_uris[i]);
            g_free(input_uris[i]);
        }

        if (g_str_has_suffix(cfg_files[i], ".yml") ||
            g_str_has_suffix(cfg_files[i], ".yaml"))
        {
            if (!parse_config_file_yaml(&appCtx[i]->config, cfg_files[i]))
            {
                NVGSTDS_ERR_MSG_V("Failed to parse config file '%s'",
                                  cfg_files[i]);
                appCtx[i]->return_value = -1;
                goto done;
            }
        }
        else if (g_str_has_suffix(cfg_files[i], ".txt"))
        {
            if (!parse_config_file(&appCtx[i]->config, cfg_files[i]))
            {
                NVGSTDS_ERR_MSG_V("Failed to parse config file '%s'",
                                  cfg_files[i]);
                appCtx[i]->return_value = -1;
                goto done;
            }
        }
    }

    for (i = 0; i < num_instances; i++)
    {
        if (!create_pipeline(appCtx[i], bbox_generated_probe_after_analytics,
                             all_bbox_generated, perf_cb, overlay_graphics,
                             NULL))
        {
            NVGSTDS_ERR_MSG_V("Failed to create pipeline");
            return_value = -1;
            goto done;
        }
    }

    main_loop = g_main_loop_new(NULL, FALSE);

    _intr_setup();
    g_timeout_add(400, check_for_interrupt, NULL);

    g_mutex_init(&disp_lock);
    display = XOpenDisplay(NULL);
    for (i = 0; i < num_instances; i++)
    {
        guint j;
#if defined(__aarch64__)
        if (gst_element_set_state(appCtx[i]->pipeline.pipeline,
                                  GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE)
        {
            NVGSTDS_ERR_MSG_V("Failed to set pipeline to PAUSED");
            return_value = -1;
            goto done;
        }
#endif
        for (j = 0; j < appCtx[i]->config.num_sink_sub_bins; j++)
        {
            XTextProperty xproperty;
            gchar        *title;
            guint         width, height;
            XSizeHints    hints = {0};

            if (!GST_IS_VIDEO_OVERLAY(appCtx[i]
                                          ->pipeline.instance_bins[0]
                                          .sink_bin.sub_bins[j]
                                          .sink))
            {
                continue;
            }

            if (!display)
            {
                NVGSTDS_ERR_MSG_V("Could not open X Display");
                return_value = -1;
                goto done;
            }

            if (appCtx[i]
                    ->config.sink_bin_sub_bin_config[j]
                    .render_config.width)
                width = appCtx[i]
                            ->config.sink_bin_sub_bin_config[j]
                            .render_config.width;
            else
                width = appCtx[i]->config.tiled_display_config.width;

            if (appCtx[i]
                    ->config.sink_bin_sub_bin_config[j]
                    .render_config.height)
                height = appCtx[i]
                             ->config.sink_bin_sub_bin_config[j]
                             .render_config.height;
            else
                height = appCtx[i]->config.tiled_display_config.height;

            width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
            height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;

            hints.flags = PPosition | PSize;
            hints.x = appCtx[i]
                          ->config.sink_bin_sub_bin_config[j]
                          .render_config.offset_x;
            hints.y = appCtx[i]
                          ->config.sink_bin_sub_bin_config[j]
                          .render_config.offset_y;
            hints.width = width;
            hints.height = height;

            windows[i] = XCreateSimpleWindow(
                display, RootWindow(display, DefaultScreen(display)), hints.x,
                hints.y, width, height, 2, 0x00000000, 0x00000000);

            XSetNormalHints(display, windows[i], &hints);

            if (num_instances > 1)
                title = g_strdup_printf(APP_TITLE "-%d", i);
            else
                title = g_strdup(APP_TITLE);
            if (XStringListToTextProperty((char **)&title, 1, &xproperty) != 0)
            {
                XSetWMName(display, windows[i], &xproperty);
                XFree(xproperty.value);
            }

            XSetWindowAttributes attr = {0};
            if ((appCtx[i]->config.tiled_display_config.enable &&
                 appCtx[i]->config.tiled_display_config.rows *
                         appCtx[i]->config.tiled_display_config.columns ==
                     1) ||
                (appCtx[i]->config.tiled_display_config.enable == 0))
            {
                attr.event_mask = KeyPress;
            }
            else if (appCtx[i]->config.tiled_display_config.enable)
            {
                attr.event_mask = ButtonPress | KeyRelease;
            }
            XChangeWindowAttributes(display, windows[i], CWEventMask, &attr);

            Atom wmDeleteMessage =
                XInternAtom(display, "WM_DELETE_WINDOW", False);
            if (wmDeleteMessage != None)
            {
                XSetWMProtocols(display, windows[i], &wmDeleteMessage, 1);
            }
            XMapRaised(display, windows[i]);
            XSync(display, 1); // discard the events for now
            gst_video_overlay_set_window_handle(
                GST_VIDEO_OVERLAY(appCtx[i]
                                      ->pipeline.instance_bins[0]
                                      .sink_bin.sub_bins[j]
                                      .sink),
                (gulong)windows[i]);
            gst_video_overlay_expose(
                GST_VIDEO_OVERLAY(appCtx[i]
                                      ->pipeline.instance_bins[0]
                                      .sink_bin.sub_bins[j]
                                      .sink));
            if (!x_event_thread)
                x_event_thread = g_thread_new("nvds-window-event-thread",
                                              nvds_x_event_thread, NULL);
        }
#if !defined(__aarch64__)
        if (!prop.integrated)
        {
            if (gst_element_set_state(appCtx[i]->pipeline.pipeline,
                                      GST_STATE_PAUSED) ==
                GST_STATE_CHANGE_FAILURE)
            {
                NVGSTDS_ERR_MSG_V("Failed to set pipeline to PAUSED");
                return_value = -1;
                goto done;
            }
        }
#endif
    }

    /* Dont try to set playing state if error is observed */
    if (return_value != -1)
    {
        for (i = 0; i < num_instances; i++)
        {
            if (gst_element_set_state(appCtx[i]->pipeline.pipeline,
                                      GST_STATE_PLAYING) ==
                GST_STATE_CHANGE_FAILURE)
            {

                g_print("\ncan't set pipeline to playing state.\n");
                return_value = -1;
                goto done;
            }
            if (appCtx[i]->config.pipeline_recreate_sec)
                g_timeout_add_seconds(appCtx[i]->config.pipeline_recreate_sec,
                                      recreate_pipeline_thread_func, appCtx[i]);
        }
    }

    print_runtime_commands();

    changemode(1);

    g_timeout_add(40, event_thread_func, NULL);
    g_main_loop_run(main_loop);

    changemode(0);

done:

    g_print("Quitting\n");
    for (i = 0; i < num_instances; i++)
    {
        if (appCtx[i]->return_value == -1)
            return_value = -1;
        destroy_pipeline(appCtx[i]);

        /* free classification aggregator cache */
        if (appCtx[i]->cls_agg_map)
        {
            /* free values: ObjClsAgg */
            GHashTableIter it;
            gpointer       k, v;
            g_hash_table_iter_init(&it, appCtx[i]->cls_agg_map);
            while (g_hash_table_iter_next(&it, &k, &v))
            {
                /* free key */
                g_free(k);
                /* free value */
                typedef struct _ObjClsAgg
                {
                    GHashTable *label_scores;
                    guint       last_seen_frame;
                } ObjClsAgg;
                ObjClsAgg *agg = (ObjClsAgg *)v;
                if (agg)
                {
                    if (agg->label_scores)
                        g_hash_table_destroy(agg->label_scores);
                    g_free(agg);
                }
            }
            g_hash_table_destroy(appCtx[i]->cls_agg_map);
            appCtx[i]->cls_agg_map = NULL;
        }

        g_mutex_lock(&disp_lock);
        if (windows[i])
            XDestroyWindow(display, windows[i]);
        windows[i] = 0;
        g_mutex_unlock(&disp_lock);

        g_free(appCtx[i]);
    }

    g_mutex_lock(&disp_lock);
    if (display)
        XCloseDisplay(display);
    display = NULL;
    g_mutex_unlock(&disp_lock);
    g_mutex_clear(&disp_lock);

    if (main_loop)
    {
        g_main_loop_unref(main_loop);
    }

    if (ctx)
    {
        g_option_context_free(ctx);
    }

    if (return_value == 0)
    {
        g_print("App run successful\n");
    }
    else
    {
        g_print("App run failed\n");
    }

    gst_deinit();

    return return_value;
}
