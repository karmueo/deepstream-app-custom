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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib/gstdio.h>
#include "deepstream_app.h"
#include "deepstream_app_callbacks.h"
#include "deepstream_app_probes.h"
#include "nvds_obj_encode.h"
#include "gstudpjsonmeta.h"

GST_DEBUG_CATEGORY_EXTERN(NVDS_APP);

GQuark _dsmeta_quark;

#define CEIL(a, b) ((a + b - 1) / b)
#define DEFAULT_CUAV_LOG_PATH "/tmp/deepstream_cuav_packets.log"
#define DEFAULT_CUAV_TEST_MULTICAST_IP "239.255.88.51"
#define DEFAULT_CUAV_TEST_MULTICAST_PORT 18003
#define CUAV_FEEDBACK_STALE_USEC (2 * G_USEC_PER_SEC)
#define CUAV_MOTION_CMD_MIN_SPACING_USEC (70 * 1000)

typedef enum
{
    CUAV_MOTION_CMD_NONE = 0,
    CUAV_MOTION_CMD_SERVO = 1,
    CUAV_MOTION_CMD_VISIBLE = 2,
} CuavMotionCommandType;

static GMutex s_cuav_csv_lock;

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
static gboolean create_cuav_control_element(NvDsConfig *config,
                                            NvDsPipeline *pipeline);
static NvDsSinkSubBinConfig *find_cuav_control_sink_config(NvDsConfig *config);
static gboolean emit_cuav_control_signal(AppCtx *appCtx,
                                         const gchar *signal_name,
                                         GstStructure *payload);
static gboolean update_cuav_eo_system_state(AppCtx *appCtx,
                                            const CuavFeedbackState *feedback_state);
static gboolean send_cuav_servo_command(AppCtx *appCtx,
                                        guint dev_id,
                                        guint dev_en,
                                        guint ctrl_en,
                                        guint mode_h,
                                        guint mode_v,
                                        guint speed_h,
                                        guint speed_v,
                                        gdouble loc_h,
                                        gdouble loc_v);
static gboolean send_cuav_servo_command_with_en(AppCtx *appCtx,
                                                guint dev_id,
                                                guint dev_en,
                                                guint ctrl_en,
                                                guint mode_h,
                                                guint mode_v,
                                                guint speed_h,
                                                guint speed_v,
                                                guint loc_en_h,
                                                gdouble loc_h,
                                                guint loc_en_v,
                                                gdouble loc_v);
static gboolean send_cuav_visible_light_command_with_en(AppCtx *appCtx,
                                                        guint pt_focal_en,
                                                        gdouble pt_focal,
                                                        guint pt_focus_en,
                                                        guint pt_focus,
                                                        guint pt_focus_mode,
                                                        guint pt_zoom);
static gboolean send_cuav_visible_light_command(AppCtx *appCtx,
                                                gdouble pt_focal,
                                                guint pt_focus_en,
                                                guint pt_focus,
                                                guint pt_focus_mode,
                                                guint pt_zoom);
static gboolean send_cuav_infrared_command(AppCtx *appCtx,
                                           gdouble ir_focal,
                                           guint ir_focus,
                                           guint ir_focus_mode,
                                           guint ir_zoom);
static gboolean send_cuav_servo_test_message(AppCtx *appCtx);
static gboolean send_cuav_visible_light_test_message(AppCtx *appCtx);
static gboolean send_cuav_infrared_test_message(AppCtx *appCtx);
static gdouble clamp_cuav_double(gdouble value, gdouble min_value, gdouble max_value);
static guint clamp_cuav_uint(guint value, guint min_value, guint max_value);
static gdouble wrap_heading_360(gdouble value);
static gdouble cuav_heading_delta(gdouble current, gdouble baseline);
static gboolean cuav_is_test_target(const NvDsCuavControlConfig *control_config);
static gboolean cuav_feedback_is_fresh(const CuavFeedbackState *feedback_state,
                                       guint stale_timeout_ms);
static gboolean cuav_startup_preset_in_progress(const CuavStartupPresetState *state);
static gboolean cuav_startup_preset_has_home_target(const NvDsCuavControlConfig *control_config);
static gboolean cuav_startup_preset_has_visible_preset(const NvDsCuavControlConfig *control_config);
static void cuav_reset_startup_preset_state(CuavStartupPresetState *state);
static gboolean process_cuav_startup_preset(AppCtx *appCtx,
                                            const NvDsCuavControlConfig *control_config,
                                            gint64 now_us);
static void cuav_reset_corner_zoom_cycle_state(CuavCornerZoomCycleState *state,
                                               const NvDsCuavControlConfig *control_config);
static gboolean cuav_corner_zoom_cycle_resolve_home_target(const NvDsCuavControlConfig *control_config,
                                                           const CuavFeedbackState *feedback_state,
                                                           gboolean feedback_fresh,
                                                           const CuavCornerZoomCycleState *state,
                                                           gdouble *loc_h,
                                                           gdouble *loc_v);
static gboolean cuav_corner_zoom_cycle_home_reached(const CuavFeedbackState *feedback_state,
                                                    const CuavCornerZoomCycleState *state,
                                                    const NvDsCuavControlConfig *control_config);
static void cuav_reset_auto_control_state(CuavAutoControlState *state, gboolean keep_last_commands);
static const gchar *cuav_corner_zoom_cycle_corner_name(guint corner_index);
static void cuav_corner_zoom_cycle_compute_target(gdouble base_h,
                                                  gdouble base_v,
                                                  gdouble offset_h,
                                                  gdouble offset_v,
                                                  guint corner_index,
                                                  gdouble *loc_h,
                                                  gdouble *loc_v);
static gboolean process_cuav_corner_zoom_cycle(AppCtx *appCtx,
                                               const NvDsCuavControlConfig *control_config,
                                               gint64 now_us);
static gboolean cuav_push_track_sample(CuavAutoControlState *state,
                                       guint history_size,
                                       const CuavTrackSample *sample);
static gboolean cuav_compute_average_velocity(const CuavAutoControlState *state,
                                              guint history_size,
                                              gdouble *vel_x,
                                              gdouble *vel_y);
static gboolean cuav_compute_servo_command(const NvDsCuavControlConfig *control_config,
                                           const CuavFeedbackState *feedback_state,
                                           const CuavAutoControlState *auto_state,
                                           const CuavTrackSample *sample,
                                           gdouble vel_x,
                                           gdouble vel_y,
                                           gdouble *loc_h,
                                           gdouble *loc_v,
                                           guint *speed_h,
                                           guint *speed_v,
                                           gboolean debug);
static gboolean cuav_compute_visible_light_command(const NvDsCuavControlConfig *control_config,
                                                   const CuavAutoControlState *auto_state,
                                                   const CuavTrackSample *sample,
                                                   guint *pt_focal_en,
                                                   gdouble *pt_focal,
                                                   guint *pt_focus);
static gboolean cuav_compute_infrared_command(const NvDsCuavControlConfig *control_config,
                                              const CuavFeedbackState *feedback_state,
                                              const CuavAutoControlState *auto_state,
                                              const CuavTrackSample *sample,
                                              gdouble *ir_focal,
                                              guint *ir_focus);
static void cuav_fill_simulated_sample(const NvDsCuavControlConfig *control_config,
                                       gint64 now_us,
                                       CuavTrackSample *sample);

/**
 * @brief 根据源ID获取传感器信息
 * @param appCtx 应用上下文
 * @param source_id 视频源索引
 * @return 传感器信息指针，未找到返回NULL
 */
NvDsSensorInfo *get_sensor_info(AppCtx *appCtx, guint source_id)
{
    NvDsSensorInfo *sensorInfo = (NvDsSensorInfo *)g_hash_table_lookup(appCtx->sensorInfoHash,
                                                                       source_id + (gchar *)NULL);
    return sensorInfo;
}

/**
 * @brief 获取CUAV日志文件路径
 * @return 优先返回环境变量DEEPSTREAM_CUAV_LOG_PATH，否则返回默认路径
 */
static const gchar *
get_cuav_log_path(void)
{
    const gchar *path = g_getenv("DEEPSTREAM_CUAV_LOG_PATH");
    return (path && *path) ? path : DEFAULT_CUAV_LOG_PATH;
}

/**
 * @brief 追加一行日志到CUAV日志文件
 * @param line 日志内容
 */
static void
append_cuav_log_line(const gchar *line)
{
    FILE *fp = NULL;

    if (!line)
        return;

    fp = fopen(get_cuav_log_path(), "a");
    if (!fp)
    {
        g_printerr("[cuav][log] failed to open %s\n", get_cuav_log_path());
        return;
    }

    fprintf(fp, "%s\n", line);
    fclose(fp);
}

/**
 * @brief 构造CUAV CSV输出文件的完整路径
 * @param appCtx 应用上下文
 * @param filename CSV文件名
 * @param path 输出路径缓冲区
 * @param path_size 缓冲区大小
 * @return 成功返回TRUE，未启用CSV记录或参数无效返回FALSE
 */
static gboolean
get_cuav_csv_path(AppCtx *appCtx, const gchar *filename, gchar *path, gsize path_size)
{
    const gchar *dir = NULL;

    if (!appCtx || !filename || !path || path_size == 0)
        return FALSE;

    if (!appCtx->config.udpjsonmeta_config.record_parsed_csv)
        return FALSE;

    dir = appCtx->config.udpjsonmeta_config.parsed_csv_output_dir;
    if (!dir || !*dir)
        return FALSE;

    if (g_mkdir_with_parents(dir, 0755) != 0)
    {
        g_printerr("[cuav][csv] failed to create dir %s\n", dir);
        return FALSE;
    }

    g_snprintf(path, path_size, "%s/%s", dir, filename);
    return TRUE;
}

/**
 * @brief 追加一行数据到CSV文件，文件不存在时自动写入表头
 * @param path CSV文件路径
 * @param header CSV表头行
 * @param row CSV数据行
 */
static void
append_cuav_csv_row(const gchar *path, const gchar *header, const gchar *row)
{
    GStatBuf st;
    gboolean need_header = FALSE;
    FILE *fp = NULL;

    if (!path || !header || !row)
        return;

    g_mutex_lock(&s_cuav_csv_lock);

    need_header = (g_stat(path, &st) != 0) || (st.st_size == 0);
    fp = fopen(path, "a");
    if (!fp)
    {
        g_printerr("[cuav][csv] failed to open %s\n", path);
        g_mutex_unlock(&s_cuav_csv_lock);
        return;
    }

    if (need_header)
    {
        fprintf(fp, "%s\n", header);
    }
    fprintf(fp, "%s\n", row);
    fclose(fp);
    g_mutex_unlock(&s_cuav_csv_lock);
}

/**
 * @brief 在配置中查找类型为CUAVCONTROL的sink子配置
 * @param config 全局配置
 * @return 找到的sink配置指针，未找到返回NULL
 */
static NvDsSinkSubBinConfig *
find_cuav_control_sink_config(NvDsConfig *config)
{
    guint i = 0;

    if (!config)
        return NULL;

    for (i = 0; i < config->num_sink_sub_bins; i++)
    {
        NvDsSinkSubBinConfig *sink_config = &config->sink_bin_sub_bin_config[i];
        if (sink_config->enable && sink_config->type == NV_DS_SINK_CUAVCONTROL)
            return sink_config;
    }

    return NULL;
}

/**
 * @brief 将gdouble值限制在[min_value, max_value]范围内
 */
static gdouble
clamp_cuav_double(gdouble value, gdouble min_value, gdouble max_value)
{
    if (value < min_value)
        return min_value;
    if (value > max_value)
        return max_value;
    return value;
}

/**
 * @brief 将guint值限制在[min_value, max_value]范围内
 */
static guint
clamp_cuav_uint(guint value, guint min_value, guint max_value)
{
    if (value < min_value)
        return min_value;
    if (value > max_value)
        return max_value;
    return value;
}

/**
 * @brief 将航向角归一化到[0, 360)范围
 */
static gdouble
wrap_heading_360(gdouble value)
{
    while (value < 0.0)
        value += 360.0;
    while (value >= 360.0)
        value -= 360.0;
    return value;
}

/**
 * @brief 计算两个航向角之间的最短角度差（0~180度）
 */
static gdouble
cuav_heading_delta(gdouble current, gdouble baseline)
{
    gdouble delta = fabs(wrap_heading_360(current) - wrap_heading_360(baseline));
    return MIN(delta, 360.0 - delta);
}

/**
 * @brief 判断当前控制目标是否为测试目标（特定组播地址+端口）
 */
static gboolean
cuav_is_test_target(const NvDsCuavControlConfig *control_config)
{
    if (!control_config || !control_config->multicast_ip)
        return FALSE;

    return g_strcmp0(control_config->multicast_ip,
                     DEFAULT_CUAV_TEST_MULTICAST_IP) == 0 &&
           control_config->port == DEFAULT_CUAV_TEST_MULTICAST_PORT;
}

/**
 * @brief 判断可见光控制是否启用
 */
static gboolean
cuav_visible_control_enabled(const NvDsCuavControlConfig *control_config)
{
    return control_config && control_config->visible_light_control_enable;
}

/**
 * @brief 判断红外控制是否启用
 */
static gboolean
cuav_infrared_control_enabled(const NvDsCuavControlConfig *control_config)
{
    return control_config && control_config->infrared_control_enable;
}

/**
 * @brief 检查设备反馈状态是否在有效期内（未过期）
 * @param feedback_state 设备反馈状态
 * @param stale_timeout_ms 过期超时时间（毫秒），0则默认2000ms
 * @return 反馈有效且未过期返回TRUE
 */
static gboolean
cuav_feedback_is_fresh(const CuavFeedbackState *feedback_state,
                       guint stale_timeout_ms)
{
    gint64 timeout_us = 0;

    if (!feedback_state || !feedback_state->valid)
        return FALSE;

    timeout_us = ((gint64)(stale_timeout_ms > 0 ? stale_timeout_ms : 2000)) * 1000;
    return (g_get_monotonic_time() - feedback_state->updated_at_us) <= timeout_us;
}

/**
 * @brief 判断启动预置位流程是否正在进行中
 */
static gboolean
cuav_startup_preset_in_progress(const CuavStartupPresetState *state)
{
    return state && state->initialized &&
           state->phase != CUAV_STARTUP_PRESET_PHASE_COMPLETE;
}

/**
 * @brief 判断启动预置位是否配置了回中目标（水平或垂直方位）
 */
static gboolean
cuav_startup_preset_has_home_target(const NvDsCuavControlConfig *control_config)
{
    return control_config &&
           !(control_config->corner_zoom_cycle_enable &&
             !control_config->corner_servo_enable) &&
           (!isnan(control_config->corner_home_loc_h_deg) ||
            !isnan(control_config->corner_home_loc_v_deg));
}

/**
 * @brief 判断启动预置位是否配置了可见光预置（焦距缩到最小或对焦值）
 */
static gboolean
cuav_startup_preset_has_visible_preset(const NvDsCuavControlConfig *control_config)
{
    return control_config &&
           (control_config->startup_pt_focal_min_enable ||
            control_config->corner_home_pt_focus != G_MAXUINT);
}

/**
 * @brief 重置启动预置位状态机到初始IDLE阶段
 * @param state 启动预置位状态
 */
static void
cuav_reset_startup_preset_state(CuavStartupPresetState *state)
{
    if (!state)
        return;

    memset(state, 0, sizeof(*state));
    state->phase = CUAV_STARTUP_PRESET_PHASE_IDLE;
    state->home_loc_h = 180.0;
    state->home_loc_v = 0.0;
}

/**
 * @brief 处理启动预置位状态机
 * 流程: IDLE → 发送回中云台指令 → 等待到位 → 发送可见光预置 → 等待保持 → 完成
 * 完成后解除对自动跟踪的阻塞
 * @param appCtx 应用上下文
 * @param control_config 控制配置
 * @param now_us 当前单调时钟时间（微秒）
 * @return TRUE表示正常处理中或已完成
 */
static gboolean
process_cuav_startup_preset(AppCtx *appCtx,
                            const NvDsCuavControlConfig *control_config,
                            gint64 now_us)
{
    CuavStartupPresetState state_snapshot;
    CuavFeedbackState feedback_snapshot;
    gboolean feedback_fresh = FALSE;
    gboolean has_home_target = FALSE;
    gboolean visible_requested = FALSE;
    gboolean has_visible_preset = FALSE;
    gboolean visible_enabled = FALSE;
    gdouble home_loc_h = 180.0;
    gdouble home_loc_v = 0.0;
    gboolean home_reached = FALSE;
    gint64 settle_timeout_us = 0;
    gint64 visible_hold_us = 0;
    gint64 visible_repeat_gap_us = 0;
    gboolean sent = FALSE;

    if (!appCtx || !control_config)
        return TRUE;

    has_home_target = cuav_startup_preset_has_home_target(control_config);
    visible_enabled = cuav_visible_control_enabled(control_config);
    visible_requested = cuav_startup_preset_has_visible_preset(control_config);
    has_visible_preset = visible_enabled && visible_requested;
    if (!visible_enabled && visible_requested && control_config->debug)
    {
        g_print("[cuav][startup-preset][warn] visible preset configured but visible-light-control-enable=0, skip visible preset\n");
    }
    if (!has_home_target && !has_visible_preset)
        return TRUE;

    settle_timeout_us = ((gint64)MAX(control_config->state_stale_timeout_ms, 1U)) * 1000;
    visible_hold_us = ((gint64)MAX(control_config->startup_pt_focal_min_hold_ms, 0U)) * 1000;
    visible_repeat_gap_us = ((gint64)MAX(control_config->control_period_ms, 1U)) * 1000;

    g_mutex_lock(&appCtx->cuav_control_lock);
    if (!appCtx->cuav_startup_preset_state.initialized)
    {
        cuav_reset_startup_preset_state(&appCtx->cuav_startup_preset_state);
        appCtx->cuav_startup_preset_state.initialized = TRUE;
        appCtx->cuav_startup_preset_state.phase =
            has_home_target ? CUAV_STARTUP_PRESET_PHASE_SEND_HOME_SERVO :
                              (has_visible_preset ? CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET :
                                                    CUAV_STARTUP_PRESET_PHASE_COMPLETE);
        appCtx->cuav_startup_preset_state.phase_started_us = now_us;
        appCtx->cuav_startup_preset_state.last_command_sent_us = 0;
    }
    feedback_snapshot = appCtx->cuav_feedback_state;
    state_snapshot = appCtx->cuav_startup_preset_state;
    g_mutex_unlock(&appCtx->cuav_control_lock);

    feedback_fresh = cuav_feedback_is_fresh(&feedback_snapshot,
                                            control_config->state_stale_timeout_ms);

    if (state_snapshot.phase == CUAV_STARTUP_PRESET_PHASE_COMPLETE)
    {
        if (!state_snapshot.final_logged)
        {
            g_print("[cuav][startup-preset] complete home=%d visible=%d\n",
                    state_snapshot.servo_applied,
                    state_snapshot.visible_applied);
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.final_logged = TRUE;
            g_mutex_unlock(&appCtx->cuav_control_lock);
        }
        return TRUE;
    }

    switch (state_snapshot.phase)
    {
    case CUAV_STARTUP_PRESET_PHASE_IDLE:
        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_startup_preset_state.phase =
            has_home_target ? CUAV_STARTUP_PRESET_PHASE_SEND_HOME_SERVO :
                              (has_visible_preset ? CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET :
                                                    CUAV_STARTUP_PRESET_PHASE_COMPLETE);
        appCtx->cuav_startup_preset_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_SEND_HOME_SERVO:
        if (!has_home_target)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.servo_applied = FALSE;
            appCtx->cuav_startup_preset_state.phase =
                has_visible_preset ? CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET :
                                    CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < 1000)
            return TRUE;

        home_loc_h = !isnan(control_config->corner_home_loc_h_deg) ?
                         wrap_heading_360(control_config->corner_home_loc_h_deg) :
                         180.0;
        home_loc_v = !isnan(control_config->corner_home_loc_v_deg) ?
                         clamp_cuav_double(control_config->corner_home_loc_v_deg, -90.0, 90.0) :
                         0.0;
        sent = send_cuav_servo_command_with_en(appCtx,
                                               control_config->servo_dev_id,
                                               1, 1, 0, 0,
                                               MAX(control_config->corner_servo_speed, 1U),
                                               MAX(control_config->corner_servo_speed, 1U),
                                               1, home_loc_h,
                                               1, home_loc_v);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.home_target_valid = TRUE;
            appCtx->cuav_startup_preset_state.home_loc_h = home_loc_h;
            appCtx->cuav_startup_preset_state.home_loc_v = home_loc_v;
            appCtx->cuav_startup_preset_state.servo_applied = TRUE;
            appCtx->cuav_startup_preset_state.last_command_sent_us = now_us;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_HOLD_HOME_SERVO;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][startup-preset] send home loc=(%.2f,%.2f) speed=%u\n",
                        home_loc_h, home_loc_v,
                        MAX(control_config->corner_servo_speed, 1U));
            }
        }
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_HOLD_HOME_SERVO:
        if (!has_home_target)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.phase =
                has_visible_preset ? CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET :
                                    CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        home_loc_h = state_snapshot.home_loc_h;
        home_loc_v = state_snapshot.home_loc_v;
        home_reached = feedback_fresh &&
                       cuav_heading_delta(feedback_snapshot.st_loc_h, home_loc_h) <=
                           MAX(control_config->servo_effect_threshold_h, 0.0) &&
                       fabs(feedback_snapshot.st_loc_v - home_loc_v) <=
                           MAX(control_config->servo_effect_threshold_v, 0.0);
        if (home_reached || ((now_us - state_snapshot.phase_started_us) >= settle_timeout_us))
        {
            if (control_config->debug && !home_reached)
            {
                g_print("[cuav][startup-preset][warn] home settle timeout, continue to visible preset\n");
            }
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.phase =
                has_visible_preset ? CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET :
                                    CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_SEND_VISIBLE_PRESET:
        if (!has_visible_preset)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < 1000)
            return TRUE;

        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       control_config->startup_pt_focal_min_enable ? 4 : 0,
                                                       control_config->startup_pt_focal_min_enable ? 8.0 : 0.0,
                                                       control_config->corner_home_pt_focus != G_MAXUINT ? 1 : 0,
                                                       control_config->corner_home_pt_focus != G_MAXUINT ? control_config->corner_home_pt_focus : 100,
                                                       1,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.visible_applied = TRUE;
            appCtx->cuav_startup_preset_state.last_command_sent_us = now_us;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            appCtx->cuav_startup_preset_state.phase =
                control_config->startup_pt_focal_min_enable ?
                    CUAV_STARTUP_PRESET_PHASE_HOLD_VISIBLE_PRESET :
                    CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][startup-preset] send visible focal_en=%u focus_en=%u\n",
                        control_config->startup_pt_focal_min_enable ? 4U : 0U,
                        control_config->corner_home_pt_focus != G_MAXUINT ? 1U : 0U);
            }
        }
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_HOLD_VISIBLE_PRESET:
        if (!control_config->startup_pt_focal_min_enable)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        if ((now_us - state_snapshot.phase_started_us) < visible_hold_us)
        {
            if (state_snapshot.last_command_sent_us <= 0 ||
                (now_us - state_snapshot.last_command_sent_us) >= visible_repeat_gap_us)
            {
                sent = send_cuav_visible_light_command_with_en(appCtx,
                                                               4,
                                                               8.0,
                                                               0,
                                                               0,
                                                               1,
                                                               0);
                if (sent)
                {
                    g_mutex_lock(&appCtx->cuav_control_lock);
                    appCtx->cuav_startup_preset_state.last_command_sent_us = now_us;
                    g_mutex_unlock(&appCtx->cuav_control_lock);

                    if (control_config->debug)
                    {
                        g_print("[cuav][startup-preset] repeat visible focal_min focal_en=4 elapsed=%" G_GINT64_FORMAT "/%" G_GINT64_FORMAT " us\n",
                                now_us - state_snapshot.phase_started_us,
                                visible_hold_us);
                    }
                }
            }
            return TRUE;
        }

        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < 1000)
            return TRUE;

        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       0,
                                                       0.0,
                                                       0,
                                                       0,
                                                       1,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.last_command_sent_us = now_us;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][startup-preset] stop visible focal after %u ms\n",
                        control_config->startup_pt_focal_min_hold_ms);
            }
        }
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_COMPLETE:
    default:
        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_startup_preset_state.final_logged = TRUE;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;
    }
}

/**
 * @brief 解析角点循环回预置位的目标位置
 * 优先级: 配置值 > 设备反馈值 > 上次状态值 > 默认值
 * @param[out] loc_h 解析后的水平方位角
 * @param[out] loc_v 解析后的俯仰角
 * @return 成功返回TRUE
 */
static gboolean
cuav_corner_zoom_cycle_resolve_home_target(const NvDsCuavControlConfig *control_config,
                                           const CuavFeedbackState *feedback_state,
                                           gboolean feedback_fresh,
                                           const CuavCornerZoomCycleState *state,
                                           gdouble *loc_h,
                                           gdouble *loc_v)
{
    if (!state || !loc_h || !loc_v)
        return FALSE;

    if (control_config && !isnan(control_config->corner_home_loc_h_deg))
        *loc_h = wrap_heading_360(control_config->corner_home_loc_h_deg);
    else if (feedback_fresh && feedback_state && feedback_state->valid)
        *loc_h = wrap_heading_360(feedback_state->st_loc_h);
    else if (state->home_target_valid)
        *loc_h = state->home_loc_h;
    else
        *loc_h = state->base_loc_h;

    if (control_config && !isnan(control_config->corner_home_loc_v_deg))
        *loc_v = clamp_cuav_double(control_config->corner_home_loc_v_deg, -90.0, 90.0);
    else if (feedback_fresh && feedback_state && feedback_state->valid)
        *loc_v = clamp_cuav_double(feedback_state->st_loc_v, -90.0, 90.0);
    else if (state->home_target_valid)
        *loc_v = state->home_loc_v;
    else
        *loc_v = state->base_loc_v;

    return TRUE;
}

/**
 * @brief 判断云台是否已到达预置位（基于设备反馈与阈值比较）
 */
static gboolean
cuav_corner_zoom_cycle_home_reached(const CuavFeedbackState *feedback_state,
                                    const CuavCornerZoomCycleState *state,
                                    const NvDsCuavControlConfig *control_config)
{
    gdouble threshold_h = 0.0;
    gdouble threshold_v = 0.0;

    if (!feedback_state || !state || !control_config || !feedback_state->valid)
        return FALSE;

    threshold_h = MAX(control_config->servo_effect_threshold_h, 0.0);
    threshold_v = MAX(control_config->servo_effect_threshold_v, 0.0);
    return cuav_heading_delta(feedback_state->st_loc_h, state->home_loc_h) <= threshold_h &&
           fabs(feedback_state->st_loc_v - state->home_loc_v) <= threshold_v;
}

/**
 * @brief 重置角点变焦循环状态机到初始阶段
 */
static void
cuav_reset_corner_zoom_cycle_state(CuavCornerZoomCycleState *state,
                                   const NvDsCuavControlConfig *control_config)
{
    if (!state)
        return;

    (void)control_config;

    memset(state, 0, sizeof(*state));
    state->phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
    state->home_target_valid = FALSE;
    state->home_loc_h = 180.0;
    state->home_loc_v = 0.0;
    state->base_loc_h = 180.0;
    state->base_loc_v = 0.0;
    state->return_home_before_zoom = FALSE;
    state->resume_cycle_after_home = FALSE;
    state->increment_repeat_after_home = FALSE;
    state->last_loc_h = 180.0;
    state->last_loc_v = 0.0;

    if (control_config)
    {
        state->last_loc_h = 180.0;
        state->last_loc_v = 0.0;
    }
}

/**
 * @brief 获取角点索引对应的名称（0=右上, 1=左上, 2=左下, 3=右下）
 */
static const gchar *
cuav_corner_zoom_cycle_corner_name(guint corner_index)
{
    switch (corner_index)
    {
    case 0:
        return "right-up";
    case 1:
        return "left-up";
    case 2:
        return "left-down";
    case 3:
        return "right-down";
    default:
        return "unknown";
    }
}

/**
 * @brief 根据基准位置和偏移量计算指定角点的目标位置
 * @param base_h 基准水平方位角
 * @param base_v 基准俯仰角
 * @param offset_h 水平偏移量（度）
 * @param offset_v 垂直偏移量（度）
 * @param corner_index 角点索引(0~3)
 * @param[out] loc_h 目标水平方位角
 * @param[out] loc_v 目标俯仰角
 */
static void
cuav_corner_zoom_cycle_compute_target(gdouble base_h,
                                      gdouble base_v,
                                      gdouble offset_h,
                                      gdouble offset_v,
                                      guint corner_index,
                                      gdouble *loc_h,
                                      gdouble *loc_v)
{
    gdouble delta_h = fabs(offset_h);
    gdouble delta_v = fabs(offset_v);
    gdouble target_h = base_h;
    gdouble target_v = base_v;

    switch (corner_index)
    {
    case 0:
        target_h = base_h + delta_h;
        target_v = base_v + delta_v;
        break;
    case 1:
        target_h = base_h - delta_h;
        target_v = base_v + delta_v;
        break;
    case 2:
        target_h = base_h - delta_h;
        target_v = base_v - delta_v;
        break;
    case 3:
        target_h = base_h + delta_h;
        target_v = base_v - delta_v;
        break;
    default:
        break;
    }

    if (loc_h)
        *loc_h = wrap_heading_360(target_h);
    if (loc_v)
        *loc_v = clamp_cuav_double(target_v, -90.0, 90.0);
}

/**
 * @brief 重置自动跟踪控制状态
 * @param state 自动控制状态
 * @param keep_last_commands 是否保留上次发送的指令值（用于目标切换时无缝衔接）
 */
static void
cuav_reset_auto_control_state(CuavAutoControlState *state, gboolean keep_last_commands)
{
    if (!state)
        return;

    state->has_lock = FALSE;
    state->locked_object_id = 0;
    state->last_target_seen_us = 0;
    state->target_stable_since_us = 0;
    state->lost_zoom_active = FALSE;
    state->lost_zoom_start_us = 0;
    state->lost_zoom_hold_complete = FALSE;
    state->history_len = 0;
    state->history_next = 0;
    memset(state->history, 0, sizeof(state->history));

    if (!keep_last_commands)
    {
        state->last_servo_valid = FALSE;
        state->last_visible_valid = FALSE;
        state->last_pt_focal_en = 0;
        state->visible_initialized = FALSE;
        state->last_loc_h = 180.0;
        state->last_loc_v = 0.0;
        state->last_speed_h = 0;
        state->last_speed_v = 0;
        state->last_pt_focal = 500.0;
        state->last_pt_focus = 100;
        state->last_infrared_valid = FALSE;
        state->last_ir_focal = 900.0;
        state->last_ir_focus = 5;
        state->last_servo_send_us = 0;
        state->last_visible_send_us = 0;
        state->last_infrared_send_us = 0;
        state->last_motion_send_us = 0;
        state->last_motion_type = CUAV_MOTION_CMD_NONE;
    }
    state->infrared_initialized = FALSE;
}

/**
 * @brief 将跟踪采样点压入循环历史缓冲区
 * @param state 自动控制状态（含历史缓冲区）
 * @param history_size 历史缓冲区容量
 * @param sample 当前帧的跟踪采样数据
 * @return 成功返回TRUE
 */
static gboolean
cuav_push_track_sample(CuavAutoControlState *state,
                       guint history_size,
                       const CuavTrackSample *sample)
{
    guint capacity = 0;

    if (!state || !sample)
        return FALSE;

    capacity = history_size;
    if (capacity == 0)
        capacity = 1;
    if (capacity > CUAV_AUTO_CONTROL_HISTORY_MAX)
        capacity = CUAV_AUTO_CONTROL_HISTORY_MAX;

    state->history[state->history_next] = *sample;
    state->history_next = (state->history_next + 1) % capacity;
    if (state->history_len < capacity)
        state->history_len++;

    return TRUE;
}

/**
 * @brief 根据历史采样缓冲区计算平均速度（用于速度前馈控制）
 * @param state 自动控制状态
 * @param history_size 历史缓冲区容量
 * @param[out] vel_x 水平归一化误差变化速率
 * @param[out] vel_y 垂直归一化误差变化速率
 * @return 采样点不足2个时返回FALSE
 */
static gboolean
cuav_compute_average_velocity(const CuavAutoControlState *state,
                              guint history_size,
                              gdouble *vel_x,
                              gdouble *vel_y)
{
    guint capacity = 0;
    guint count = 0;
    guint start = 0;
    const CuavTrackSample *first = NULL;
    const CuavTrackSample *last = NULL;
    gdouble dt_sec = 0.0;

    if (vel_x)
        *vel_x = 0.0;
    if (vel_y)
        *vel_y = 0.0;

    if (!state || !vel_x || !vel_y || state->history_len < 2)
        return FALSE;

    capacity = history_size;
    if (capacity == 0)
        capacity = 1;
    if (capacity > CUAV_AUTO_CONTROL_HISTORY_MAX)
        capacity = CUAV_AUTO_CONTROL_HISTORY_MAX;

    count = MIN(state->history_len, capacity);
    if (count < 2)
        return FALSE;

    start = (state->history_next + capacity - count) % capacity;
    first = &state->history[start];
    last = &state->history[(start + count - 1) % capacity];

    dt_sec = (last->sample_time_us - first->sample_time_us) / 1000000.0;
    if (dt_sec <= 0.0)
        return FALSE;

    *vel_x = (last->err_x - first->err_x) / dt_sec;
    *vel_y = (last->err_y - first->err_y) / dt_sec;
    return TRUE;
}

/**
 * @brief 计算云台伺服控制指令
 * 采用P控制+速度前馈: delta = kp * err + kv * vel
 * 支持焦距自适应: 长焦时自动缩小步进和降低速度，防止振荡
 * @param control_config 控制配置
 * @param feedback_state 设备反馈状态（提供基准位置）
 * @param auto_state 自动控制状态（提供上次发送值）
 * @param sample 当前跟踪采样
 * @param vel_x 水平速度
 * @param vel_y 垂直速度
 * @param[out] loc_h 目标水平方位角
 * @param[out] loc_v 目标俯仰角
 * @param[out] speed_h 水平运动速度
 * @param[out] speed_v 垂直运动速度
 * @param debug 是否输出调试日志
 * @return 成功返回TRUE
 */
static gboolean
cuav_compute_servo_command(const NvDsCuavControlConfig *control_config,
                           const CuavFeedbackState *feedback_state,
                           const CuavAutoControlState *auto_state,
                           const CuavTrackSample *sample,
                           gdouble vel_x,
                           gdouble vel_y,
                           gdouble *loc_h,
                           gdouble *loc_v,
                           guint *speed_h,
                           guint *speed_v,
                           gboolean debug)
{
    gdouble base_h = 180.0;
    gdouble base_v = 0.0;
    gdouble delta_h = 0.0;
    gdouble delta_v = 0.0;
    gdouble max_step_h = 0.0;
    gdouble max_step_v = 0.0;
    gdouble movement_norm = 0.0;
    gdouble focal_norm = 0.0;
    gdouble step_scale = 1.0;
    gdouble speed_scale = 1.0;
    gdouble focal = 0.0;
    guint speed = 0;
    const gchar *base_source = "initial";
    gboolean h_in_deadband = FALSE;
    gboolean v_in_deadband = FALSE;

    if (!control_config || !auto_state || !sample ||
        !loc_h || !loc_v || !speed_h || !speed_v)
        return FALSE;

    if (feedback_state && feedback_state->valid &&
        (g_get_monotonic_time() - feedback_state->updated_at_us) <= CUAV_FEEDBACK_STALE_USEC)
    {
        base_h = feedback_state->st_loc_h;
        base_v = feedback_state->st_loc_v;
        base_source = "feedback";
    }
    else if (auto_state->last_servo_valid)
    {
        base_h = auto_state->last_loc_h;
        base_v = auto_state->last_loc_v;
        base_source = "last-servo";
    }
    else
    {
        base_source = "default";
    }

    h_in_deadband = fabs(sample->err_x) <= control_config->center_deadband_x;
    v_in_deadband = fabs(sample->err_y) <= control_config->center_deadband_y;
    max_step_h = control_config->servo_max_step_h;
    max_step_v = control_config->servo_max_step_v;

    if (control_config->servo_focal_adaptive_enable &&
        feedback_state && feedback_state->valid &&
        feedback_state->pt_focal > 0.0 &&
        control_config->pt_focal_max > control_config->pt_focal_min &&
        (g_get_monotonic_time() - feedback_state->updated_at_us) <= CUAV_FEEDBACK_STALE_USEC)
    {
        gdouble min_step_scale = clamp_cuav_double(
            control_config->servo_focal_max_step_scale_min, 0.05, 1.0);
        gdouble min_speed_scale = clamp_cuav_double(
            control_config->servo_focal_speed_scale_min, 0.05, 1.0);

        focal = clamp_cuav_double(feedback_state->pt_focal,
                                  control_config->pt_focal_min,
                                  control_config->pt_focal_max);
        focal_norm = (focal - control_config->pt_focal_min) /
                     (control_config->pt_focal_max - control_config->pt_focal_min);
        focal_norm = clamp_cuav_double(focal_norm, 0.0, 1.0);
        step_scale = 1.0 - (focal_norm * (1.0 - min_step_scale));
        speed_scale = 1.0 - (focal_norm * (1.0 - min_speed_scale));
        max_step_h *= step_scale;
        max_step_v *= step_scale;
    }

    if (!h_in_deadband)
    {
        delta_h = (control_config->servo_kp_x * sample->err_x) +
                  (control_config->servo_kv_x * vel_x);
        delta_h *= control_config->servo_dir_x;
        delta_h = clamp_cuav_double(delta_h,
                                    -max_step_h,
                                    max_step_h);
    }

    if (!v_in_deadband)
    {
        delta_v = (control_config->servo_kp_y * sample->err_y) +
                  (control_config->servo_kv_y * vel_y);
        delta_v *= control_config->servo_dir_y;
        delta_v = clamp_cuav_double(delta_v,
                                    -max_step_v,
                                    max_step_v);
    }

    *loc_h = wrap_heading_360(base_h + delta_h);
    *loc_v = clamp_cuav_double(base_v + delta_v, -90.0, 90.0);

    movement_norm = MAX(fabs(sample->err_x), fabs(sample->err_y));
    if (max_step_h > 0.0)
        movement_norm = MAX(movement_norm,
                            fabs(delta_h) / max_step_h);
    if (max_step_v > 0.0)
        movement_norm = MAX(movement_norm,
                            fabs(delta_v) / max_step_v);
    movement_norm = clamp_cuav_double(movement_norm, 0.0, 1.0);

    speed = control_config->servo_min_speed;
    if (control_config->servo_max_speed > control_config->servo_min_speed)
    {
        speed = control_config->servo_min_speed +
                (guint)round(movement_norm *
                             (control_config->servo_max_speed -
                              control_config->servo_min_speed));
    }
    speed = clamp_cuav_uint(speed,
                            control_config->servo_min_speed,
                            control_config->servo_max_speed);
    if (speed_scale < 1.0 && speed > control_config->servo_min_speed)
    {
        speed = control_config->servo_min_speed +
                (guint)round((speed - control_config->servo_min_speed) * speed_scale);
        speed = clamp_cuav_uint(speed,
                                control_config->servo_min_speed,
                                control_config->servo_max_speed);
    }
    *speed_h = speed;
    *speed_v = speed;

    if (debug)
    {
        g_print("[cuav][control][servo-compute] base=%s(%.2f,%.2f) err=(%.3f,%.3f) deadband=(%.3f,%.3f) "
                "vel=(%.3f,%.3f) focal=%.1f norm=%.3f scale=(step=%.3f,speed=%.3f) "
                "max_step=(%.2f,%.2f) delta=(%.2f,%.2f) out=(%.2f,%.2f) speed=(%u,%u)%s%s\n",
                base_source,
                base_h,
                base_v,
                sample->err_x,
                sample->err_y,
                control_config->center_deadband_x,
                control_config->center_deadband_y,
                vel_x,
                vel_y,
                focal,
                focal_norm,
                step_scale,
                speed_scale,
                max_step_h,
                max_step_v,
                delta_h,
                delta_v,
                *loc_h,
                *loc_v,
                *speed_h,
                *speed_v,
                h_in_deadband ? " h-deadband" : "",
                v_in_deadband ? " v-deadband" : "");
    }
    return TRUE;
}

/**
 * @brief 计算可见光变焦控制指令
 * target_ratio < (min - deadband) → focal_en=3(放大)
 * target_ratio > (max + deadband) → focal_en=4(缩小)
 * 否则 → focal_en=0(停止)
 * @param[out] pt_focal_en 变焦使能指令(0=停止,3=放大,4=缩小)
 * @param[out] pt_focal 焦距值（当前未使用，固定0.0）
 * @param[out] pt_focus 对焦值
 * @return 成功返回TRUE
 */
static gboolean
cuav_compute_visible_light_command(const NvDsCuavControlConfig *control_config,
                                   const CuavAutoControlState *auto_state,
                                   const CuavTrackSample *sample,
                                   guint *pt_focal_en,
                                   gdouble *pt_focal,
                                   guint *pt_focus)
{
    guint focal_en = 0;
    gint64 zoom_in_stable_us = 0;

    if (!control_config || !auto_state || !sample || !pt_focal_en || !pt_focal || !pt_focus)
        return FALSE;

    if (auto_state->last_visible_valid)
    {
        *pt_focus = auto_state->last_pt_focus;
    }
    else
    {
        if (control_config->corner_home_pt_focus != G_MAXUINT)
            *pt_focus = control_config->corner_home_pt_focus;
        else
            *pt_focus = 100;
    }

    if (sample->target_ratio < (control_config->zoom_target_ratio_min - control_config->zoom_deadband))
    {
        focal_en = 3;
        zoom_in_stable_us =
            ((gint64)MAX(control_config->visible_focal_hold_ms,
                         control_config->control_period_ms *
                            MAX(control_config->tracking_history_size, 1U))) * 1000;
        if (auto_state->target_stable_since_us <= 0 ||
            sample->sample_time_us - auto_state->target_stable_since_us < zoom_in_stable_us)
        {
            focal_en = 0;
        }
    }
    else if (sample->target_ratio > (control_config->zoom_target_ratio_max + control_config->zoom_deadband))
    {
        focal_en = 4;
    }
    else
    {
        focal_en = 0;
    }

    *pt_focal_en = focal_en;
    *pt_focal = 0.0;
    return TRUE;
}

/**
 * @brief 判断可见光变焦是否需要发送停止指令（hold定时器到期）
 * @return 上次发送变焦指令后超过visible_focal_hold_ms时返回TRUE
 */
static gboolean
cuav_visible_focal_stop_due(const NvDsCuavControlConfig *control_config,
                            const CuavAutoControlState *auto_state,
                            gint64 now_us)
{
    if (!control_config || !auto_state || !auto_state->visible_initialized)
        return FALSE;

    if (control_config->visible_focal_hold_ms == 0)
        return FALSE;

    if (auto_state->last_pt_focal_en == 0 || auto_state->last_visible_send_us <= 0)
        return FALSE;

    return (now_us - auto_state->last_visible_send_us) >=
           ((gint64)control_config->visible_focal_hold_ms * 1000);
}

/**
 * @brief 计算红外变焦控制指令（连续比例P控制）
 * delta = ir_zoom_kp * (target_ratio_min/max - current_ratio)
 * target_focal = base_focal + delta，限制在[ir_focal_min, ir_focal_max]
 * @param[out] ir_focal 目标红外焦距
 * @param[out] ir_focus 红外对焦值
 * @return 成功返回TRUE
 */
static gboolean
cuav_compute_infrared_command(const NvDsCuavControlConfig *control_config,
                              const CuavFeedbackState *feedback_state,
                              const CuavAutoControlState *auto_state,
                              const CuavTrackSample *sample,
                              gdouble *ir_focal,
                              guint *ir_focus)
{
    gdouble base_focal = 900.0;
    gdouble target_focal = 0.0;
    gdouble delta = 0.0;

    if (!control_config || !auto_state || !sample || !ir_focal || !ir_focus)
        return FALSE;

    if (feedback_state && feedback_state->valid &&
        feedback_state->ir_focal > 0.0 &&
        (g_get_monotonic_time() - feedback_state->updated_at_us) <= CUAV_FEEDBACK_STALE_USEC)
    {
        base_focal = feedback_state->ir_focal;
        *ir_focus = feedback_state->ir_focus > 0 ?
                        feedback_state->ir_focus :
                        MAX(control_config->ir_focus_default, 1U);
    }
    else if (auto_state->last_infrared_valid)
    {
        base_focal = auto_state->last_ir_focal;
        *ir_focus = auto_state->last_ir_focus > 0 ?
                        auto_state->last_ir_focus :
                        MAX(control_config->ir_focus_default, 1U);
    }
    else
    {
        *ir_focus = MAX(control_config->ir_focus_default, 1U);
    }

    if (sample->target_ratio < (control_config->zoom_target_ratio_min - control_config->zoom_deadband))
    {
        delta = control_config->ir_zoom_kp *
                (control_config->zoom_target_ratio_min - sample->target_ratio);
    }
    else if (sample->target_ratio > (control_config->zoom_target_ratio_max + control_config->zoom_deadband))
    {
        delta = -control_config->ir_zoom_kp *
                (sample->target_ratio - control_config->zoom_target_ratio_max);
    }

    delta = clamp_cuav_double(delta,
                              -control_config->ir_zoom_max_step,
                              control_config->ir_zoom_max_step);
    target_focal = base_focal + delta;
    target_focal = clamp_cuav_double(target_focal,
                                     control_config->ir_focal_min,
                                     control_config->ir_focal_max);

    *ir_focal = target_focal;
    return TRUE;
}

/**
 * @brief 通过GSignal向cuavcontrolsink元素发送控制信号
 * @param signal_name 信号名称（如send-servo-control）
 * @param payload GstStructure格式的指令载荷
 * @return 信号发送成功返回TRUE
 */
static gboolean
emit_cuav_control_signal(AppCtx *appCtx,
                         const gchar *signal_name,
                         GstStructure *payload)
{
    gboolean result = FALSE;
    GstElement *element = NULL;

    if (!appCtx || !signal_name || !payload)
        return FALSE;

    element = appCtx->pipeline.common_elements.cuav_control;
    if (!element)
        return FALSE;

    g_signal_emit_by_name(element, signal_name, payload, &result);
    return result;
}

/**
 * @brief 将最新的EO系统反馈状态通过信号转发给cuavcontrolsink元素
 * @param feedback_state 设备反馈状态快照
 * @return 成功返回TRUE
 */
static gboolean
update_cuav_eo_system_state(AppCtx *appCtx,
                            const CuavFeedbackState *feedback_state)
{
    GstStructure *payload = NULL;
    gboolean result = FALSE;

    if (!appCtx || !feedback_state || !feedback_state->valid)
        return FALSE;

    payload = gst_structure_new("cuav-eo-system-state",
                                "updated-at-us", G_TYPE_DOUBLE,
                                (gdouble)feedback_state->updated_at_us,
                                "st-loc-h", G_TYPE_DOUBLE,
                                feedback_state->st_loc_h,
                                "st-loc-v", G_TYPE_DOUBLE,
                                feedback_state->st_loc_v,
                                "pt-focal", G_TYPE_DOUBLE,
                                feedback_state->pt_focal,
                                "pt-focus", G_TYPE_INT,
                                (gint)feedback_state->pt_focus,
                                "ir-focal", G_TYPE_DOUBLE,
                                feedback_state->ir_focal,
                                "ir-focus", G_TYPE_INT,
                                (gint)feedback_state->ir_focus,
                                "sv-stat", G_TYPE_INT,
                                (gint)feedback_state->sv_stat,
                                "trk-dev", G_TYPE_INT,
                                (gint)feedback_state->trk_dev,
                                "pt-trk-link", G_TYPE_INT,
                                (gint)feedback_state->pt_trk_link,
                                "ir-trk-link", G_TYPE_INT,
                                (gint)feedback_state->ir_trk_link,
                                "trk-stat", G_TYPE_INT,
                                (gint)feedback_state->trk_stat,
                                NULL);

    result = emit_cuav_control_signal(appCtx, "update-eo-system-state", payload);
    gst_structure_free(payload);
    return result;
}

/**
 * @brief 发送云台伺服控制指令(完整版，可单独控制各轴使能)
 * @param dev_id 设备ID
 * @param dev_en 设备使能
 * @param ctrl_en 控制使能
 * @param mode_h 水平模式(0=绝对位置)
 * @param mode_v 垂直模式(0=绝对位置)
 * @param speed_h 水平速度
 * @param speed_v 垂直速度
 * @param loc_en_h 水平位置使能
 * @param loc_h 水平方位角
 * @param loc_en_v 垂直位置使能
 * @param loc_v 俯仰角
 * @return 发送成功返回TRUE
 */
static gboolean
send_cuav_servo_command_with_en(AppCtx *appCtx,
                                guint dev_id,
                                guint dev_en,
                                guint ctrl_en,
                                guint mode_h,
                                guint mode_v,
                                guint speed_h,
                                guint speed_v,
                                guint loc_en_h,
                                gdouble loc_h,
                                guint loc_en_v,
                                gdouble loc_v)
{
    GstStructure *payload = NULL;
    gboolean result = FALSE;

    payload = gst_structure_new("cuav-servo-control",
                                "dev-id", G_TYPE_INT, (gint)dev_id,
                                "dev-en", G_TYPE_INT, (gint)dev_en,
                                "ctrl-en", G_TYPE_INT, (gint)ctrl_en,
                                "mode-h", G_TYPE_INT, (gint)mode_h,
                                "mode-v", G_TYPE_INT, (gint)mode_v,
                                "speed-en-h", G_TYPE_INT, 1,
                                "speed-h", G_TYPE_INT, (gint)speed_h,
                                "speed-en-v", G_TYPE_INT, 1,
                                "speed-v", G_TYPE_INT, (gint)speed_v,
                                "loc-en-h", G_TYPE_INT, (gint)loc_en_h,
                                "loc-h", G_TYPE_DOUBLE, loc_h,
                                "loc-en-v", G_TYPE_INT, (gint)loc_en_v,
                                "loc-v", G_TYPE_DOUBLE, loc_v,
                                "offset-en", G_TYPE_INT, 0,
                                "offset-h", G_TYPE_INT, 0,
                                "offset-v", G_TYPE_INT, 0,
                                NULL);

    result = emit_cuav_control_signal(appCtx, "send-servo-control", payload);
    gst_structure_free(payload);
    return result;
}

/**
 * @brief 发送云台伺服控制指令（简化版，默认启用两个轴的位置控制）
 */
static gboolean
send_cuav_servo_command(AppCtx *appCtx,
                        guint dev_id,
                        guint dev_en,
                        guint ctrl_en,
                        guint mode_h,
                        guint mode_v,
                        guint speed_h,
                        guint speed_v,
                        gdouble loc_h,
                        gdouble loc_v)
{
    return send_cuav_servo_command_with_en(appCtx,
                                           dev_id,
                                           dev_en,
                                           ctrl_en,
                                           mode_h,
                                           mode_v,
                                           speed_h,
                                           speed_v,
                                           1,
                                           loc_h,
                                           1,
                                           loc_v);
}

/**
 * @brief 发送可见光控制指令（含焦距、对焦、对焦模式等完整参数）
 * @param pt_focal_en 焦距控制使能(0=停止,1=绝对值,3=放大,4=缩小)
 * @param pt_focal 焦距目标值
 * @param pt_focus_en 对焦使能
 * @param pt_focus 对焦目标值
 * @param pt_focus_mode 对焦模式
 * @param pt_zoom 变倍
 * @return 发送成功返回TRUE
 */
static gboolean
send_cuav_visible_light_command_with_en(AppCtx *appCtx,
                                        guint pt_focal_en,
                                        gdouble pt_focal,
                                        guint pt_focus_en,
                                        guint pt_focus,
                                        guint pt_focus_mode,
                                        guint pt_zoom)
{
    GstStructure *payload = NULL;
    gboolean result = FALSE;
    guint effective_pt_focus_mode = pt_focal_en == 0 ? 0 : pt_focus_mode;

    payload = gst_structure_new("cuav-visible-light-control",
                                "pt-dev-en", G_TYPE_INT, 1,
                                "pt-ctrl-en", G_TYPE_INT, 1,
                                "pt-fov-en", G_TYPE_INT, 0,
                                "pt-fov-h", G_TYPE_DOUBLE, 0.0,
                                "pt-fov-v", G_TYPE_DOUBLE, 0.0,
                                "pt-focal-en", G_TYPE_INT, (gint)pt_focal_en,
                                "pt-focal", G_TYPE_DOUBLE, pt_focal,
                                "pt-focus-en", G_TYPE_INT, (gint)pt_focus_en,
                                "pt-focus", G_TYPE_INT, (gint)pt_focus,
                                "pt-speed-en", G_TYPE_INT, 0,
                                "pt-focus-speed", G_TYPE_INT, 0,
                                "pt-bri-en", G_TYPE_INT, 0,
                                "pt-bri-ctrs", G_TYPE_INT, 0,
                                "pt-ctrs-en", G_TYPE_INT, 0,
                                "pt-ctrs", G_TYPE_INT, 0,
                                "pt-ofr-en", G_TYPE_INT, 0,
                                "pt-ofr", G_TYPE_INT, 0,
                                "pt-focus-mode", G_TYPE_INT, (gint)effective_pt_focus_mode,
                                "pt-zoom", G_TYPE_INT, (gint)pt_zoom,
                                NULL);

    result = emit_cuav_control_signal(appCtx, "send-visible-light-control", payload);
    gst_structure_free(payload);
    return result;
}

/**
 * @brief 发送可见光控制指令（简化版，默认focal_en=1即绝对焦距模式）
 */
static gboolean
send_cuav_visible_light_command(AppCtx *appCtx,
                                gdouble pt_focal,
                                guint pt_focus_en,
                                guint pt_focus,
                                guint pt_focus_mode,
                                guint pt_zoom)
{
    return send_cuav_visible_light_command_with_en(appCtx, 1, pt_focal,
                                                   pt_focus_en, pt_focus,
                                                   pt_focus_mode, pt_zoom);
}

/**
 * @brief 发送红外控制指令（焦距、对焦、对焦模式、变倍）
 * @param ir_focal 红外焦距目标值
 * @param ir_focus 红外对焦值
 * @param ir_focus_mode 红外对焦模式
 * @param ir_zoom 红外变倍
 * @return 发送成功返回TRUE
 */
static gboolean
send_cuav_infrared_command(AppCtx *appCtx,
                           gdouble ir_focal,
                           guint ir_focus,
                           guint ir_focus_mode,
                           guint ir_zoom)
{
    GstStructure *payload = NULL;
    gboolean result = FALSE;

    payload = gst_structure_new("cuav-infrared-control",
                                "ir-dev-en", G_TYPE_INT, 1,
                                "ir-ctrl-en", G_TYPE_INT, 1,
                                "ir-fov-en", G_TYPE_INT, 0,
                                "ir-fov-h", G_TYPE_DOUBLE, 0.0,
                                "ir-fov-v", G_TYPE_DOUBLE, 0.0,
                                "ir-focal-en", G_TYPE_INT, 1,
                                "ir-focal", G_TYPE_DOUBLE, ir_focal,
                                "ir-focus-en", G_TYPE_INT, 0,
                                "ir-focus", G_TYPE_INT, (gint)ir_focus,
                                "ir-speed-en", G_TYPE_INT, 0,
                                "ir-focus-speed", G_TYPE_INT, 0,
                                "ir-bri-en", G_TYPE_INT, 0,
                                "ir-bri-ctrs", G_TYPE_INT, 0,
                                "ir-ctrs-en", G_TYPE_INT, 0,
                                "ir-ctrs", G_TYPE_INT, 0,
                                "ir-focus-mode", G_TYPE_INT, (gint)ir_focus_mode,
                                "ir-zoom", G_TYPE_INT, (gint)ir_zoom,
                                NULL);

    result = emit_cuav_control_signal(appCtx, "send-infrared-control", payload);
    gst_structure_free(payload);
    return result;
}

/**
 * @brief 创建并初始化cuavcontrolsink GStreamer元素，设置组播网络参数
 * 同时重置启动预置位和角点循环状态机
 * @param config 全局配置
 * @param pipeline 管道结构
 * @return 成功返回TRUE
 */
static gboolean
create_cuav_control_element(NvDsConfig *config, NvDsPipeline *pipeline)
{
    GstElement *cuav_control = NULL;
    NvDsSinkSubBinConfig *sink_config = NULL;
    NvDsCuavControlConfig *control_config = NULL;

    sink_config = find_cuav_control_sink_config(config);
    if (!sink_config)
        return TRUE;

    control_config = &sink_config->cuav_control_config;

    cuav_control = gst_element_factory_make(NVDS_ELEM_CUAVCONTROL_ELEMENT,
                                            "cuav_control");
    if (!cuav_control)
    {
        NVGSTDS_ERR_MSG_V("Failed to create element '%s'. Build/install the plugin in src/gst-cuavcontrolsink first.",
                          NVDS_ELEM_CUAVCONTROL_ELEMENT);
        return FALSE;
    }

    if (control_config->multicast_ip)
        g_object_set(G_OBJECT(cuav_control), "multicast-ip",
                     control_config->multicast_ip, NULL);
    if (control_config->port)
        g_object_set(G_OBJECT(cuav_control), "port",
                     control_config->port, NULL);
    if (control_config->iface)
        g_object_set(G_OBJECT(cuav_control), "iface",
                     control_config->iface, NULL);
    g_object_set(G_OBJECT(cuav_control),
                 "ttl", control_config->ttl,
                 "compat-cmd-wrapper", control_config->compat_cmd_wrapper,
                 "debug", control_config->debug,
                 "print-upstream-state", control_config->print_upstream_state,
                 "tx-sys-id", control_config->tx_sys_id,
                 "tx-dev-type", control_config->tx_dev_type,
                 "tx-dev-id", control_config->tx_dev_id,
                 "tx-subdev-id", control_config->tx_subdev_id,
                 "rx-sys-id", control_config->rx_sys_id,
                 "rx-dev-type", control_config->rx_dev_type,
                 "rx-dev-id", control_config->rx_dev_id,
                 "rx-subdev-id", control_config->rx_subdev_id,
                 NULL);

    gst_bin_add(GST_BIN(pipeline->pipeline), cuav_control);
    pipeline->common_elements.cuav_control = cuav_control;

    g_print("[cuav][control] enabled via sink type=8 target=%s:%u iface=%s compat=%d startup-test=%d auto-track=%d visible=%d infrared=%d servo-dev-id=%u\n",
            control_config->multicast_ip ?
                control_config->multicast_ip : "(default)",
            control_config->port,
            control_config->iface ?
                control_config->iface : "(default)",
            control_config->compat_cmd_wrapper,
            control_config->send_test_on_startup,
            control_config->auto_track_enable,
            control_config->visible_light_control_enable,
            control_config->infrared_control_enable,
            control_config->servo_dev_id);

    if (pipeline->appCtx)
    {
        g_mutex_lock(&pipeline->appCtx->cuav_control_lock);
        cuav_reset_startup_preset_state(&pipeline->appCtx->cuav_startup_preset_state);
        cuav_reset_corner_zoom_cycle_state(&pipeline->appCtx->cuav_corner_zoom_cycle_state,
                                           control_config);
        g_mutex_unlock(&pipeline->appCtx->cuav_control_lock);
    }

    if (control_config->auto_track_enable && !cuav_is_test_target(control_config))
    {
        g_printerr("[cuav][control][warn] auto-track is enabled with non-test target %s:%u\n",
                   control_config->multicast_ip ? control_config->multicast_ip : "(null)",
                   control_config->port);
    }
    if (control_config->corner_zoom_cycle_enable)
    {
        g_print("[cuav][corner-zoom] enabled repeat=%u servo=%u corner-cycle=%u offset=(%.1f,%.1f) dwell=%u ms zoom-in=%u ms zoom-out=%u ms speed=%u\n",
                control_config->sequence_repeat_count,
                control_config->corner_servo_enable ? 1U : 0U,
                control_config->corner_cycle_count,
                control_config->corner_offset_h_deg,
                control_config->corner_offset_v_deg,
                control_config->corner_dwell_ms,
                control_config->zoom_in_duration_ms,
                control_config->zoom_out_duration_ms,
                control_config->corner_servo_speed);
        g_print("[cuav][corner-zoom] home target loc=(%.1f,%.1f) preset_focus_en=%u\n",
                control_config->corner_home_loc_h_deg,
                control_config->corner_home_loc_v_deg,
                control_config->corner_home_pt_focus != G_MAXUINT ? 1U : 0U);
    }
    return TRUE;
}

/**
 * @brief 发送云台伺服测试指令（固定参数: loc_h=180, loc_v=10, speed=20）
 */
static gboolean
send_cuav_servo_test_message(AppCtx *appCtx)
{
    gboolean result = FALSE;
    NvDsSinkSubBinConfig *sink_config = NULL;
    guint servo_dev_id = 2;

    if (!appCtx)
        return FALSE;

    sink_config = find_cuav_control_sink_config(&appCtx->config);
    if (sink_config)
        servo_dev_id = sink_config->cuav_control_config.servo_dev_id;

    result = send_cuav_servo_command(appCtx, servo_dev_id, 1, 1, 0, 0, 20, 20, 180.0, 10.0);

    g_print("[cuav][control-test] servo test send result=%d\n", result);
    return result;
}

/**
 * @brief 发送可见光测试指令（固定参数: focal=500, focus=100）
 */
static gboolean
send_cuav_visible_light_test_message(AppCtx *appCtx)
{
    gboolean result = FALSE;

    if (!appCtx)
        return FALSE;

    result = send_cuav_visible_light_command(appCtx, 500.0, 1, 100, 1, 0);

    g_print("[cuav][control-test] visible-light test send result=%d\n", result);
    return result;
}

/**
 * @brief 发送红外测试指令（固定参数: focal=900, focus=5）
 */
static gboolean
send_cuav_infrared_test_message(AppCtx *appCtx)
{
    gboolean result = FALSE;

    if (!appCtx)
        return FALSE;

    result = send_cuav_infrared_command(appCtx, 900.0, 5, 0, 0);

    g_print("[cuav][control-test] infrared test send result=%d\n", result);
    return result;
}

/**
 * @brief 依次发送云台、可见光、红外测试指令（仅在启动测试模式且非自动跟踪时生效）
 * @return 全部发送成功返回TRUE
 */
gboolean
send_cuav_test_messages(AppCtx *appCtx)
{
    gboolean servo_ok = FALSE;
    gboolean visible_ok = FALSE;
    gboolean infrared_ok = TRUE;
    NvDsSinkSubBinConfig *sink_config = NULL;

    if (!appCtx)
        return FALSE;

    sink_config = find_cuav_control_sink_config(&appCtx->config);
    if (!sink_config || !sink_config->cuav_control_config.send_test_on_startup)
        return FALSE;

    if (sink_config->cuav_control_config.auto_track_enable)
        return FALSE;
    if (sink_config->cuav_control_config.corner_zoom_cycle_enable)
        return FALSE;

    servo_ok = send_cuav_servo_test_message(appCtx);
    if (sink_config->cuav_control_config.visible_light_control_enable)
        visible_ok = send_cuav_visible_light_test_message(appCtx);
    else
        visible_ok = TRUE;

    if (sink_config->cuav_control_config.infrared_control_enable)
        infrared_ok = send_cuav_infrared_test_message(appCtx);

    return servo_ok && visible_ok && infrared_ok;
}

/**
 * @brief 判断目标元数据是否为有效跟踪对象（有跟踪ID且bbox尺寸大于1像素）
 */
static gboolean
cuav_is_valid_tracked_object(const NvDsObjectMeta *obj_meta)
{
    return obj_meta &&
           obj_meta->object_id != UNTRACKED_OBJECT_ID &&
           obj_meta->rect_params.width > 1.0f &&
           obj_meta->rect_params.height > 1.0f;
}

/**
 * @brief 获取目标置信度分数，优先使用跟踪器置信度，其次使用检测器置信度
 */
static gdouble
cuav_get_object_score(const NvDsObjectMeta *obj_meta)
{
    if (!obj_meta)
        return -1.0;

    if (obj_meta->tracker_confidence > 0.0f)
        return obj_meta->tracker_confidence;

    return obj_meta->confidence;
}

/**
 * @brief 从帧元数据中选择跟踪控制目标
 * 优先返回与locked_object_id匹配的已锁定目标，否则选择置信度最高的目标
 * @param frame_meta 帧元数据
 * @param locked_object_id 已锁定的目标ID，0表示无锁定
 * @return 目标对象元数据指针，无有效目标返回NULL
 */
static NvDsObjectMeta *
cuav_select_control_target(NvDsFrameMeta *frame_meta,
                           guint64 locked_object_id)
{
    NvDsObjectMeta *best = NULL;
    gdouble best_score = -G_MAXDOUBLE;

    if (!frame_meta)
        return NULL;

    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
        gdouble score = 0.0;

        if (!cuav_is_valid_tracked_object(obj_meta))
            continue;

        if (locked_object_id != 0 && obj_meta->object_id == locked_object_id)
            return obj_meta;

        score = cuav_get_object_score(obj_meta);
        if (!best || score > best_score)
        {
            best = obj_meta;
            best_score = score;
        }
    }

    return best;
}

/**
 * @brief 生成正弦波模拟跟踪采样数据（用于无真实目标时的闭环控制测试）
 * err_x/err_y按正弦振荡，target_ratio按低频正弦在[min,max]之间变化
 * @param control_config 控制配置（含模拟参数）
 * @param now_us 当前时间
 * @param[out] sample 填充的模拟采样数据
 */
static void
cuav_fill_simulated_sample(const NvDsCuavControlConfig *control_config,
                           gint64 now_us,
                           CuavTrackSample *sample)
{
    gdouble period_sec = 0.0;
    gdouble phase = 0.0;
    gdouble ratio_phase = 0.0;
    gdouble ratio_mid = 0.0;
    gdouble ratio_amp = 0.0;

    if (!control_config || !sample)
        return;

    period_sec = MAX(control_config->simulate_target_period_ms, 1000U) / 1000.0;
    phase = (2.0 * G_PI * ((now_us / 1000000.0) / period_sec));
    ratio_phase = phase * 0.5;
    ratio_mid = (control_config->simulate_target_ratio_min +
                 control_config->simulate_target_ratio_max) * 0.5;
    ratio_amp = (control_config->simulate_target_ratio_max -
                 control_config->simulate_target_ratio_min) * 0.5;

    memset(sample, 0, sizeof(*sample));
    sample->valid = TRUE;
    sample->object_id = 0;
    sample->sample_time_us = now_us;
    sample->err_x = control_config->simulate_target_amplitude_x * sin(phase);
    sample->err_y = control_config->simulate_target_amplitude_y * sin(phase * 0.7);
    sample->target_ratio = ratio_mid + (ratio_amp * sin(ratio_phase));
    sample->center_x = sample->err_x;
    sample->center_y = sample->err_y;
    sample->width = sample->target_ratio;
    sample->height = sample->target_ratio;
}

/**
 * @brief 处理角点变焦循环测试状态机
 * 流程: 回预置位 → 四角循环运动 → 回预置位 → 可见光预置 → 拉到最大焦距 → 拉到最小焦距 → 重复
 * @param appCtx 应用上下文
 * @param control_config 控制配置
 * @param now_us 当前单调时钟时间（微秒）
 * @return 正常处理返回TRUE
 */
static gboolean
process_cuav_corner_zoom_cycle(AppCtx *appCtx,
                               const NvDsCuavControlConfig *control_config,
                               gint64 now_us)
{
    CuavCornerZoomCycleState state_snapshot;
    CuavFeedbackState feedback_snapshot;
    gboolean feedback_fresh = FALSE;
    gboolean sent = FALSE;
    CuavStartupPresetState startup_snapshot;
    guint repeat_limit = 1;
    guint corner_cycle_limit = 1;
    guint corner_count = 4;
    guint corner_index = 0;
    guint corner_cycle_index = 0;
    guint corner_servo_speed = 0;
    gdouble corner_offset_h = 0.0;
    gdouble corner_offset_v = 0.0;
    gdouble home_loc_h = 180.0;
    gdouble home_loc_v = 0.0;
    gdouble base_loc_h = 180.0;
    gdouble base_loc_v = 0.0;
    gdouble target_loc_h = 180.0;
    gdouble target_loc_v = 0.0;
    guint visible_focus = 100;
    gdouble visible_focal_target = 0.0;
    gboolean home_visible_preset_valid = FALSE;
    gboolean home_visible_focus_valid = FALSE;
    gboolean home_loc_configured = FALSE;
    guint home_visible_focus = 100;
    guint min_gap_ms = 1;
    gint64 min_gap_us = 0;
    gint64 corner_dwell_us = 0;
    gint64 home_settle_timeout_us = 0;
    gint64 zoom_in_hold_us = 0;
    gint64 zoom_out_hold_us = 0;
    gboolean visible_enabled = FALSE;

    if (!appCtx || !control_config || !appCtx->pipeline.common_elements.cuav_control)
        return FALSE;

    visible_enabled = cuav_visible_control_enabled(control_config);
    min_gap_ms = MAX(control_config->control_period_ms, 1U);
    min_gap_us = ((gint64)min_gap_ms) * 1000;
    corner_dwell_us = ((gint64)MAX(control_config->corner_dwell_ms, 1U)) * 1000;
    zoom_in_hold_us = ((gint64)MAX(control_config->zoom_in_duration_ms, 1U)) * 1000;
    zoom_out_hold_us = ((gint64)MAX(control_config->zoom_out_duration_ms, 1U)) * 1000;
    repeat_limit = MAX(control_config->sequence_repeat_count, 1U);
    corner_cycle_limit = MAX(control_config->corner_cycle_count, 1U);
    corner_servo_speed = clamp_cuav_uint(MAX(control_config->corner_servo_speed, 1U),
                                         1, 200);
    corner_offset_h = fabs(control_config->corner_offset_h_deg);
    corner_offset_v = fabs(control_config->corner_offset_v_deg);
    home_loc_h = control_config->corner_home_loc_h_deg;
    home_loc_v = control_config->corner_home_loc_v_deg;
    home_settle_timeout_us = ((gint64)MAX(control_config->corner_dwell_ms,
                                          control_config->state_stale_timeout_ms)) * 1000;
    home_visible_focus_valid = control_config->corner_home_pt_focus != G_MAXUINT;
    home_loc_configured = !isnan(control_config->corner_home_loc_h_deg) &&
                          !isnan(control_config->corner_home_loc_v_deg);
    home_visible_preset_valid = visible_enabled && home_visible_focus_valid;
    if (home_visible_focus_valid)
        home_visible_focus = control_config->corner_home_pt_focus;

    g_mutex_lock(&appCtx->cuav_control_lock);
    if (!appCtx->cuav_corner_zoom_cycle_state.initialized)
    {
        cuav_reset_corner_zoom_cycle_state(&appCtx->cuav_corner_zoom_cycle_state,
                                           control_config);
        appCtx->cuav_corner_zoom_cycle_state.initialized = TRUE;
        startup_snapshot = appCtx->cuav_startup_preset_state;
        if (!control_config->corner_servo_enable)
        {
            appCtx->cuav_corner_zoom_cycle_state.return_home_before_zoom = TRUE;
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN;
        }
        else
        {
            appCtx->cuav_corner_zoom_cycle_state.phase =
                startup_snapshot.servo_applied ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
        }
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = 0;
    }
    feedback_snapshot = appCtx->cuav_feedback_state;
    state_snapshot = appCtx->cuav_corner_zoom_cycle_state;
    startup_snapshot = appCtx->cuav_startup_preset_state;
    g_mutex_unlock(&appCtx->cuav_control_lock);

    feedback_fresh = cuav_feedback_is_fresh(&feedback_snapshot,
                                            control_config->state_stale_timeout_ms);

    if (state_snapshot.phase == CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE)
    {
        if (!state_snapshot.final_logged)
        {
            g_print("[cuav][corner-zoom] complete repeat=%u/%u\n",
                    MIN(state_snapshot.outer_repeat_index + 1, repeat_limit),
                    repeat_limit);
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.final_logged = TRUE;
            g_mutex_unlock(&appCtx->cuav_control_lock);
        }
        return TRUE;
    }

    switch (state_snapshot.phase)
    {
    case CUAV_CORNER_ZOOM_CYCLE_PHASE_IDLE:
        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO:
        if (!control_config->corner_servo_enable)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        if (!state_snapshot.home_target_valid)
        {
            if (!cuav_corner_zoom_cycle_resolve_home_target(control_config,
                                                            &feedback_snapshot,
                                                            feedback_fresh,
                                                            &state_snapshot,
                                                            &home_loc_h,
                                                            &home_loc_v))
                return FALSE;

            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.home_target_valid = TRUE;
            appCtx->cuav_corner_zoom_cycle_state.home_loc_h = home_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.home_loc_v = home_loc_v;
            appCtx->cuav_corner_zoom_cycle_state.base_loc_h = home_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.base_loc_v = home_loc_v;
            state_snapshot.home_target_valid = TRUE;
            state_snapshot.home_loc_h = home_loc_h;
            state_snapshot.home_loc_v = home_loc_v;
            state_snapshot.base_loc_h = home_loc_h;
            state_snapshot.base_loc_v = home_loc_v;
            appCtx->cuav_corner_zoom_cycle_state.corner_cycle_index = 0;
            appCtx->cuav_corner_zoom_cycle_state.corner_index = 0;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] home start repeat=%u/%u home=(%.2f,%.2f) "
                        "corner_cycle=%u offset=(%.1f,%.1f) preset_focus_en=%u\n",
                        state_snapshot.outer_repeat_index +
                            (state_snapshot.increment_repeat_after_home ? 1 : 0) + 1,
                        repeat_limit,
                        home_loc_h, home_loc_v,
                        corner_cycle_limit,
                        corner_offset_h, corner_offset_v,
                        home_visible_focus_valid ? 1 : 0);
            }
        }
        else
        {
            home_loc_h = state_snapshot.home_loc_h;
            home_loc_v = state_snapshot.home_loc_v;
        }
        sent = send_cuav_servo_command_with_en(appCtx,
                                               control_config->servo_dev_id,
                                               1, 1, 0, 0,
                                               corner_servo_speed,
                                               corner_servo_speed,
                                               1, home_loc_h,
                                               1, home_loc_v);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_loc_h = home_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.last_loc_v = home_loc_v;
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_HOME_SERVO;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] send home repeat=%u/%u loc=(%.2f,%.2f) speed=%u\n",
                        state_snapshot.outer_repeat_index +
                            (state_snapshot.increment_repeat_after_home ? 1 : 0) + 1,
                        repeat_limit,
                        home_loc_h, home_loc_v,
                        corner_servo_speed);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_HOME_SERVO:
        if (feedback_fresh && cuav_corner_zoom_cycle_home_reached(&feedback_snapshot,
                                                                  &state_snapshot,
                                                                  control_config))
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            if (state_snapshot.increment_repeat_after_home)
            {
                appCtx->cuav_corner_zoom_cycle_state.outer_repeat_index++;
                appCtx->cuav_corner_zoom_cycle_state.increment_repeat_after_home = FALSE;
            }
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            if (state_snapshot.return_home_before_zoom)
            {
                appCtx->cuav_corner_zoom_cycle_state.phase =
                    home_visible_preset_valid ?
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN;
            }
            else if (state_snapshot.resume_cycle_after_home)
            {
                appCtx->cuav_corner_zoom_cycle_state.phase =
                    home_visible_preset_valid ?
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
            }
            else
            {
                appCtx->cuav_corner_zoom_cycle_state.phase =
                    home_visible_preset_valid ?
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE;
            }
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }
        if ((now_us - state_snapshot.phase_started_us) < home_settle_timeout_us)
            return TRUE;

        if (control_config->debug && !cuav_corner_zoom_cycle_home_reached(&feedback_snapshot,
                                                                          &state_snapshot,
                                                                          control_config))
        {
            if (home_loc_configured)
                g_print("[cuav][corner-zoom][warn] home settle timeout at preset, continue to next stage\n");
            else
                g_print("[cuav][corner-zoom][warn] home settle timeout, continue to next stage\n");
        }

        if (state_snapshot.return_home_before_zoom)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        g_mutex_lock(&appCtx->cuav_control_lock);
        if (state_snapshot.increment_repeat_after_home)
        {
            appCtx->cuav_corner_zoom_cycle_state.outer_repeat_index++;
            appCtx->cuav_corner_zoom_cycle_state.increment_repeat_after_home = FALSE;
        }
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        if (state_snapshot.return_home_before_zoom)
        {
            appCtx->cuav_corner_zoom_cycle_state.phase =
                home_visible_preset_valid ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN;
        }
        else if (state_snapshot.resume_cycle_after_home)
        {
            appCtx->cuav_corner_zoom_cycle_state.phase =
                home_visible_preset_valid ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
        }
        else
        {
            appCtx->cuav_corner_zoom_cycle_state.phase =
                home_visible_preset_valid ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE;
        }
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        if (!home_visible_preset_valid)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.phase =
                state_snapshot.return_home_before_zoom ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN :
                    (state_snapshot.resume_cycle_after_home ?
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER :
                        CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE);
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       0,
                                                       0.0,
                                                       home_visible_focus_valid ? 1 : 0,
                                                       home_visible_focus,
                                                       1,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_HOME_VISIBLE_PRESET;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] home preset repeat=%u/%u focus_en=%u focus=%u\n",
                        state_snapshot.outer_repeat_index +
                            (state_snapshot.increment_repeat_after_home ? 1 : 0) + 1,
                        repeat_limit,
                        home_visible_focus_valid ? 1 : 0,
                        home_visible_focus);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_HOME_VISIBLE_PRESET:
        if ((now_us - state_snapshot.phase_started_us) < home_settle_timeout_us)
            return TRUE;

        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_corner_zoom_cycle_state.phase =
            state_snapshot.return_home_before_zoom ?
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN :
                (state_snapshot.resume_cycle_after_home ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE);
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        if (!state_snapshot.home_target_valid)
        {
            if (!cuav_corner_zoom_cycle_resolve_home_target(control_config,
                                                            &feedback_snapshot,
                                                            feedback_fresh,
                                                            &state_snapshot,
                                                            &home_loc_h,
                                                            &home_loc_v))
                return FALSE;

            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.home_target_valid = TRUE;
            appCtx->cuav_corner_zoom_cycle_state.home_loc_h = home_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.home_loc_v = home_loc_v;
            appCtx->cuav_corner_zoom_cycle_state.base_loc_h = home_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.base_loc_v = home_loc_v;
            state_snapshot.home_target_valid = TRUE;
            state_snapshot.home_loc_h = home_loc_h;
            state_snapshot.home_loc_v = home_loc_v;
            state_snapshot.base_loc_h = home_loc_h;
            state_snapshot.base_loc_v = home_loc_v;
            g_mutex_unlock(&appCtx->cuav_control_lock);
        }
        else
        {
            home_loc_h = state_snapshot.home_loc_h;
            home_loc_v = state_snapshot.home_loc_v;
        }

        visible_focus = home_visible_focus;

        base_loc_h = state_snapshot.base_loc_h;
        base_loc_v = state_snapshot.base_loc_v;
        corner_index = state_snapshot.corner_index;
        corner_cycle_index = state_snapshot.corner_cycle_index;
        cuav_corner_zoom_cycle_compute_target(base_loc_h, base_loc_v,
                                              corner_offset_h, corner_offset_v,
                                              corner_index,
                                              &target_loc_h, &target_loc_v);
        sent = send_cuav_servo_command_with_en(appCtx,
                                               control_config->servo_dev_id,
                                               1, 1, 0, 0,
                                               corner_servo_speed,
                                               corner_servo_speed,
                                               1, target_loc_h,
                                               1, target_loc_v);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_loc_h = target_loc_h;
            appCtx->cuav_corner_zoom_cycle_state.last_loc_v = target_loc_v;
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_CORNER;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] send corner repeat=%u/%u cycle=%u/%u corner=%s loc=(%.2f,%.2f) speed=%u\n",
                        state_snapshot.outer_repeat_index + 1,
                        repeat_limit,
                        corner_cycle_index + 1,
                        corner_cycle_limit,
                        cuav_corner_zoom_cycle_corner_name(corner_index),
                        target_loc_h, target_loc_v,
                        corner_servo_speed);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_CORNER:
        if ((now_us - state_snapshot.phase_started_us) < corner_dwell_us)
            return TRUE;

        g_mutex_lock(&appCtx->cuav_control_lock);
        if ((state_snapshot.corner_index + 1) < corner_count)
        {
            appCtx->cuav_corner_zoom_cycle_state.corner_index++;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
        }
        else if ((state_snapshot.corner_cycle_index + 1) < corner_cycle_limit)
        {
            appCtx->cuav_corner_zoom_cycle_state.corner_cycle_index++;
            appCtx->cuav_corner_zoom_cycle_state.corner_index = 0;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
        }
        else
        {
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_SERVO_STOP;
        }
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_SERVO_STOP:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        sent = send_cuav_servo_command_with_en(appCtx,
                                               control_config->servo_dev_id,
                                               1, 1, 0, 0,
                                               corner_servo_speed,
                                               corner_servo_speed,
                                               0, state_snapshot.last_loc_h,
                                               0, state_snapshot.last_loc_v);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.corner_cycle_index = 0;
            appCtx->cuav_corner_zoom_cycle_state.corner_index = 0;
            appCtx->cuav_corner_zoom_cycle_state.return_home_before_zoom = visible_enabled;
            appCtx->cuav_corner_zoom_cycle_state.resume_cycle_after_home =
                !visible_enabled && ((state_snapshot.outer_repeat_index + 1) < repeat_limit);
            appCtx->cuav_corner_zoom_cycle_state.increment_repeat_after_home =
                appCtx->cuav_corner_zoom_cycle_state.resume_cycle_after_home;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] servo stop repeat=%u/%u loc=(%.2f,%.2f)\n",
                        state_snapshot.outer_repeat_index + 1,
                        repeat_limit,
                        state_snapshot.last_loc_h,
                        state_snapshot.last_loc_v);
                if (visible_enabled)
                    g_print("[cuav][corner-zoom] return to home preset before zoom\n");
                else
                    g_print("[cuav][corner-zoom][warn] visible-light-control-enable=0, skip zoom phases\n");
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        visible_focal_target = !isnan(control_config->corner_zoom_in_focal) ?
            control_config->corner_zoom_in_focal :
            clamp_cuav_double(control_config->pt_focal_max,
                              control_config->pt_focal_min,
                              control_config->pt_focal_max);
        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       1,
                                                       visible_focal_target,
                                                       0,
                                                       visible_focus,
                                                       2,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_ZOOM_IN;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] zoom-in absolute repeat=%u/%u target_focal=%.1f current_focal=%.1f wait=%u ms\n",
                        state_snapshot.outer_repeat_index + 1,
                        repeat_limit,
                        visible_focal_target,
                        feedback_snapshot.pt_focal,
                        control_config->zoom_in_duration_ms);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_ZOOM_IN:
        if ((now_us - state_snapshot.phase_started_us) < zoom_in_hold_us)
            return TRUE;

        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_OUT;
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_OUT:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        visible_focal_target = !isnan(control_config->corner_zoom_out_focal) ?
            control_config->corner_zoom_out_focal :
            clamp_cuav_double(control_config->pt_focal_min,
                              control_config->pt_focal_min,
                              control_config->pt_focal_max);
        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       1,
                                                       visible_focal_target,
                                                       0,
                                                       visible_focus,
                                                       2,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.last_command_sent_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_ZOOM_OUT;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-zoom] zoom-out absolute repeat=%u/%u target_focal=%.1f current_focal=%.1f wait=%u ms\n",
                        state_snapshot.outer_repeat_index + 1,
                        repeat_limit,
                        visible_focal_target,
                        feedback_snapshot.pt_focal,
                        control_config->zoom_out_duration_ms);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_HOLD_ZOOM_OUT:
        if ((now_us - state_snapshot.phase_started_us) < zoom_out_hold_us)
            return TRUE;

        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        appCtx->cuav_corner_zoom_cycle_state.corner_cycle_index = 0;
        appCtx->cuav_corner_zoom_cycle_state.corner_index = 0;
        appCtx->cuav_corner_zoom_cycle_state.return_home_before_zoom = FALSE;
        if (control_config->corner_servo_enable)
        {
            appCtx->cuav_corner_zoom_cycle_state.resume_cycle_after_home =
                (state_snapshot.outer_repeat_index + 1) < repeat_limit;
            appCtx->cuav_corner_zoom_cycle_state.increment_repeat_after_home =
                appCtx->cuav_corner_zoom_cycle_state.resume_cycle_after_home;
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
        }
        else if ((state_snapshot.outer_repeat_index + 1) < repeat_limit)
        {
            appCtx->cuav_corner_zoom_cycle_state.outer_repeat_index++;
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_ZOOM_IN;
        }
        else
        {
            appCtx->cuav_corner_zoom_cycle_state.phase =
                CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE;
        }
        g_mutex_unlock(&appCtx->cuav_control_lock);

        if (control_config->debug)
        {
            g_print("[cuav][corner-zoom] zoom-out hold done repeat=%u/%u current_focal=%.1f next=%s\n",
                    state_snapshot.outer_repeat_index + 1,
                    repeat_limit,
                    feedback_snapshot.pt_focal,
                    (state_snapshot.outer_repeat_index + 1) < repeat_limit ?
                        "home+repeat" : "home+complete");
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE:
    default:
        g_mutex_lock(&appCtx->cuav_control_lock);
        if (!appCtx->cuav_corner_zoom_cycle_state.final_logged)
        {
            appCtx->cuav_corner_zoom_cycle_state.final_logged = TRUE;
            g_print("[cuav][corner-zoom] complete repeat=%u/%u\n",
                    MIN(appCtx->cuav_corner_zoom_cycle_state.outer_repeat_index + 1,
                        repeat_limit),
                    repeat_limit);
        }
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;
    }
}

/**
 * @brief 自动跟踪控制主入口（每帧调用）
 * 执行优先级: 启动预置位 → 角点循环 → 自动跟踪 → 模拟目标
 * 自动跟踪流程: 选择目标 → 计算采样 → 速度估计 → 计算云台/可见光/红外指令 → 发送
 * @param appCtx 应用上下文
 * @param batch_meta 当前帧的批量元数据
 */
void
process_cuav_auto_control(AppCtx *appCtx, NvDsBatchMeta *batch_meta)
{
    NvDsSinkSubBinConfig *sink_config = NULL;
    NvDsCuavControlConfig *control_config = NULL;
    NvDsFrameMeta *selected_frame = NULL;
    NvDsObjectMeta *target_obj = NULL;
    guint64 locked_object_id = 0;
    gint64 now_us = 0;
    gint64 hold_deadline_us = 0;
    gint64 lost_zoom_hold_us = 0;
    gint64 lost_zoom_elapsed_us = 0;
    gboolean startup_preset_required = FALSE;
    CuavTrackSample sample;
    CuavFeedbackState feedback_snapshot;
    CuavAutoControlState state_snapshot;
    gdouble vel_x = 0.0;
    gdouble vel_y = 0.0;
    gboolean should_send_servo = FALSE;
    gboolean should_send_visible = FALSE;
    gboolean should_send_infrared = FALSE;
    gboolean visible_cmd_changed = FALSE;
    gboolean motion_spacing_ok = FALSE;
    gboolean motion_cmd_sent = FALSE;
    gboolean had_tracking_before_reset = FALSE;
    gboolean lost_zoom_active = FALSE;
    gboolean lost_zoom_stop_due = FALSE;
    gdouble loc_h = 0.0;
    gdouble loc_v = 0.0;
    guint speed_h = 0;
    guint speed_v = 0;
    guint pt_focal_en = 0;
    gdouble pt_focal = 0.0;
    guint pt_focus = 100;
    gdouble ir_focal = 0.0;
    guint ir_focus = 5;
    gboolean visible_stop_due = FALSE;
    gdouble offset_px_x = 0.0;
    gdouble offset_px_y = 0.0;
    gdouble deadband_px_x = 0.0;
    gdouble deadband_px_y = 0.0;
    gint control_frame_width = 0;
    gint control_frame_height = 0;
    gboolean servo_sent = FALSE;
    gboolean visible_sent = FALSE;
    gboolean infrared_sent = FALSE;
    gboolean debug_enabled = FALSE;

    if (!appCtx || !batch_meta)
        return;

    sink_config = find_cuav_control_sink_config(&appCtx->config);
    if (!sink_config)
        return;

    control_config = &sink_config->cuav_control_config;
    debug_enabled = control_config->debug;
    now_us = g_get_monotonic_time();
    visible_stop_due = FALSE;
    startup_preset_required = cuav_startup_preset_has_home_target(control_config) ||
                              (cuav_visible_control_enabled(control_config) &&
                               cuav_startup_preset_has_visible_preset(control_config));
    if (startup_preset_required)
    {
        process_cuav_startup_preset(appCtx, control_config, now_us);
        g_mutex_lock(&appCtx->cuav_control_lock);
        if (cuav_startup_preset_in_progress(&appCtx->cuav_startup_preset_state))
        {
            if (debug_enabled)
            {
                g_print("[cuav][control][auto] startup preset not complete, phase=%d auto-track frozen\n",
                        appCtx->cuav_startup_preset_state.phase);
            }
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return;
        }
        g_mutex_unlock(&appCtx->cuav_control_lock);
    }
    if (control_config->corner_zoom_cycle_enable)
    {
        if (debug_enabled)
        {
            g_print("[cuav][control][auto] corner zoom cycle enabled, auto-track skipped\n");
        }
        process_cuav_corner_zoom_cycle(appCtx, control_config, now_us);
        return;
    }
    if (!control_config->auto_track_enable ||
        !appCtx->pipeline.common_elements.cuav_control)
    {
        if (debug_enabled)
        {
            g_print("[cuav][control][auto] auto-track disabled or control sink missing (auto=%d cuav_control=%p)\n",
                    control_config->auto_track_enable,
                    appCtx->pipeline.common_elements.cuav_control);
        }
        return;
    }

    g_mutex_lock(&appCtx->cuav_control_lock);
    appCtx->cuav_auto_control_state.control_source_id =
        control_config->control_source_id;
    locked_object_id = appCtx->cuav_auto_control_state.has_lock ?
                       appCtx->cuav_auto_control_state.locked_object_id : 0;
    g_mutex_unlock(&appCtx->cuav_control_lock);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (frame_meta && frame_meta->source_id == control_config->control_source_id)
        {
            selected_frame = frame_meta;
            break;
        }
    }

    if (selected_frame)
    {
        control_frame_width = selected_frame->pipeline_width > 0 ?
            selected_frame->pipeline_width :
            (appCtx->config.streammux_config.pipeline_width > 0 ?
             appCtx->config.streammux_config.pipeline_width :
             selected_frame->source_frame_width);
        control_frame_height = selected_frame->pipeline_height > 0 ?
            selected_frame->pipeline_height :
            (appCtx->config.streammux_config.pipeline_height > 0 ?
             appCtx->config.streammux_config.pipeline_height :
             selected_frame->source_frame_height);
        target_obj = cuav_select_control_target(selected_frame, locked_object_id);
        if (debug_enabled)
        {
            g_print("[cuav][control][auto] frame source=%u source_size=%dx%d ctrl_size=%dx%d lock=%" G_GUINT64_FORMAT
                    " target_candidates_scan=%s\n",
                    selected_frame->source_id,
                    selected_frame->source_frame_width,
                    selected_frame->source_frame_height,
                    control_frame_width,
                    control_frame_height,
                    locked_object_id,
                    target_obj ? "matched" : "none");
        }
    }
    else if (debug_enabled)
    {
        g_print("[cuav][control][auto] source=%u frame not found in current batch\n",
                control_config->control_source_id);
    }

    if (!selected_frame || !target_obj)
    {
        if (debug_enabled)
        {
            g_print("[cuav][control][auto] source=%u no valid tracked target, simulate=%d\n",
                    control_config->control_source_id,
                    control_config->simulate_target_enable);
        }
        if (control_config->simulate_target_enable)
        {
            cuav_fill_simulated_sample(control_config, now_us, &sample);
            sample.object_id = 0;

            g_mutex_lock(&appCtx->cuav_control_lock);
            if (!appCtx->cuav_auto_control_state.has_lock ||
                appCtx->cuav_auto_control_state.locked_object_id != sample.object_id)
            {
                cuav_reset_auto_control_state(&appCtx->cuav_auto_control_state, TRUE);
                appCtx->cuav_auto_control_state.has_lock = TRUE;
                appCtx->cuav_auto_control_state.locked_object_id = sample.object_id;
                appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
            }
            else if (appCtx->cuav_auto_control_state.last_target_seen_us <= 0 ||
                     (now_us - appCtx->cuav_auto_control_state.last_target_seen_us) >
                        ((gint64)MAX(control_config->control_period_ms * 2U, 1U) * 1000))
            {
                appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
            }
            else if (appCtx->cuav_auto_control_state.target_stable_since_us <= 0)
            {
                appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
            }
            appCtx->cuav_auto_control_state.last_target_seen_us = now_us;
            appCtx->cuav_auto_control_state.lost_zoom_start_us = 0;
            appCtx->cuav_auto_control_state.lost_zoom_hold_complete = FALSE;
            cuav_push_track_sample(&appCtx->cuav_auto_control_state,
                                   control_config->tracking_history_size,
                                   &sample);
            feedback_snapshot = appCtx->cuav_feedback_state;
            state_snapshot = appCtx->cuav_auto_control_state;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            cuav_compute_average_velocity(&state_snapshot,
                                          control_config->tracking_history_size,
                                          &vel_x, &vel_y);

            motion_spacing_ok = (state_snapshot.last_motion_send_us <= 0) ||
                ((now_us - state_snapshot.last_motion_send_us) >=
                 CUAV_MOTION_CMD_MIN_SPACING_USEC);

            if ((now_us - state_snapshot.last_servo_send_us) >=
                ((gint64)control_config->control_period_ms * 1000))
            {
                should_send_servo = cuav_compute_servo_command(control_config,
                                                               &feedback_snapshot,
                                                               &state_snapshot,
                                                               &sample,
                                                               vel_x, vel_y,
                                                               &loc_h, &loc_v,
                                                               &speed_h, &speed_v,
                                                               debug_enabled);
            }
            else if (debug_enabled)
            {
                g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                        " servo suppressed by control period (%" G_GINT64_FORMAT " us remaining)\n",
                        control_config->control_source_id,
                        sample.object_id,
                        (((gint64)control_config->control_period_ms * 1000) -
                         (now_us - state_snapshot.last_servo_send_us)));
            }

            visible_stop_due = cuav_visible_focal_stop_due(control_config,
                                                           &state_snapshot,
                                                           now_us);
            if (cuav_visible_control_enabled(control_config) &&
                (!state_snapshot.visible_initialized ||
                 visible_stop_due ||
                 (now_us - state_snapshot.last_visible_send_us) >=
                    ((gint64)control_config->control_period_ms * 1000)))
            {
                if (visible_stop_due)
                {
                    should_send_visible = TRUE;
                    pt_focal_en = 0;
                    pt_focal = 0.0;
                }
                else
                {
                    should_send_visible = cuav_compute_visible_light_command(control_config,
                                                                             &state_snapshot,
                                                                             &sample,
                                                                             &pt_focal_en,
                                                                             &pt_focal,
                                                                             &pt_focus);
                }
                visible_cmd_changed = should_send_visible &&
                    (!state_snapshot.visible_initialized ||
                     pt_focal_en != state_snapshot.last_pt_focal_en);
                should_send_visible = visible_cmd_changed;
                if (should_send_visible && !motion_spacing_ok && !visible_stop_due && debug_enabled)
                {
                    g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                            " visible focus cmd suppressed by 70ms spacing (%" G_GINT64_FORMAT " us remaining)\n",
                            control_config->control_source_id,
                            sample.object_id,
                            CUAV_MOTION_CMD_MIN_SPACING_USEC -
                                (now_us - state_snapshot.last_motion_send_us));
                }
            }

            if (cuav_infrared_control_enabled(control_config) &&
                (!state_snapshot.infrared_initialized ||
                 (now_us - state_snapshot.last_infrared_send_us) >=
                    ((gint64)control_config->control_period_ms * 1000)))
            {
                should_send_infrared = cuav_compute_infrared_command(control_config,
                                                                     &feedback_snapshot,
                                                                     &state_snapshot,
                                                                     &sample,
                                                                     &ir_focal,
                                                                     &ir_focus);
                if (should_send_infrared && state_snapshot.infrared_initialized &&
                    fabs(ir_focal - state_snapshot.last_ir_focal) <= 1.0)
                {
                    should_send_infrared = FALSE;
                }
            }

            if (should_send_visible && (motion_spacing_ok || visible_stop_due))
            {
                visible_sent = send_cuav_visible_light_command_with_en(appCtx, pt_focal_en,
                                                                        0.0, 0, 0, 0, 0);
                motion_cmd_sent = visible_sent;
                if (visible_sent && control_config->debug)
                {
                    g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                            " ratio=%.3f focal_en=%u focus=%u%s\n",
                            control_config->control_source_id,
                            sample.object_id,
                            sample.target_ratio,
                            pt_focal_en,
                            pt_focus,
                            visible_stop_due ? " stop-after-hold" : "");
                }
            }
            else if (should_send_visible && debug_enabled && !motion_spacing_ok && !visible_stop_due)
            {
                g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                        " visible focus cmd blocked by 70ms spacing\n",
                        control_config->control_source_id,
                        sample.object_id);
            }

            if (!motion_cmd_sent && should_send_servo && motion_spacing_ok)
            {
                servo_sent = send_cuav_servo_command(appCtx, control_config->servo_dev_id, 1, 1, 0, 0,
                                                     speed_h, speed_v, loc_h, loc_v);
                if (!servo_sent && debug_enabled)
                {
                    g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                            " servo send failed loc=(%.2f,%.2f) speed=(%u,%u)\n",
                            control_config->control_source_id,
                            sample.object_id,
                            loc_h,
                            loc_v,
                            speed_h,
                            speed_v);
                }
            }
            else if (should_send_servo && debug_enabled && !motion_spacing_ok)
            {
                g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                        " servo cmd blocked by 70ms spacing\n",
                        control_config->control_source_id,
                        sample.object_id);
            }

            if (should_send_infrared)
            {
                infrared_sent = send_cuav_infrared_command(appCtx, ir_focal,
                                                           ir_focus, 0, 0);
            }

            if ((servo_sent || visible_sent || infrared_sent) && control_config->debug)
            {
                g_print("[cuav][control][sim] err=(%.3f,%.3f) ratio=%.3f "
                        "servo=(%.2f,%.2f) speed=(%u,%u) pt_focal_en=%u ir_focal=%.1f\n",
                        sample.err_x, sample.err_y, sample.target_ratio,
                        loc_h, loc_v, speed_h, speed_v, pt_focal_en, ir_focal);
            }

            if (servo_sent || visible_sent || infrared_sent)
            {
                g_mutex_lock(&appCtx->cuav_control_lock);
                if (servo_sent)
                {
                    appCtx->cuav_auto_control_state.last_servo_valid = TRUE;
                    appCtx->cuav_auto_control_state.last_loc_h = loc_h;
                    appCtx->cuav_auto_control_state.last_loc_v = loc_v;
                    appCtx->cuav_auto_control_state.last_speed_h = speed_h;
                    appCtx->cuav_auto_control_state.last_speed_v = speed_v;
                    appCtx->cuav_auto_control_state.last_servo_send_us = now_us;
                    appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
                    appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_SERVO;
                }
                if (visible_sent)
                {
                    appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
                    appCtx->cuav_auto_control_state.last_pt_focal_en = pt_focal_en;
                    appCtx->cuav_auto_control_state.last_pt_focal = pt_focal;
                    appCtx->cuav_auto_control_state.last_pt_focus = pt_focus;
                    appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
                    appCtx->cuav_auto_control_state.visible_initialized = TRUE;
                    appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
                    appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
                }
                if (infrared_sent)
                {
                    appCtx->cuav_auto_control_state.last_infrared_valid = TRUE;
                    appCtx->cuav_auto_control_state.last_ir_focal = ir_focal;
                    appCtx->cuav_auto_control_state.last_ir_focus = ir_focus;
                    appCtx->cuav_auto_control_state.last_infrared_send_us = now_us;
                    appCtx->cuav_auto_control_state.infrared_initialized = TRUE;
                }
                g_mutex_unlock(&appCtx->cuav_control_lock);
            }
            return;
        }

        g_mutex_lock(&appCtx->cuav_control_lock);
        state_snapshot = appCtx->cuav_auto_control_state;
        had_tracking_before_reset = state_snapshot.has_lock &&
                                    state_snapshot.last_target_seen_us > 0;
        hold_deadline_us = appCtx->cuav_auto_control_state.last_target_seen_us +
                           ((gint64)control_config->target_lost_hold_ms * 1000);
        if (appCtx->cuav_auto_control_state.has_lock &&
            appCtx->cuav_auto_control_state.last_target_seen_us > 0 &&
            now_us > hold_deadline_us)
        {
            cuav_reset_auto_control_state(&appCtx->cuav_auto_control_state, TRUE);
        }
        state_snapshot = appCtx->cuav_auto_control_state;
        g_mutex_unlock(&appCtx->cuav_control_lock);

        lost_zoom_hold_us =
            ((gint64)control_config->startup_pt_focal_min_hold_ms) * 1000;
        lost_zoom_active = !state_snapshot.lost_zoom_hold_complete &&
                           lost_zoom_hold_us > 0 &&
                           (had_tracking_before_reset ||
                            state_snapshot.lost_zoom_active ||
                            !state_snapshot.has_lock);
        if (state_snapshot.lost_zoom_start_us > 0)
        {
            lost_zoom_elapsed_us = now_us - state_snapshot.lost_zoom_start_us;
            lost_zoom_stop_due = lost_zoom_elapsed_us >= lost_zoom_hold_us;
        }
        if (cuav_visible_control_enabled(control_config) && lost_zoom_active)
        {
            motion_spacing_ok = (state_snapshot.last_motion_send_us <= 0) ||
                ((now_us - state_snapshot.last_motion_send_us) >=
                 CUAV_MOTION_CMD_MIN_SPACING_USEC);
            if (lost_zoom_stop_due)
            {
                if (state_snapshot.last_pt_focal_en == 4 && motion_spacing_ok)
                {
                    visible_sent = send_cuav_visible_light_command_with_en(appCtx,
                                                                            0,
                                                                            0.0,
                                                                            0,
                                                                            0,
                                                                            0,
                                                                            0);
                    if (visible_sent)
                    {
                        g_mutex_lock(&appCtx->cuav_control_lock);
                        appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
                        appCtx->cuav_auto_control_state.last_pt_focal_en = 0;
                        appCtx->cuav_auto_control_state.last_pt_focal = 0.0;
                        appCtx->cuav_auto_control_state.last_pt_focus = 0;
                        appCtx->cuav_auto_control_state.lost_zoom_active = FALSE;
                        appCtx->cuav_auto_control_state.lost_zoom_hold_complete = TRUE;
                        appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
                        appCtx->cuav_auto_control_state.visible_initialized = TRUE;
                        appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
                        appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
                        g_mutex_unlock(&appCtx->cuav_control_lock);

                        if (debug_enabled)
                        {
                            g_print("[cuav][control][lost] stop zoom-out after %u ms without target\n",
                                    control_config->startup_pt_focal_min_hold_ms);
                        }
                    }
                }
                else if (state_snapshot.last_pt_focal_en != 4)
                {
                    g_mutex_lock(&appCtx->cuav_control_lock);
                    appCtx->cuav_auto_control_state.lost_zoom_active = FALSE;
                    appCtx->cuav_auto_control_state.lost_zoom_hold_complete = TRUE;
                    g_mutex_unlock(&appCtx->cuav_control_lock);
                }
                else if (debug_enabled && !motion_spacing_ok)
                {
                    g_print("[cuav][control][lost] zoom-out stop blocked by 70ms spacing\n");
                }
            }
            else if ((state_snapshot.last_pt_focal_en != 4 ||
                      state_snapshot.last_visible_send_us <= 0 ||
                      (now_us - state_snapshot.last_visible_send_us) >=
                        ((gint64)MAX(control_config->control_period_ms, 1U) * 1000)) &&
                     motion_spacing_ok)
            {
                visible_sent = send_cuav_visible_light_command_with_en(appCtx,
                                                                        4,
                                                                        0.0,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0);
                if (visible_sent)
                {
                    g_mutex_lock(&appCtx->cuav_control_lock);
                    appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
                    appCtx->cuav_auto_control_state.last_pt_focal_en = 4;
                    appCtx->cuav_auto_control_state.last_pt_focal = 0.0;
                    appCtx->cuav_auto_control_state.last_pt_focus = 0;
                    appCtx->cuav_auto_control_state.lost_zoom_active = TRUE;
                    if (appCtx->cuav_auto_control_state.lost_zoom_start_us <= 0)
                        appCtx->cuav_auto_control_state.lost_zoom_start_us = now_us;
                    appCtx->cuav_auto_control_state.lost_zoom_hold_complete = FALSE;
                    appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
                    appCtx->cuav_auto_control_state.visible_initialized = TRUE;
                    appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
                    appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
                    g_mutex_unlock(&appCtx->cuav_control_lock);

                    if (debug_enabled)
                    {
                        g_print("[cuav][control][lost] no target, zoom out focal_en=4 elapsed=%" G_GINT64_FORMAT "/%" G_GINT64_FORMAT " us\n",
                                state_snapshot.lost_zoom_start_us > 0 ?
                                    lost_zoom_elapsed_us : 0,
                                lost_zoom_hold_us);
                    }
                }
            }
            else if (debug_enabled && !motion_spacing_ok)
            {
                g_print("[cuav][control][lost] zoom-out blocked by 70ms spacing\n");
            }
        }
        return;
    }

    memset(&sample, 0, sizeof(sample));
    sample.valid = TRUE;
    sample.object_id = target_obj->object_id;
    sample.sample_time_us = now_us;
    sample.width = target_obj->rect_params.width;
    sample.height = target_obj->rect_params.height;
    sample.center_x = target_obj->rect_params.left + (sample.width * 0.5);
    sample.center_y = target_obj->rect_params.top + (sample.height * 0.5);
    sample.target_ratio = control_frame_height > 0 ?
        (sample.height / control_frame_height) : 0.0;
    sample.err_x = control_frame_width > 0 ?
        ((sample.center_x - (control_frame_width * 0.5)) /
         (control_frame_width * 0.5)) : 0.0;
    sample.err_y = control_frame_height > 0 ?
        ((sample.center_y - (control_frame_height * 0.5)) /
         (control_frame_height * 0.5)) : 0.0;
    offset_px_x = sample.center_x - (control_frame_width * 0.5);
    offset_px_y = sample.center_y - (control_frame_height * 0.5);
    deadband_px_x = control_frame_width > 0 ?
        (control_config->center_deadband_x * control_frame_width * 0.5) : 0.0;
    deadband_px_y = control_frame_height > 0 ?
        (control_config->center_deadband_y * control_frame_height * 0.5) : 0.0;

    if (debug_enabled)
    {
        g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                " conf=%.3f trk_conf=%.3f bbox=(%.1f,%.1f,%.1f,%.1f) center=(%.1f,%.1f) "
                "offset_px=(%.1f,%.1f) deadband_px=(%.1f,%.1f) err=(%.3f,%.3f) ratio=%.3f\n",
                control_config->control_source_id,
                sample.object_id,
                target_obj->confidence,
                target_obj->tracker_confidence,
                sample.center_x - (sample.width * 0.5),
                sample.center_y - (sample.height * 0.5),
                sample.width,
                sample.height,
                sample.center_x,
                sample.center_y,
                offset_px_x,
                offset_px_y,
                deadband_px_x,
                deadband_px_y,
                sample.err_x,
                sample.err_y,
                sample.target_ratio);
    }

    g_mutex_lock(&appCtx->cuav_control_lock);
    if (!appCtx->cuav_auto_control_state.has_lock ||
        appCtx->cuav_auto_control_state.locked_object_id != sample.object_id)
    {
        cuav_reset_auto_control_state(&appCtx->cuav_auto_control_state, TRUE);
        appCtx->cuav_auto_control_state.has_lock = TRUE;
        appCtx->cuav_auto_control_state.locked_object_id = sample.object_id;
        appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
    }
    else if (appCtx->cuav_auto_control_state.last_target_seen_us <= 0 ||
             (now_us - appCtx->cuav_auto_control_state.last_target_seen_us) >
                ((gint64)MAX(control_config->control_period_ms * 2U, 1U) * 1000))
    {
        appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
    }
    else if (appCtx->cuav_auto_control_state.target_stable_since_us <= 0)
    {
        appCtx->cuav_auto_control_state.target_stable_since_us = now_us;
    }
    appCtx->cuav_auto_control_state.last_target_seen_us = now_us;
    appCtx->cuav_auto_control_state.lost_zoom_start_us = 0;
    appCtx->cuav_auto_control_state.lost_zoom_hold_complete = FALSE;
    cuav_push_track_sample(&appCtx->cuav_auto_control_state,
                           control_config->tracking_history_size,
                           &sample);
    feedback_snapshot = appCtx->cuav_feedback_state;
    state_snapshot = appCtx->cuav_auto_control_state;
    g_mutex_unlock(&appCtx->cuav_control_lock);

    if (cuav_visible_control_enabled(control_config) &&
        state_snapshot.lost_zoom_active)
    {
        visible_sent = send_cuav_visible_light_command_with_en(appCtx,
                                                                0,
                                                                0.0,
                                                                0,
                                                                0,
                                                                0,
                                                                0);
        if (visible_sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
            appCtx->cuav_auto_control_state.last_pt_focal_en = 0;
            appCtx->cuav_auto_control_state.last_pt_focal = 0.0;
            appCtx->cuav_auto_control_state.last_pt_focus = 0;
            appCtx->cuav_auto_control_state.lost_zoom_active = FALSE;
            appCtx->cuav_auto_control_state.lost_zoom_start_us = 0;
            appCtx->cuav_auto_control_state.lost_zoom_hold_complete = FALSE;
            appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
            appCtx->cuav_auto_control_state.visible_initialized = TRUE;
            appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
            appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (debug_enabled)
            {
                g_print("[cuav][control][lost] target reacquired, stop zoom-out focal_en=0 target=%" G_GUINT64_FORMAT "\n",
                        sample.object_id);
            }
            return;
        }
    }

    cuav_compute_average_velocity(&state_snapshot,
                                  control_config->tracking_history_size,
                                  &vel_x, &vel_y);
    motion_spacing_ok = (state_snapshot.last_motion_send_us <= 0) ||
        ((now_us - state_snapshot.last_motion_send_us) >=
         CUAV_MOTION_CMD_MIN_SPACING_USEC);

    if ((now_us - state_snapshot.last_servo_send_us) >=
        ((gint64)control_config->control_period_ms * 1000))
    {
        should_send_servo = cuav_compute_servo_command(control_config,
                                                       &feedback_snapshot,
                                                       &state_snapshot,
                                                       &sample,
                                                       vel_x, vel_y,
                                                       &loc_h, &loc_v,
                                                       &speed_h, &speed_v,
                                                       debug_enabled);
    }
    else if (debug_enabled)
    {
        g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                " servo suppressed by control period (%" G_GINT64_FORMAT " us remaining)\n",
                control_config->control_source_id,
                sample.object_id,
                (((gint64)control_config->control_period_ms * 1000) -
                 (now_us - state_snapshot.last_servo_send_us)));
    }

    visible_stop_due = cuav_visible_focal_stop_due(control_config,
                                                   &state_snapshot,
                                                   now_us);
    if (cuav_visible_control_enabled(control_config) &&
        (!state_snapshot.visible_initialized ||
         visible_stop_due ||
         (now_us - state_snapshot.last_visible_send_us) >=
            ((gint64)control_config->control_period_ms * 1000)))
    {
        if (visible_stop_due)
        {
            should_send_visible = TRUE;
            pt_focal_en = 0;
            pt_focal = 0.0;
        }
        else
        {
            should_send_visible = cuav_compute_visible_light_command(control_config,
                                                                     &state_snapshot,
                                                                     &sample,
                                                                     &pt_focal_en,
                                                                     &pt_focal,
                                                                     &pt_focus);
        }
        visible_cmd_changed = should_send_visible &&
            (!state_snapshot.visible_initialized ||
             pt_focal_en != state_snapshot.last_pt_focal_en);
        should_send_visible = visible_cmd_changed;
        if (should_send_visible && !motion_spacing_ok && !visible_stop_due && debug_enabled)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " visible focus cmd suppressed by 70ms spacing (%" G_GINT64_FORMAT " us remaining)\n",
                    control_config->control_source_id,
                    sample.object_id,
                    CUAV_MOTION_CMD_MIN_SPACING_USEC -
                        (now_us - state_snapshot.last_motion_send_us));
        }
    }

    if (cuav_infrared_control_enabled(control_config) &&
        (!state_snapshot.infrared_initialized ||
         (now_us - state_snapshot.last_infrared_send_us) >=
            ((gint64)control_config->control_period_ms * 1000)))
    {
        should_send_infrared = cuav_compute_infrared_command(control_config,
                                                             &feedback_snapshot,
                                                             &state_snapshot,
                                                             &sample,
                                                             &ir_focal,
                                                             &ir_focus);
        if (should_send_infrared && state_snapshot.infrared_initialized &&
            fabs(ir_focal - state_snapshot.last_ir_focal) <= 1.0)
        {
            should_send_infrared = FALSE;
        }
    }

    if (should_send_visible && (motion_spacing_ok || visible_stop_due))
    {
        visible_sent = send_cuav_visible_light_command_with_en(appCtx, pt_focal_en,
                                                                0.0, 0, 0, 0, 0);
        motion_cmd_sent = visible_sent;
        if (visible_sent && control_config->debug)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " ratio=%.3f focal_en=%u focus=%u%s\n",
                    control_config->control_source_id,
                    sample.object_id,
                    sample.target_ratio,
                    pt_focal_en,
                    pt_focus,
                    visible_stop_due ? " stop-after-hold" : "");
        }
    }
    else if (should_send_visible && debug_enabled && !motion_spacing_ok && !visible_stop_due)
    {
        g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                " visible focus cmd blocked by 70ms spacing\n",
                control_config->control_source_id,
                sample.object_id);
    }

    if (!motion_cmd_sent && should_send_servo && motion_spacing_ok)
    {
        servo_sent = send_cuav_servo_command(appCtx, control_config->servo_dev_id, 1, 1, 0, 0,
                                             speed_h, speed_v, loc_h, loc_v);
        if (servo_sent && control_config->debug)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " err=(%.3f,%.3f) vel=(%.3f,%.3f) servo=(%.2f,%.2f) speed=(%u,%u)\n",
                    control_config->control_source_id,
                    sample.object_id,
                    sample.err_x, sample.err_y, vel_x, vel_y,
                    loc_h, loc_v, speed_h, speed_v);
        }
        else if (!servo_sent && control_config->debug)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " servo send failed loc=(%.2f,%.2f) speed=(%u,%u)\n",
                    control_config->control_source_id,
                    sample.object_id,
                    loc_h,
                    loc_v,
                    speed_h,
                    speed_v);
        }
    }
    else if (should_send_servo && debug_enabled && !motion_spacing_ok)
    {
        g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                " servo cmd blocked by 70ms spacing\n",
                control_config->control_source_id,
                sample.object_id);
    }

    if (should_send_infrared)
    {
        infrared_sent = send_cuav_infrared_command(appCtx, ir_focal,
                                                   ir_focus, 0, 0);
        if (infrared_sent && control_config->debug)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " ratio=%.3f ir_focal=%.1f ir_focus=%u\n",
                    control_config->control_source_id,
                    sample.object_id,
                    sample.target_ratio,
                    ir_focal,
                    ir_focus);
        }
    }

    if (servo_sent || visible_sent || infrared_sent)
    {
        g_mutex_lock(&appCtx->cuav_control_lock);
        if (servo_sent)
        {
            appCtx->cuav_auto_control_state.last_servo_valid = TRUE;
            appCtx->cuav_auto_control_state.last_loc_h = loc_h;
            appCtx->cuav_auto_control_state.last_loc_v = loc_v;
            appCtx->cuav_auto_control_state.last_speed_h = speed_h;
            appCtx->cuav_auto_control_state.last_speed_v = speed_v;
            appCtx->cuav_auto_control_state.last_servo_send_us = now_us;
            appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
            appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_SERVO;
        }
        if (visible_sent)
        {
            appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
            appCtx->cuav_auto_control_state.last_pt_focal_en = pt_focal_en;
            appCtx->cuav_auto_control_state.last_pt_focal = pt_focal;
            appCtx->cuav_auto_control_state.last_pt_focus = pt_focus;
            appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
            appCtx->cuav_auto_control_state.visible_initialized = TRUE;
            appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
            appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
        }
        if (infrared_sent)
        {
            appCtx->cuav_auto_control_state.last_infrared_valid = TRUE;
            appCtx->cuav_auto_control_state.last_ir_focal = ir_focal;
            appCtx->cuav_auto_control_state.last_ir_focus = ir_focus;
            appCtx->cuav_auto_control_state.last_infrared_send_us = now_us;
            appCtx->cuav_auto_control_state.infrared_initialized = TRUE;
        }
        g_mutex_unlock(&appCtx->cuav_control_lock);
    }
}

/**
 * @brief 引导信息回调（0x7203报文），打印并记录目标引导数据到日志和CSV
 * @param header 报文公共头
 * @param guidance 引导信息结构体
 * @param user_data 用户数据（AppCtx指针）
 */
static void on_cuav_guidance(const CUAVCommonHeader *header,
                             const CUAVGuidanceInfo *guidance,
                             gpointer user_data)
{
    AppCtx *appCtx = (AppCtx *)user_data;
    gchar line[1024] = {0};
    gchar csv_path[1024] = {0};
    gchar csv_row[1024] = {0};

    (void)user_data;
    if (!header || !guidance)
        return;

    if (appCtx && appCtx->config.udpjsonmeta_config.enable_cuav_debug)
    {
        g_print("[cuav][guidance] msg_sn=%u time=%u-%02u-%02u %02u:%02u:%02u.%.0f "
                "tar_id=%u cat=%u stat=%u enu_a=%.2f enu_e=%.2f lon=%.6f lat=%.6f alt=%.2f\n",
                header->msg_sn,
                guidance->yr, guidance->mo, guidance->dy,
                guidance->h, guidance->min, guidance->sec, guidance->msec,
                guidance->tar_id, guidance->tar_category, guidance->guid_stat,
                guidance->enu_a, guidance->enu_e, guidance->lon, guidance->lat, guidance->alt);
    }

    g_snprintf(line, sizeof(line),
               "[cuav][guidance] msg_id=0x%04X msg_sn=%u msg_type=%u tar_id=%u cat=%u stat=%u "
               "enu_a=%.2f enu_e=%.2f lon=%.6f lat=%.6f alt=%.2f",
               header->msg_id, header->msg_sn, header->msg_type,
               guidance->tar_id, guidance->tar_category, guidance->guid_stat,
               guidance->enu_a, guidance->enu_e, guidance->lon, guidance->lat, guidance->alt);
    append_cuav_log_line(line);

    if (get_cuav_csv_path(appCtx, "cuav_guidance.csv", csv_path, sizeof(csv_path)))
    {
        g_snprintf(csv_row, sizeof(csv_row),
                   "%u,%u,%u,%u,%u,%u,%.2f,%.2f,%.6f,%.6f,%.2f",
                   header->msg_id, header->msg_sn, header->msg_type,
                   guidance->tar_id, guidance->tar_category, guidance->guid_stat,
                   guidance->enu_a, guidance->enu_e,
                   guidance->lon, guidance->lat, guidance->alt);
        append_cuav_csv_row(csv_path,
                            "msg_id,msg_sn,msg_type,tar_id,tar_category,guid_stat,enu_a,enu_e,lon,lat,alt",
                            csv_row);
    }
}

/**
 * @brief EO系统参数反馈回调（0x7201报文）
 * 更新全局反馈状态（云台位置、焦距、跟踪状态等），并转发给cuavcontrolsink
 * @param header 报文公共头
 * @param eo_param EO系统参数
 * @param user_data 用户数据（AppCtx指针）
 */
static void on_cuav_eo_system(const CUAVCommonHeader *header,
                              const CUAVEOSystemParam *eo_param,
                              gpointer user_data)
{
    AppCtx *appCtx = (AppCtx *)user_data;
    gchar line[1024] = {0};
    gchar csv_path[1024] = {0};
    gchar csv_row[1024] = {0};

    (void)user_data;
    if (!header || !eo_param)
        return;

    if (appCtx && appCtx->config.udpjsonmeta_config.enable_cuav_debug)
    {
        g_print("[cuav][eo-system] msg_sn=%u sv_stat=%u st_loc_h=%.2f st_loc_v=%.2f "
                "pt_focal=%.1f ir_focal=%.1f trk_dev=%u trk_stat=%u\n",
                header->msg_sn,
                eo_param->sv_stat, eo_param->st_loc_h, eo_param->st_loc_v,
                eo_param->pt_focal, eo_param->ir_focal,
                eo_param->trk_dev, eo_param->trk_stat);
    }

    g_snprintf(line, sizeof(line),
               "[cuav][eo-system] msg_id=0x%04X msg_sn=%u msg_type=%u "
               "sv_stat=%u st_loc_h=%.2f st_loc_v=%.2f pt_focal=%.1f ir_focal=%.1f "
               "trk_dev=%u pt_link=%u ir_link=%u trk_stat=%u",
               header->msg_id, header->msg_sn, header->msg_type,
               eo_param->sv_stat, eo_param->st_loc_h, eo_param->st_loc_v,
               eo_param->pt_focal, eo_param->ir_focal,
               eo_param->trk_dev, eo_param->pt_trk_link,
               eo_param->ir_trk_link, eo_param->trk_stat);
    append_cuav_log_line(line);

    if (get_cuav_csv_path(appCtx, "cuav_eo_system.csv", csv_path, sizeof(csv_path)))
    {
        g_snprintf(csv_row, sizeof(csv_row),
                   "%u,%u,%u,%u,%.2f,%.2f,%.1f,%.1f,%u,%u,%u,%u",
                   header->msg_id, header->msg_sn, header->msg_type,
                   eo_param->sv_stat, eo_param->st_loc_h, eo_param->st_loc_v,
                   eo_param->pt_focal, eo_param->ir_focal,
                   eo_param->trk_dev, eo_param->pt_trk_link,
                   eo_param->ir_trk_link, eo_param->trk_stat);
        append_cuav_csv_row(csv_path,
                            "msg_id,msg_sn,msg_type,sv_stat,st_loc_h,st_loc_v,pt_focal,ir_focal,trk_dev,pt_trk_link,ir_trk_link,trk_stat",
                            csv_row);
    }

    if (appCtx)
    {
        CuavFeedbackState feedback_snapshot;

        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_feedback_state.valid = TRUE;
        appCtx->cuav_feedback_state.updated_at_us = g_get_monotonic_time();
        appCtx->cuav_feedback_state.st_loc_h = eo_param->st_loc_h;
        appCtx->cuav_feedback_state.st_loc_v = eo_param->st_loc_v;
        appCtx->cuav_feedback_state.pt_focal = eo_param->pt_focal;
        appCtx->cuav_feedback_state.pt_focus = eo_param->pt_focus;
        appCtx->cuav_feedback_state.ir_focal = eo_param->ir_focal;
        appCtx->cuav_feedback_state.ir_focus = eo_param->ir_focus;
        appCtx->cuav_feedback_state.sv_stat = eo_param->sv_stat;
        appCtx->cuav_feedback_state.trk_dev = eo_param->trk_dev;
        appCtx->cuav_feedback_state.pt_trk_link = eo_param->pt_trk_link;
        appCtx->cuav_feedback_state.ir_trk_link = eo_param->ir_trk_link;
        appCtx->cuav_feedback_state.trk_stat = eo_param->trk_stat;
        feedback_snapshot = appCtx->cuav_feedback_state;
        g_mutex_unlock(&appCtx->cuav_control_lock);

        update_cuav_eo_system_state(appCtx, &feedback_snapshot);
    }
}

/**
 * @brief 云台伺服控制反馈回调（0x7204报文回显），记录到日志和CSV
 * @param header 报文公共头
 * @param servo 伺服控制数据
 * @param user_data 用户数据（AppCtx指针）
 */
static void on_cuav_servo_control(const CUAVCommonHeader *header,
                                  const CUAVServoControl *servo,
                                  gpointer user_data)
{
    AppCtx *appCtx = (AppCtx *)user_data;
    gchar line[1024] = {0};
    gchar csv_path[1024] = {0};
    gchar csv_row[1024] = {0};

    (void)user_data;
    if (!header || !servo)
        return;

    if (appCtx && appCtx->config.udpjsonmeta_config.enable_cuav_debug)
    {
        g_print("[cuav][servo] msg_sn=%u dev_id=%u ctrl_en=%u mode_h=%u mode_v=%u "
                "loc_h=%.2f loc_v=%.2f speed_h=%u speed_v=%u\n",
                header->msg_sn, servo->dev_id, servo->ctrl_en,
                servo->mode_h, servo->mode_v,
                servo->loc_h, servo->loc_v, servo->speed_h, servo->speed_v);
    }

    g_snprintf(line, sizeof(line),
               "[cuav][servo] msg_id=0x%04X msg_sn=%u msg_type=%u "
               "dev_id=%u ctrl_en=%u mode_h=%u mode_v=%u loc_h=%.2f loc_v=%.2f speed_h=%u speed_v=%u",
               header->msg_id, header->msg_sn, header->msg_type,
               servo->dev_id, servo->ctrl_en, servo->mode_h, servo->mode_v,
               servo->loc_h, servo->loc_v, servo->speed_h, servo->speed_v);
    append_cuav_log_line(line);

    if (get_cuav_csv_path(appCtx, "cuav_servo.csv", csv_path, sizeof(csv_path)))
    {
        g_snprintf(csv_row, sizeof(csv_row),
                   "%u,%u,%u,%u,%u,%u,%u,%.2f,%.2f,%u,%u",
                   header->msg_id, header->msg_sn, header->msg_type,
                   servo->dev_id, servo->ctrl_en, servo->mode_h, servo->mode_v,
                   servo->loc_h, servo->loc_v, servo->speed_h, servo->speed_v);
        append_cuav_csv_row(csv_path,
                            "msg_id,msg_sn,msg_type,dev_id,ctrl_en,mode_h,mode_v,loc_h,loc_v,speed_h,speed_v",
                            csv_row);
    }
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

/**
 * @brief 创建解复用后的处理分支（sink bin + 可选OSD），添加bbox回调探针
 * @param appCtx 应用上下文
 * @param index 源索引
 * @return 成功返回TRUE
 */
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

/**
 * @brief udpmulticast源收到新数据时的回调，打印接收帧计数和缓冲区大小
 * @param sink appsink元素
 * @param appCtx 应用上下文
 * @return GST_FLOW_OK
 */
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

/**
 * @brief 创建并添加独立的UDP组播源分支到pipeline（udpsrc→queue→appsink）
 * 该分支独立于视频推理链路，仅接收和打印组播数据
 * @param appCtx 应用上下文
 * @return 成功返回TRUE，未启用时也返回TRUE
 */
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
            gst_udpjson_meta_set_eo_system_callback(GST_UDPJSON_META(udpjsonmeta),
                                                    on_cuav_eo_system,
                                                    pipeline->common_elements.appCtx);
            gst_udpjson_meta_set_servo_control_callback(GST_UDPJSON_META(udpjsonmeta),
                                                        on_cuav_servo_control,
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

    if (!create_cuav_control_element(config, pipeline))
    {
        goto done;
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

/**
 * @brief 检查指定源ID是否配置了可用的sink（非demux模式）
 * @param config 全局配置
 * @param source_id 源索引
 * @return 存在可用sink返回TRUE
 */
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

/**
 * @brief 暂停pipeline，将状态切换到PAUSED并暂停性能测量
 * @return 成功暂停返回TRUE，当前非PLAYING状态或异步切换返回FALSE
 */
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

/**
 * @brief 恢复pipeline，将状态切换到PLAYING并恢复性能测量
 * @return 成功恢复返回TRUE，当前非PAUSED状态或异步切换返回FALSE
 */
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
