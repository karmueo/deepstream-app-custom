/**
 * @file deepstream_app_cuav_control.c
 * @brief C-UAV自动控制模块实现。
 *
 * 负责C-UAV控制发送元素创建、协议反馈回调注册与处理、启动测试报文、
 * 自动跟踪控制、启动预置位和角点循环等逻辑。该文件仅承载C-UAV
 * 业务控制代码，DeepStream主pipeline的创建与生命周期管理仍保留在
 * deepstream_app.c中。
 */

#include <gst/gst.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <glib/gstdio.h>
#include "deepstream_app_cuav_control.h"
#include "gstudpjsonmeta.h"

GST_DEBUG_CATEGORY_EXTERN(NVDS_APP);

#define DEFAULT_CUAV_LOG_PATH "/tmp/deepstream_cuav_packets.log"
#define DEFAULT_CUAV_TEST_MULTICAST_IP "239.255.88.51"
#define DEFAULT_CUAV_TEST_MULTICAST_PORT 18003
#define CUAV_FEEDBACK_STALE_USEC (2 * G_USEC_PER_SEC)
#define CUAV_MOTION_CMD_MIN_SPACING_USEC (70 * 1000)
#define CUAV_FOCAL_REACHED_EPSILON 1.0

typedef enum
{
    CUAV_MOTION_CMD_NONE = 0,
    CUAV_MOTION_CMD_SERVO = 1,
    CUAV_MOTION_CMD_VISIBLE = 2,
} CuavMotionCommandType;

static GMutex s_cuav_csv_lock;

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
                                                        guint pt_focal,
                                                        guint pt_focus_en,
                                                        guint pt_focus,
                                                        guint pt_focus_mode,
                                                        guint pt_zoom);
static gboolean send_cuav_visible_light_command(AppCtx *appCtx,
                                                guint pt_focal,
                                                guint pt_focus_en,
                                                guint pt_focus,
                                                guint pt_focus_mode,
                                                guint pt_zoom);
static gboolean send_cuav_servo_test_message(AppCtx *appCtx);
static gboolean send_cuav_visible_light_test_message(AppCtx *appCtx);
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
                                                   const CuavFeedbackState *feedback_state,
                                                   const CuavAutoControlState *auto_state,
                                                   const CuavTrackSample *sample,
                                                   guint *pt_focal_en,
                                                   gdouble *pt_focal,
                                                   guint *pt_focus);
static gdouble cuav_get_current_pt_focal(const NvDsCuavControlConfig *control_config,
                                         const CuavFeedbackState *feedback_state,
                                         const CuavAutoControlState *auto_state);

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
 * @brief 获取当前可见光焦距，按优先级回退
 * 1. 新鲜的设备反馈值（feedback_state->pt_focal）
 * 2. 上次成功发送并确认的值（auto_state->last_pt_focal）
 * 3. 配置最小值（pt_focal_min）
 * 结果会被 clamp 到 [pt_focal_min, pt_focal_max] 区间
 */
static gdouble
cuav_get_current_pt_focal(const NvDsCuavControlConfig *control_config,
                          const CuavFeedbackState *feedback_state,
                          const CuavAutoControlState *auto_state)
{
    gdouble focal_min = 19.0;
    gdouble focal_max = 4000.0;
    gdouble focal = 0.0;

    if (control_config)
    {
        focal_min = control_config->pt_focal_min;
        focal_max = control_config->pt_focal_max;
        if (focal_max < focal_min)
            focal_max = focal_min;
    }

    if (feedback_state &&
        cuav_feedback_is_fresh(feedback_state,
                               control_config ? control_config->state_stale_timeout_ms : 2000) &&
        feedback_state->pt_focal > 0.0)
    {
        focal = feedback_state->pt_focal;
    }
    else if (auto_state && auto_state->last_visible_valid &&
             auto_state->last_pt_focal > 0.0)
    {
        focal = auto_state->last_pt_focal;
    }
    else
    {
        focal = focal_min;
    }

    return clamp_cuav_double(focal, focal_min, focal_max);
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
           (!isnan(control_config->corner_home_loc_h_deg) ||
            !isnan(control_config->corner_home_loc_v_deg));
}

/**
 * @brief 判断启动预置位是否配置了可见光预置（启动焦距/对焦或对焦值）
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
                                                       control_config->startup_pt_focal_min_enable ? 1 : 0,
                                                       control_config->startup_pt_focal_min_enable ?
                                                           control_config->startup_pt_focal : 0,
                                                       control_config->startup_pt_focal_min_enable ||
                                                           control_config->corner_home_pt_focus != G_MAXUINT ? 1 : 0,
                                                       control_config->startup_pt_focal_min_enable ?
                                                           control_config->startup_pt_focus :
                                                           (control_config->corner_home_pt_focus != G_MAXUINT ?
                                                                control_config->corner_home_pt_focus : 100),
                                                       1,
                                                       0);
        if (sent)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_startup_preset_state.visible_applied = TRUE;
            appCtx->cuav_startup_preset_state.last_command_sent_us = now_us;
            appCtx->cuav_startup_preset_state.phase_started_us = now_us;
            appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_COMPLETE;
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][startup-preset] send visible focal_en=%u focal=%u focus_en=%u focus=%u\n",
                        control_config->startup_pt_focal_min_enable ? 1U : 0U,
                        control_config->startup_pt_focal_min_enable ?
                            control_config->startup_pt_focal : 0U,
                        (control_config->startup_pt_focal_min_enable ||
                         control_config->corner_home_pt_focus != G_MAXUINT) ? 1U : 0U,
                        control_config->startup_pt_focal_min_enable ?
                            control_config->startup_pt_focus :
                            (control_config->corner_home_pt_focus != G_MAXUINT ?
                                 control_config->corner_home_pt_focus : 100U));
            }
        }
        return TRUE;

    case CUAV_STARTUP_PRESET_PHASE_HOLD_VISIBLE_PRESET:
        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_startup_preset_state.phase = CUAV_STARTUP_PRESET_PHASE_COMPLETE;
        appCtx->cuav_startup_preset_state.phase_started_us = now_us;
        g_mutex_unlock(&appCtx->cuav_control_lock);
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
 * @brief 重置角点循环状态机到初始阶段
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
    state->pending_pt_focal_valid = FALSE;
    state->pending_pt_focal = 0.0;
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
        state->last_servo_send_us = 0;
        state->last_visible_send_us = 0;
        state->last_motion_send_us = 0;
        state->last_motion_type = CUAV_MOTION_CMD_NONE;
    }
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
 * @brief 计算可见光绝对焦距控制指令
 * target_ratio < (min - deadband) → pt_focal += pt_focal_step
 * target_ratio > (max + deadband) → pt_focal -= pt_focal_step
 * 否则 → 不调整
 * @param[out] pt_focal_en 变焦使能指令(0=不调整,1=设置绝对焦距)
 * @param[out] pt_focal 焦距目标值
 * @param[out] pt_focus 对焦值
 * @return 成功返回TRUE
 */
static gboolean
cuav_compute_visible_light_command(const NvDsCuavControlConfig *control_config,
                                   const CuavFeedbackState *feedback_state,
                                   const CuavAutoControlState *auto_state,
                                   const CuavTrackSample *sample,
                                   guint *pt_focal_en,
                                   gdouble *pt_focal,
                                   guint *pt_focus)
{
    guint focal_en = 0;
    gint64 zoom_in_stable_us = 0;
    gdouble focal_min = 0.0;
    gdouble focal_max = 0.0;
    gdouble focal_step = 0.0;
    gdouble current_focal = 0.0;
    gdouble target_focal = 0.0;

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

    focal_min = control_config->pt_focal_min;
    focal_max = control_config->pt_focal_max;
    if (focal_max < focal_min)
        focal_max = focal_min;
    focal_step = control_config->pt_focal_step > 0.0 ?
                 control_config->pt_focal_step : 10.0;
    current_focal = cuav_get_current_pt_focal(control_config,
                                              feedback_state,
                                              auto_state);
    target_focal = current_focal;

    if (sample->target_ratio < (control_config->zoom_target_ratio_min - control_config->zoom_deadband))
    {
        focal_en = 1;
        target_focal = clamp_cuav_double(current_focal + focal_step,
                                         focal_min,
                                         focal_max);
        zoom_in_stable_us =
            ((gint64)MAX(control_config->visible_focal_hold_ms,
                         control_config->control_period_ms *
                            MAX(control_config->tracking_history_size, 1U))) * 1000;
        if (auto_state->target_stable_since_us <= 0 ||
            sample->sample_time_us - auto_state->target_stable_since_us < zoom_in_stable_us)
        {
            focal_en = 0;
        }
        else if (fabs(target_focal - current_focal) <= CUAV_FOCAL_REACHED_EPSILON)
        {
            focal_en = 0;
        }
    }
    else if (sample->target_ratio > (control_config->zoom_target_ratio_max + control_config->zoom_deadband))
    {
        focal_en = 1;
        target_focal = clamp_cuav_double(current_focal - focal_step,
                                         focal_min,
                                         focal_max);
        if (fabs(target_focal - current_focal) <= CUAV_FOCAL_REACHED_EPSILON)
            focal_en = 0;
    }
    else
    {
        focal_en = 0;
    }

    *pt_focal_en = focal_en;
    *pt_focal = target_focal;
    return TRUE;
}

/**
 * @brief 判断旧连续变焦指令（focal_en=3/4）是否需要发送停止指令
 * @note  新绝对焦距模式下 focal_en 仅为 0 或 1，此函数始终返回 FALSE
 * @return 仅当上次发送 focal_en 为 3 或 4 且超过 visible_focal_hold_ms 时返回 TRUE
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

    if ((auto_state->last_pt_focal_en != 3 && auto_state->last_pt_focal_en != 4) ||
        auto_state->last_visible_send_us <= 0)
        return FALSE;

    return (now_us - auto_state->last_visible_send_us) >=
           ((gint64)control_config->visible_focal_hold_ms * 1000);
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
                                "sv-stat", G_TYPE_INT,
                                (gint)feedback_state->sv_stat,
                                "trk-dev", G_TYPE_INT,
                                (gint)feedback_state->trk_dev,
                                "pt-trk-link", G_TYPE_INT,
                                (gint)feedback_state->pt_trk_link,
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
 * @param pt_focal_en 焦距控制使能(0=不调整,1=设置绝对焦距；3/4为旧连续变焦模式，已废弃)
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
                                        guint pt_focal,
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
                                "pt-focal", G_TYPE_INT, (gint)pt_focal,
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
                                guint pt_focal,
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
 * @brief 创建并初始化cuavcontrolsink GStreamer元素，设置组播网络参数
 * 同时重置启动预置位和角点循环状态机
 * @param config 全局配置
 * @param pipeline 管道结构
 * @return 成功返回TRUE
 */
gboolean
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

    g_print("[cuav][control] enabled via sink type=8 target=%s:%u iface=%s startup-test=%d auto-track=%d visible=%d servo-dev-id=%u\n",
            control_config->multicast_ip ?
                control_config->multicast_ip : "(default)",
            control_config->port,
            control_config->iface ?
                control_config->iface : "(default)",
            control_config->send_test_on_startup,
            control_config->auto_track_enable,
            control_config->visible_light_control_enable,
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
        g_print("[cuav][corner-cycle] enabled repeat=%u corner-cycle=%u offset=(%.1f,%.1f) dwell=%u ms speed=%u\n",
                control_config->sequence_repeat_count,
                control_config->corner_cycle_count,
                control_config->corner_offset_h_deg,
                control_config->corner_offset_v_deg,
                control_config->corner_dwell_ms,
                control_config->corner_servo_speed);
        g_print("[cuav][corner-cycle] home target loc=(%.1f,%.1f) preset_focus_en=%u\n",
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

    result = send_cuav_visible_light_command(appCtx, 500, 1, 100, 1, 0);

    g_print("[cuav][control-test] visible-light test send result=%d\n", result);
    return result;
}

/**
 * @brief 依次发送云台、可见光测试指令（仅在启动测试模式且非自动跟踪时生效）
 * @return 全部发送成功返回TRUE
 */
gboolean
send_cuav_test_messages(AppCtx *appCtx)
{
    gboolean servo_ok = FALSE;
    gboolean visible_ok = FALSE;
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

    return servo_ok && visible_ok;
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
 * @brief 处理角点循环测试状态机
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
    gboolean home_visible_preset_valid = FALSE;
    gboolean home_visible_focus_valid = FALSE;
    gboolean home_loc_configured = FALSE;
    guint home_visible_focus = 100;
    guint min_gap_ms = 1;
    gint64 min_gap_us = 0;
    gint64 corner_dwell_us = 0;
    gint64 home_settle_timeout_us = 0;
    gboolean visible_enabled = FALSE;

    if (!appCtx || !control_config || !appCtx->pipeline.common_elements.cuav_control)
        return FALSE;

    visible_enabled = cuav_visible_control_enabled(control_config);
    min_gap_ms = MAX(control_config->control_period_ms, 1U);
    min_gap_us = ((gint64)min_gap_ms) * 1000;
    corner_dwell_us = ((gint64)MAX(control_config->corner_dwell_ms, 1U)) * 1000;
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
        appCtx->cuav_corner_zoom_cycle_state.phase =
            startup_snapshot.servo_applied ?
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER :
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
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
            g_print("[cuav][corner-cycle] complete repeat=%u/%u\n",
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
                g_print("[cuav][corner-cycle] home start repeat=%u/%u home=(%.2f,%.2f) "
                        "corner_cycle=%u offset=(%.1f,%.1f) preset_focus_en=%u\n",
                        state_snapshot.outer_repeat_index + 1,
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
                g_print("[cuav][corner-cycle] send home repeat=%u/%u loc=(%.2f,%.2f) speed=%u\n",
                        state_snapshot.outer_repeat_index + 1,
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
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            appCtx->cuav_corner_zoom_cycle_state.phase =
                home_visible_preset_valid ?
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                    CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
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
                g_print("[cuav][corner-cycle][warn] home settle timeout at preset, continue to next stage\n");
            else
                g_print("[cuav][corner-cycle][warn] home settle timeout, continue to next stage\n");
        }

        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
        appCtx->cuav_corner_zoom_cycle_state.phase =
            home_visible_preset_valid ?
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET :
                CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
        g_mutex_unlock(&appCtx->cuav_control_lock);
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_VISIBLE_PRESET:
        if (state_snapshot.last_command_sent_us > 0 &&
            (now_us - state_snapshot.last_command_sent_us) < min_gap_us)
            return TRUE;

        if (!home_visible_preset_valid)
        {
            g_mutex_lock(&appCtx->cuav_control_lock);
            appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
            appCtx->cuav_corner_zoom_cycle_state.phase_started_us = now_us;
            g_mutex_unlock(&appCtx->cuav_control_lock);
            return TRUE;
        }

        sent = send_cuav_visible_light_command_with_en(appCtx,
                                                       0,
                                                       0,
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
                g_print("[cuav][corner-cycle] home preset repeat=%u/%u focus_en=%u focus=%u\n",
                        state_snapshot.outer_repeat_index + 1,
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
        appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_CORNER;
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
                g_print("[cuav][corner-cycle] send corner repeat=%u/%u cycle=%u/%u corner=%s loc=(%.2f,%.2f) speed=%u\n",
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
            if ((state_snapshot.outer_repeat_index + 1) < repeat_limit)
            {
                appCtx->cuav_corner_zoom_cycle_state.outer_repeat_index++;
                appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_SEND_HOME_SERVO;
            }
            else
            {
                appCtx->cuav_corner_zoom_cycle_state.phase = CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE;
            }
            g_mutex_unlock(&appCtx->cuav_control_lock);

            if (control_config->debug)
            {
                g_print("[cuav][corner-cycle] servo stop repeat=%u/%u loc=(%.2f,%.2f)\n",
                        state_snapshot.outer_repeat_index + 1,
                        repeat_limit,
                        state_snapshot.last_loc_h,
                        state_snapshot.last_loc_v);
            }
        }
        return TRUE;

    case CUAV_CORNER_ZOOM_CYCLE_PHASE_COMPLETE:
    default:
        g_mutex_lock(&appCtx->cuav_control_lock);
        if (!appCtx->cuav_corner_zoom_cycle_state.final_logged)
        {
            appCtx->cuav_corner_zoom_cycle_state.final_logged = TRUE;
            g_print("[cuav][corner-cycle] complete repeat=%u/%u\n",
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
 * 自动跟踪流程: 选择目标 → 计算采样 → 速度估计 → 计算云台/可见光指令 → 发送
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
    gboolean startup_preset_required = FALSE;
    CuavTrackSample sample;
    CuavFeedbackState feedback_snapshot;
    CuavAutoControlState state_snapshot;
    gdouble vel_x = 0.0;
    gdouble vel_y = 0.0;
    gboolean should_send_servo = FALSE;
    gboolean should_send_visible = FALSE;
    gboolean visible_cmd_changed = FALSE;
    gboolean motion_spacing_ok = FALSE;
    gboolean motion_cmd_sent = FALSE;
    gboolean had_tracking_before_reset = FALSE;
    gboolean lost_zoom_active = FALSE;
    gdouble loc_h = 0.0;
    gdouble loc_v = 0.0;
    guint speed_h = 0;
    guint speed_v = 0;
    guint pt_focal_en = 0;
    gdouble pt_focal = 0.0;
    guint pt_focus = 100;
    gboolean visible_stop_due = FALSE;
    gdouble offset_px_x = 0.0;
    gdouble offset_px_y = 0.0;
    gdouble deadband_px_x = 0.0;
    gdouble deadband_px_y = 0.0;
    gint control_frame_width = 0;
    gint control_frame_height = 0;
    gboolean servo_sent = FALSE;
    gboolean visible_sent = FALSE;
    gboolean debug_enabled = FALSE;
    gboolean visible_focal_pending = FALSE;

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
            g_print("[cuav][control][auto] source=%u no valid tracked target\n",
                    control_config->control_source_id);
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
        feedback_snapshot = appCtx->cuav_feedback_state;
        if (appCtx->cuav_auto_control_state.pending_pt_focal_valid &&
            cuav_feedback_is_fresh(&feedback_snapshot,
                                   control_config->state_stale_timeout_ms) &&
            fabs(feedback_snapshot.pt_focal -
                 appCtx->cuav_auto_control_state.pending_pt_focal) <=
                CUAV_FOCAL_REACHED_EPSILON)
        {
            appCtx->cuav_auto_control_state.pending_pt_focal_valid = FALSE;
            appCtx->cuav_auto_control_state.last_pt_focal =
                feedback_snapshot.pt_focal;
        }
        state_snapshot = appCtx->cuav_auto_control_state;
        g_mutex_unlock(&appCtx->cuav_control_lock);

        lost_zoom_active = !state_snapshot.lost_zoom_hold_complete &&
                           (had_tracking_before_reset ||
                            state_snapshot.lost_zoom_active);
        if (cuav_visible_control_enabled(control_config) && lost_zoom_active)
        {
            motion_spacing_ok = (state_snapshot.last_motion_send_us <= 0) ||
                ((now_us - state_snapshot.last_motion_send_us) >=
                 CUAV_MOTION_CMD_MIN_SPACING_USEC);
            if (!state_snapshot.pending_pt_focal_valid &&
                (state_snapshot.last_pt_focal_en != 1 ||
                 state_snapshot.last_visible_send_us <= 0 ||
                 (now_us - state_snapshot.last_visible_send_us) >=
                    ((gint64)MAX(control_config->control_period_ms, 1U) * 1000)) &&
                motion_spacing_ok)
            {
                gdouble current_focal =
                    cuav_get_current_pt_focal(control_config,
                                              &feedback_snapshot,
                                              &state_snapshot);
                pt_focal = clamp_cuav_double(current_focal -
                                                 MAX(control_config->pt_focal_step, 1.0),
                                             control_config->pt_focal_min,
                                             control_config->pt_focal_max);
                if (fabs(pt_focal - current_focal) <= CUAV_FOCAL_REACHED_EPSILON)
                {
                    g_mutex_lock(&appCtx->cuav_control_lock);
                    appCtx->cuav_auto_control_state.lost_zoom_active = FALSE;
                    appCtx->cuav_auto_control_state.lost_zoom_hold_complete = TRUE;
                    g_mutex_unlock(&appCtx->cuav_control_lock);
                    if (debug_enabled)
                    {
                        g_print("[cuav][control][lost] zoom-out reached pt_focal_min=%.1f\n",
                                control_config->pt_focal_min);
                    }
                    return;
                }
                visible_sent = send_cuav_visible_light_command_with_en(appCtx,
                                                                        1,
                                                                        (guint)round(pt_focal),
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0);
                if (visible_sent)
                {
                    g_mutex_lock(&appCtx->cuav_control_lock);
                    appCtx->cuav_auto_control_state.last_visible_valid = TRUE;
                    appCtx->cuav_auto_control_state.last_pt_focal_en = 1;
                    appCtx->cuav_auto_control_state.last_pt_focal = pt_focal;
                    appCtx->cuav_auto_control_state.last_pt_focus = 0;
                    appCtx->cuav_auto_control_state.pending_pt_focal_valid = TRUE;
                    appCtx->cuav_auto_control_state.pending_pt_focal = pt_focal;
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
                        g_print("[cuav][control][lost] no target, zoom out focal_en=1 focal=%.1f min=%.1f step=%.1f\n",
                                pt_focal,
                                control_config->pt_focal_min,
                                control_config->pt_focal_step);
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
    if (appCtx->cuav_auto_control_state.pending_pt_focal_valid)
    {
        if (cuav_feedback_is_fresh(&feedback_snapshot,
                                   control_config->state_stale_timeout_ms) &&
            fabs(feedback_snapshot.pt_focal -
                 appCtx->cuav_auto_control_state.pending_pt_focal) <=
                CUAV_FOCAL_REACHED_EPSILON)
        {
            appCtx->cuav_auto_control_state.pending_pt_focal_valid = FALSE;
            appCtx->cuav_auto_control_state.last_pt_focal =
                feedback_snapshot.pt_focal;
            if (debug_enabled)
            {
                g_print("[cuav][control][auto] focal confirmed target=%.1f feedback=%.1f\n",
                        appCtx->cuav_auto_control_state.pending_pt_focal,
                        feedback_snapshot.pt_focal);
            }
        }
        else
        {
            visible_focal_pending = TRUE;
        }
    }
    state_snapshot = appCtx->cuav_auto_control_state;
    g_mutex_unlock(&appCtx->cuav_control_lock);

    if (cuav_visible_control_enabled(control_config) &&
        state_snapshot.lost_zoom_active)
    {
        g_mutex_lock(&appCtx->cuav_control_lock);
        appCtx->cuav_auto_control_state.last_pt_focal =
            cuav_get_current_pt_focal(control_config,
                                      &feedback_snapshot,
                                      &state_snapshot);
        appCtx->cuav_auto_control_state.pending_pt_focal_valid = FALSE;
        appCtx->cuav_auto_control_state.pending_pt_focal = 0.0;
        appCtx->cuav_auto_control_state.lost_zoom_active = FALSE;
        appCtx->cuav_auto_control_state.lost_zoom_start_us = 0;
        appCtx->cuav_auto_control_state.lost_zoom_hold_complete = FALSE;
        state_snapshot = appCtx->cuav_auto_control_state;
        visible_focal_pending = FALSE;
        g_mutex_unlock(&appCtx->cuav_control_lock);

        if (debug_enabled)
        {
            g_print("[cuav][control][lost] target reacquired, stop lost-target zoom-out target=%" G_GUINT64_FORMAT "\n",
                    sample.object_id);
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
        !visible_focal_pending &&
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
                                                                     &feedback_snapshot,
                                                                     &state_snapshot,
                                                                     &sample,
                                                                     &pt_focal_en,
                                                                     &pt_focal,
                                                                     &pt_focus);
        }
        visible_cmd_changed = should_send_visible &&
            (!state_snapshot.visible_initialized ||
             pt_focal_en != state_snapshot.last_pt_focal_en ||
             (pt_focal_en == 1 &&
              fabs(pt_focal - state_snapshot.last_pt_focal) >
                CUAV_FOCAL_REACHED_EPSILON));
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

    if (should_send_visible && (motion_spacing_ok || visible_stop_due))
    {
        visible_sent = send_cuav_visible_light_command_with_en(appCtx, pt_focal_en,
                                                                (guint)round(pt_focal),
                                                                0, 0, 0, 0);
        motion_cmd_sent = visible_sent;
        if (visible_sent && control_config->debug)
        {
            g_print("[cuav][control][auto] source=%u target=%" G_GUINT64_FORMAT
                    " ratio=%.3f focal_en=%u focal=%.1f focus=%u%s\n",
                    control_config->control_source_id,
                    sample.object_id,
                    sample.target_ratio,
                    pt_focal_en,
                    pt_focal,
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

    if (servo_sent || visible_sent)
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
            if (pt_focal_en == 1)
            {
                appCtx->cuav_auto_control_state.pending_pt_focal_valid = TRUE;
                appCtx->cuav_auto_control_state.pending_pt_focal = pt_focal;
            }
            else
            {
                appCtx->cuav_auto_control_state.pending_pt_focal_valid = FALSE;
                appCtx->cuav_auto_control_state.pending_pt_focal = 0.0;
            }
            appCtx->cuav_auto_control_state.last_visible_send_us = now_us;
            appCtx->cuav_auto_control_state.visible_initialized = TRUE;
            appCtx->cuav_auto_control_state.last_motion_send_us = now_us;
            appCtx->cuav_auto_control_state.last_motion_type = CUAV_MOTION_CMD_VISIBLE;
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
 * @brief 统一注册udpjsonmeta元素中的C-UAV协议解析回调。
 *
 * 将C-UAV解析回调集中在控制模块内部，避免主pipeline构建代码
 * 直接依赖各类协议报文处理函数。
 *
 * @param udpjsonmeta udpjsonmeta GStreamer元素。
 * @param appCtx 应用上下文，用作回调用户数据。
 */
void
register_cuav_udpjson_callbacks(GstElement *udpjsonmeta, AppCtx *appCtx)
{
    if (!udpjsonmeta)
        return;

    gst_udpjson_meta_set_guidance_callback(GST_UDPJSON_META(udpjsonmeta),
                                           on_cuav_guidance,
                                           appCtx);
    gst_udpjson_meta_set_eo_system_callback(GST_UDPJSON_META(udpjsonmeta),
                                            on_cuav_eo_system,
                                            appCtx);
    gst_udpjson_meta_set_servo_control_callback(GST_UDPJSON_META(udpjsonmeta),
                                                on_cuav_servo_control,
                                                appCtx);
}
