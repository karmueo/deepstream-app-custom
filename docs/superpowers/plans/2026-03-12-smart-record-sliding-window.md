# Smart Record 滑动窗口触发机制实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为智能录像功能添加滑动窗口机制，过滤瞬态误检触发录像

**Architecture:** 每个视频源维护独立的检测结果滑动窗口队列，当窗口内检测到目标的帧数占比超过阈值时才触发录像。复用现有 `g_pending_request` 防抖机制。

**Tech Stack:** C (GLib/GQueue), YAML/CSV 配置解析

**Spec:** `docs/superpowers/specs/2026-03-12-smart-record-sliding-window-design.md`

---

## Chunk 1: 数据结构与配置解析

### Task 1: 新增 NvDsSourceConfig 配置字段

**Files:**
- Modify: `apps-common/includes/deepstream_sources.h:92` (结构体末尾)

- [ ] **Step 1: 在 NvDsSourceConfig 结构体末尾添加新字段**

```c
  /** Video format to be applied at nvvideoconvert source pad. */
  gchar* video_format;
  /** 滑动窗口配置 */
  guint smart_rec_window_size;      /**< 滑动窗口大小（帧数），0=禁用直接触发 */
  gfloat smart_rec_trigger_ratio;   /**< 触发阈值比例 [0.0, 1.0] */
} NvDsSourceConfig;
```

- [ ] **Step 2: 提交**

```bash
git add apps-common/includes/deepstream_sources.h
git commit -m "feat(smart-record): 添加滑动窗口配置字段到 NvDsSourceConfig"
```

---

### Task 2: 新增配置解析宏定义

**Files:**
- Modify: `apps-common/includes/deepstream_config_file_parser.h:121` (smart-record 相关宏之后)

- [ ] **Step 1: 添加配置键名宏定义**

在 `#define CONFIG_GROUP_SOURCE_SMART_RECORD_RETENTION_DAYS` 之后添加：

```c
#define CONFIG_GROUP_SOURCE_SMART_RECORD_RETENTION_DAYS "smart-rec-retention-days"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_WINDOW_SIZE "smart-rec-window-size"
#define CONFIG_GROUP_SOURCE_SMART_RECORD_TRIGGER_RATIO "smart-rec-trigger-ratio"
```

- [ ] **Step 2: 提交**

```bash
git add apps-common/includes/deepstream_config_file_parser.h
git commit -m "feat(smart-record): 添加滑动窗口配置宏定义"
```

---

### Task 3: CSV 配置解析实现

**Files:**
- Modify: `apps-common/src/deepstream_config_file_parser.c` (parse_source 函数)

- [ ] **Step 1: 添加 window_size 解析逻辑**

在 `smart_rec_retention_days` 解析代码块之后添加：

```c
        else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_SMART_RECORD_WINDOW_SIZE))
        {
            config->smart_rec_window_size =
                g_key_file_get_integer(key_file, group,
                                       CONFIG_GROUP_SOURCE_SMART_RECORD_WINDOW_SIZE, &error);
            if (error)
            {
                config->smart_rec_window_size = 0;  // 默认禁用
                g_clear_error(&error);
            }
        }
```

- [ ] **Step 2: 添加 trigger_ratio 解析逻辑**

紧接上一步代码之后添加：

```c
        else if (!g_strcmp0(*key, CONFIG_GROUP_SOURCE_SMART_RECORD_TRIGGER_RATIO))
        {
            config->smart_rec_trigger_ratio =
                g_key_file_get_double(key_file, group,
                                      CONFIG_GROUP_SOURCE_SMART_RECORD_TRIGGER_RATIO, &error);
            if (error)
            {
                config->smart_rec_trigger_ratio = 0.8f;  // 默认 80%
                g_clear_error(&error);
            }
        }
```

- [ ] **Step 3: 提交**

```bash
git add apps-common/src/deepstream_config_file_parser.c
git commit -m "feat(smart-record): 添加 CSV 滑动窗口参数解析"
```

---

### Task 4: YAML 配置解析实现

**Files:**
- Modify: `apps-common/src/deepstream-yaml/deepstream_source_yaml.cpp:174` (smart-rec-retention-days 之后)

- [ ] **Step 1: 添加 window_size 解析逻辑**

在 `smart-rec-retention-days` 解析代码块之后添加：

```cpp
    } else if (paramKey == "smart-rec-window-size") {
      config->smart_rec_window_size = std::stoul(source_values[i]);
    } else if (paramKey == "smart-rec-trigger-ratio") {
      config->smart_rec_trigger_ratio = std::stof(source_values[i]);
```

- [ ] **Step 2: 提交**

```bash
git add apps-common/src/deepstream-yaml/deepstream_source_yaml.cpp
git commit -m "feat(smart-record): 添加 YAML 滑动窗口参数解析"
```

---

## Chunk 2: 核心逻辑实现

### Task 5: 新增 AppCtx 字段和源状态结构体

**Files:**
- Modify: `src/deepstream-app/deepstream_app.h:331` (AppCtx 结构体末尾，static_target_filter_states 之后)

- [ ] **Step 1: 定义源状态结构体**

在 `StaticTargetFilterState` 结构体定义之后、`struct _AppCtx` 之前添加：

```c
/**
 * @brief 单个视频源的滑动窗口检测状态
 *
 * 用于智能录像触发条件的滑动窗口确认机制。
 */
typedef struct {
    GQueue *detection_window;    /**< 滑动窗口队列，存储 gboolean 值 */
    guint detection_hit_count;   /**< 窗口内检测到目标的帧数 */
} SourceDetectionState;
```

- [ ] **Step 2: 在 AppCtx 结构体末尾添加字段**

在 `src/deepstream-app/deepstream_app.h` 第 331 行，找到 `static_target_filter_states` 字段：

```c
    /** 静止目标误检过滤状态 */
    StaticTargetFilterState static_target_filter_states[MAX_SOURCE_BINS]; /**< 各源静止目标过滤状态 */
};
```

在此字段之后、结构体闭合 `};` 之前添加新字段：

```c
    /** 静止目标误检过滤状态 */
    StaticTargetFilterState static_target_filter_states[MAX_SOURCE_BINS]; /**< 各源静止目标过滤状态 */

    /** 滑动窗口检测状态（用于智能录像触发） */
    guint num_source_states;               /**< 源状态数组长度，来自 config.num_source_bins */
    SourceDetectionState *source_states;   /**< 按源索引的滑动窗口状态数组 */
};
```

- [ ] **Step 3: 提交**

```bash
git add src/deepstream-app/deepstream_app.h
git commit -m "feat(smart-record): 添加滑动窗口状态结构体和 AppCtx 字段"
```

---

### Task 6: 实现滑动窗口初始化函数

**Files:**
- Modify: `src/deepstream-app/deepstream_app_main.c`

- [ ] **Step 1: 在文件顶部包含区域后添加初始化函数**

在 `#define` 和全局变量定义之后，添加滑动窗口相关函数：

```c
/**
 * @brief 初始化所有视频源的滑动窗口状态
 *
 * @param appCtx 应用上下文
 * @param num_sources 视频源数量
 */
static void
init_source_detection_states(AppCtx *appCtx, guint num_sources)
{
    if (appCtx->source_states != NULL)
    {
        // 已初始化，先清理
        cleanup_source_detection_states(appCtx);
    }

    appCtx->num_source_states = num_sources;
    appCtx->source_states = g_new0(SourceDetectionState, num_sources);

    for (guint i = 0; i < num_sources; i++)
    {
        appCtx->source_states[i].detection_window = g_queue_new();
        appCtx->source_states[i].detection_hit_count = 0;
    }

    GST_INFO("Initialized sliding window detection states for %u sources", num_sources);
}

/**
 * @brief 清理所有视频源的滑动窗口状态
 *
 * @param appCtx 应用上下文
 */
static void
cleanup_source_detection_states(AppCtx *appCtx)
{
    if (appCtx->source_states == NULL)
        return;

    for (guint i = 0; i < appCtx->num_source_states; i++)
    {
        if (appCtx->source_states[i].detection_window != NULL)
        {
            g_queue_free(appCtx->source_states[i].detection_window);
            appCtx->source_states[i].detection_window = NULL;
        }
    }

    g_free(appCtx->source_states);
    appCtx->source_states = NULL;
    appCtx->num_source_states = 0;

    GST_INFO("Cleaned up sliding window detection states");
}
```

- [ ] **Step 2: 提交**

```bash
git add src/deepstream-app/deepstream_app_main.c
git commit -m "feat(smart-record): 添加滑动窗口初始化和清理函数"
```

---

### Task 7: 实现滑动窗口更新和触发判断函数

**Files:**
- Modify: `src/deepstream-app/deepstream_app_main.c`

- [ ] **Step 1: 在清理函数之后添加滑动窗口更新函数**

```c
/**
 * @brief 更新滑动窗口并判断是否应触发录像
 *
 * @param appCtx 应用上下文
 * @param source_id 视频源 ID
 * @param config 源配置
 * @param current_has_target 当前帧是否有检测目标
 * @return gboolean 是否应触发录像
 */
static gboolean
update_sliding_window_and_check_trigger(AppCtx *appCtx,
                                         guint source_id,
                                         NvDsSourceConfig *config,
                                         gboolean current_has_target)
{
    // 如果未配置滑动窗口，直接返回当前检测结果
    guint window_size = config->smart_rec_window_size;
    if (window_size == 0)
    {
        return current_has_target;
    }

    // 边界检查
    if (source_id >= appCtx->num_source_states ||
        appCtx->source_states == NULL)
    {
        GST_WARNING("Invalid source_id %u or states not initialized", source_id);
        return FALSE;
    }

    SourceDetectionState *state = &appCtx->source_states[source_id];

    // 更新滑动窗口
    // 如果队列已满，移除最旧元素
    if (g_queue_get_length(state->detection_window) >= window_size)
    {
        gboolean oldest = GPOINTER_TO_INT(g_queue_pop_head(state->detection_window));
        if (oldest)
        {
            state->detection_hit_count--;
        }
    }

    // 入队新结果
    g_queue_push_tail(state->detection_window,
                      GINT_TO_POINTER(current_has_target));
    if (current_has_target)
    {
        state->detection_hit_count++;
    }

    // 判断是否满足触发条件
    if (g_queue_get_length(state->detection_window) >= window_size)
    {
        gfloat hit_ratio = (gfloat)state->detection_hit_count / (gfloat)window_size;
        gfloat trigger_ratio = config->smart_rec_trigger_ratio;

        if (hit_ratio >= trigger_ratio)
        {
            GST_DEBUG("Source %u: hit_ratio=%.2f >= trigger_ratio=%.2f, triggering recording",
                      source_id, hit_ratio, trigger_ratio);
            return TRUE;
        }
    }

    return FALSE;
}
```

- [ ] **Step 2: 提交**

```bash
git add src/deepstream-app/deepstream_app_main.c
git commit -m "feat(smart-record): 添加滑动窗口更新和触发判断函数"
```

---

### Task 8: 集成滑动窗口到检测触发逻辑

**Files:**
- Modify: `src/deepstream-app/deepstream_app_main.c` (tiled_display_sink_pad_buffer_probe 函数)

**设计说明：** 滑动窗口更新逻辑集中在帧级别处理，确保每帧只更新一次。SOT 模式保持原有 3 秒连续跟踪确认机制不变，无跟踪模式和 MOT 模式使用滑动窗口。

- [ ] **Step 1: 找到帧级滑动窗口更新位置**

在 `tiled_display_sink_pad_buffer_probe` 函数中，找到 `for (l = frame_meta->obj_meta_list; ...)` 循环结束后（约第 850 行附近），SOT 跟踪丢失处理逻辑之前的位置。

- [ ] **Step 2: 添加帧级滑动窗口更新逻辑**

在目标循环结束后、SOT 跟踪丢失逻辑之前，添加：

```c
        /* ===== 滑动窗口检测状态更新（无跟踪/MOT 模式） ===== */
        /* SOT 模式使用原有的 3 秒连续跟踪确认机制，不走滑动窗口 */
        if (!single_object_tracker &&
            (src_bin->config->smart_record == 2 || src_bin->config->smart_record == 3) &&
            is_detect_record_enabled(appCtx) &&
            src_bin->config->smart_rec_window_size > 0)
        {
            /* 判断当前帧是否有检测目标 */
            gboolean frame_has_target = FALSE;
            for (GList *l = frame_meta->obj_meta_list; l != NULL; l = l->next)
            {
                NvDsObjectMeta *obj = (NvDsObjectMeta *)l->data;
                if (has_detection_target(obj))
                {
                    frame_has_target = TRUE;
                    break;
                }
            }

            /* 更新滑动窗口并检查是否触发 */
            gboolean should_trigger = update_sliding_window_and_check_trigger(
                appCtx, stream_id, src_bin->config, frame_has_target);

            /* 触发录像（带防抖） */
            if (should_trigger && !g_pending_request)
            {
                /* 获取目标标签用于录像文件命名 */
                const gchar *class_label = "unknown";
                for (GList *l = frame_meta->obj_meta_list; l != NULL; l = l->next)
                {
                    NvDsObjectMeta *obj = (NvDsObjectMeta *)l->data;
                    if (has_detection_target(obj))
                    {
                        if (obj->obj_label[0] != '\0')
                        {
                            class_label = obj->obj_label;
                        }
                        break;
                    }
                }

                g_pending_request = TRUE;
                smart_record_event_generator(src_bin, class_label);
                GST_INFO("Source %u: Sliding window triggered recording, label=%s",
                         stream_id, class_label);
            }
        }
```

- [ ] **Step 3: 移除原有的无跟踪模式直接触发逻辑**

找到并注释或删除原有的无跟踪模式触发逻辑（约第 786-790 行）：

```c
                else
                {
                    /* 未启用单目标跟踪，保持原有逻辑：直接触发 */
                    should_trigger_recording = TRUE;
                }
```

替换为：

```c
                else
                {
                    /* 未启用单目标跟踪：滑动窗口逻辑在帧级别处理（见下方） */
                    /* 此处不设置 should_trigger_recording，避免重复触发 */
                }
```

- [ ] **Step 4: 提交**

```bash
git add src/deepstream-app/deepstream_app_main.c
git commit -m "feat(smart-record): 集成滑动窗口到检测触发逻辑"
```

---

### Task 9: 添加函数声明到头文件

**Files:**
- Modify: `src/deepstream-app/deepstream_app.h` (AppCtx 结构体之后)

- [ ] **Step 1: 在 deepstream_app.h 中添加函数声明**

在 `struct _AppCtx` 定义之后，`create_pipeline` 函数声明之前添加：

```c
/**
 * @brief 初始化所有视频源的滑动窗口检测状态
 *
 * @param appCtx 应用上下文
 * @param num_sources 视频源数量
 */
void init_source_detection_states(AppCtx *appCtx, guint num_sources);

/**
 * @brief 清理所有视频源的滑动窗口检测状态
 *
 * @param appCtx 应用上下文
 */
void cleanup_source_detection_states(AppCtx *appCtx);
```

- [ ] **Step 2: 移除 deepstream_app_main.c 中函数的 static 关键字**

在 `deepstream_app_main.c` 中，将 `init_source_detection_states` 和 `cleanup_source_detection_states` 函数的 `static` 关键字移除。

将：
```c
static void
init_source_detection_states(AppCtx *appCtx, guint num_sources)
```

改为：
```c
void
init_source_detection_states(AppCtx *appCtx, guint num_sources)
```

同样处理 `cleanup_source_detection_states` 函数。

- [ ] **Step 3: 提交**

```bash
git add src/deepstream-app/deepstream_app.h src/deepstream-app/deepstream_app_main.c
git commit -m "feat(smart-record): 添加滑动窗口函数声明到头文件"
```

---

### Task 10: 在流水线创建和销毁时管理滑动窗口

**Files:**
- Modify: `src/deepstream-app/deepstream_app.c` (create_pipeline 和 destroy_pipeline 函数)

- [ ] **Step 1: 在 create_pipeline 中初始化**

在 `create_pipeline` 函数中，找到返回 TRUE 之前的位置，添加：

```c
    /* 初始化滑动窗口检测状态 */
    init_source_detection_states(appCtx, appCtx->config.num_source_bins);
```

- [ ] **Step 2: 在 destroy_pipeline 中清理**

在 `destroy_pipeline` 函数开头，添加：

```c
    /* 清理滑动窗口检测状态 */
    cleanup_source_detection_states(appCtx);
```

- [ ] **Step 3: 提交**

```bash
git add src/deepstream-app/deepstream_app.c
git commit -m "feat(smart-record): 在流水线生命周期中管理滑动窗口"
```

---

## Chunk 3: 配置示例与编译验证

### Task 11: 更新配置文件示例

**Files:**
- Modify: `src/deepstream-app/configs/yml/sources.csv`

- [ ] **Step 1: 添加新配置列到 CSV 头部**

将第一行修改为：

```csv
enable,type,uri,num-sources,gpu-id,cudadec-memtype,rtsp-reconnect-interval-sec,rtsp-reconnect-attempts,select-rtp-protocol,smart-record,smart-rec-dir-path,smart-rec-duration,smart-rec-start-time,smart-rec-window-size,smart-rec-trigger-ratio
```

- [ ] **Step 2: 为启用的源添加示例配置值**

将第二行修改为（添加 `30,0.8` 作为示例，使用相对路径）：

```csv
1,4,rtsp://192.168.1.80/live/test,1,0,0,3,-1,4,2,../recordings/1,30,3,30,0.8
```

- [ ] **Step 3: 为其他源添加默认值（可选）**

```csv
0,4,rtsp://192.168.1.80/live/test1,1,0,0,3,-1,0,2,../recordings/2,30,3,0,0.8
0,4,rtsp://192.168.1.80/live/test2,1,0,0,3,-1,0,2,../recordings/3,30,3,0,0.8
0,4,rtsp://192.168.1.80/live/test3,1,0,0,3,-1,0,2,../recordings/4,30,3,0,0.8
```

- [ ] **Step 4: 提交**

```bash
git add src/deepstream-app/configs/yml/sources.csv
git commit -m "feat(smart-record): 更新 sources.csv 配置示例"
```

---

### Task 12: 编译验证

- [ ] **Step 1: 编译项目**

```bash
cd /home/nvidia/work/deepstream-app-custom/src/deepstream-app && make
```

预期输出：编译成功，无错误。退出码为 0。

- [ ] **Step 2: 验证编译结果**

```bash
ls -la build/deepstream-app
```

预期：可执行文件存在且时间戳已更新

- [ ] **Step 3: 运行基本功能测试**

```bash
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$GST_PLUGIN_PATH
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
./build/deepstream-app -c configs/yml/app_config.yml
```

预期行为：
- 应用正常启动
- 无段错误
- 日志中可见 "Initialized sliding window detection states for X sources" 消息
- 按 Ctrl+C 可正常退出

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "feat(smart-record): 完成滑动窗口触发机制实现

- 新增 smart-rec-window-size 和 smart-rec-trigger-ratio 配置参数
- 每个视频源维护独立的滑动窗口队列
- 支持多视频源独立配置
- 向后兼容，默认禁用滑动窗口

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 验证清单

- [ ] 编译无错误无警告
- [ ] 应用正常启动和退出
- [ ] 滑动窗口状态正确初始化和清理
- [ ] 配置参数正确解析
- [ ] 无目标时不触发录像
- [ ] 持续检测到目标超过阈值时触发录像
- [ ] 短暂误检（< 阈值帧数）不触发录像
- [ ] 多源独立运行互不干扰
- [ ] 不配置新参数时保持原有行为
