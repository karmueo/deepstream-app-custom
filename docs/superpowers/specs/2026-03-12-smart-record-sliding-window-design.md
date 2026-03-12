# Smart Record 滑动窗口触发机制设计

## 问题描述

当 `sources.csv` 中 `smart-record` 设置为 2 或 3 时，在无跟踪模式下，单帧误检就会触发录像，导致大量无效录像文件。

### 当前行为

| smart_record | 行为 |
|--------------|------|
| 0 | 禁用智能录像 |
| 1 | 启用智能录像，需要外部触发 |
| 2 | 检测到目标时触发录像 |
| 3 | 检测到目标时触发录像 |

### 问题根因

无跟踪模式下，`has_detection_target()` 只要单帧满足条件（置信度 >= 0.5，obj_label 非空，bbox 有效）就直接触发录像，缺乏连续性确认机制。

## 解决方案

### 滑动窗口机制

为每个视频源维护一个检测结果滑动窗口队列，只有当窗口内检测到目标的帧数占比超过阈值时，才触发录像。

```
┌─────────────────────────────────────────────────────────────┐
│                     滑动窗口示意                             │
├─────────────────────────────────────────────────────────────┤
│  窗口大小: 30 帧, 触发阈值: 0.8                              │
│                                                             │
│  [1][1][1][0][1][1][1][1][0][1][1][1][1][1][1][0][1][1]...  │
│   ↑  最近 30 帧检测结果 (1=有目标, 0=无目标)                 │
│                                                             │
│  命中帧数: 24, 命中率: 24/30 = 0.8 >= 0.8 → 触发录像        │
└─────────────────────────────────────────────────────────────┘
```

### 优势

1. **过滤瞬态误检**：偶尔一两帧的误检不会触发录像
2. **容忍短暂丢失**：目标短暂消失（如遮挡）不会立即重置
3. **灵活可配置**：不同视频源可设置不同参数
4. **帧率无关**：按帧数配置，适应不同帧率场景

## 配置参数

### CSV 新增列

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `smart-rec-window-size` | int | 0 | 滑动窗口大小（帧数），0 表示禁用滑动窗口直接触发 |
| `smart-rec-trigger-ratio` | float | 0.8 | 触发阈值比例，范围 [0.0, 1.0] |

### 配置示例

```csv
enable,type,uri,num-sources,gpu-id,cudadec-memtype,rtsp-reconnect-interval-sec,rtsp-reconnect-attempts,select-rtp-protocol,smart-record,smart-rec-dir-path,smart-rec-duration,smart-rec-start-time,smart-rec-window-size,smart-rec-trigger-ratio
1,4,rtsp://192.168.1.80/live/test,1,0,0,3,-1,4,2,/home/nvidia/recordings,30,3,30,0.8
1,4,rtsp://192.168.1.80/live/test1,1,0,0,3,-1,4,2,/home/nvidia/recordings,30,3,50,0.9
```

### 参数说明

- `smart-rec-window-size = 30`：最近 30 帧的检测结果参与判断
- `smart-rec-trigger-ratio = 0.8`：30 帧中有 24 帧检测到目标才触发
- `smart-rec-window-size = 0`：禁用滑动窗口，保持原有直接触发行为

## 数据结构

### NvDsSourceConfig 新增字段

**文件：** `apps-common/includes/deepstream_sources.h`

```c
typedef struct _NvDsSourceConfig {
    // ... 现有字段 ...
    guint smart_rec_window_size;      // 滑动窗口大小，0=禁用
    gfloat smart_rec_trigger_ratio;   // 触发阈值比例
} NvDsSourceConfig;
```

### 源状态结构体

**文件：** `src/deepstream-app/deepstream_app_main.c`

```c
/**
 * @brief 单个视频源的滑动窗口检测状态
 */
typedef struct {
    GQueue *detection_window;    // 滑动窗口队列，存储 gboolean 值
    guint detection_hit_count;   // 窗口内检测到目标的帧数
} SourceDetectionState;
```

### AppCtx 新增字段

```c
typedef struct _AppCtx {
    // ... 现有字段 ...
    SourceDetectionState *source_states;  // 按源索引的状态数组
} AppCtx;
```

## 核心逻辑

### 初始化

```c
// 应用启动时，根据源数量分配状态数组
ctx->source_states = g_new0(SourceDetectionState, num_sources);
for (guint i = 0; i < num_sources; i++) {
    ctx->source_states[i].detection_window = g_queue_new();
    ctx->source_states[i].detection_hit_count = 0;
}
```

### 帧处理流程

```c
// 1. 获取当前源的状态和配置
guint source_id = frame_meta->source_id;
SourceDetectionState *state = &appCtx->source_states[source_id];
NvDsSourceConfig *config = src_bin->config;

// 2. 判断当前帧是否有检测目标
gboolean current_has_target = FALSE;
for (GList *l = frame_meta->obj_meta_list; l != NULL; l = l->next) {
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l->data;
    if (has_detection_target(obj_meta)) {
        current_has_target = TRUE;
        break;
    }
}

// 3. 更新滑动窗口
guint window_size = config->smart_rec_window_size;
if (window_size > 0) {
    // 队列满则移除最旧元素
    if (g_queue_get_length(state->detection_window) >= window_size) {
        gboolean oldest = GPOINTER_TO_INT(g_queue_pop_head(state->detection_window));
        if (oldest) {
            state->detection_hit_count--;
        }
    }
    // 入队新结果
    g_queue_push_tail(state->detection_window,
                      GINT_TO_POINTER(current_has_target));
    if (current_has_target) {
        state->detection_hit_count++;
    }
}

// 4. 判断是否触发录像
gboolean should_trigger = FALSE;
if (window_size > 0) {
    // 滑动窗口模式：检查命中率
    if (g_queue_get_length(state->detection_window) >= window_size) {
        gfloat hit_ratio = (gfloat)state->detection_hit_count / window_size;
        if (hit_ratio >= config->smart_rec_trigger_ratio) {
            should_trigger = TRUE;
        }
    }
} else {
    // 原有逻辑：直接触发
    should_trigger = current_has_target;
}

// 5. 触发录像（带防抖）
if (should_trigger && !g_pending_request) {
    // 调用 smart_record_event_generator 触发录像
    // ...
}
```

### 清理

```c
// 应用退出时释放资源
if (ctx->source_states) {
    for (guint i = 0; i < num_sources; i++) {
        if (ctx->source_states[i].detection_window) {
            g_queue_free(ctx->source_states[i].detection_window);
        }
    }
    g_free(ctx->source_states);
}
```

## 涉及文件

| 文件 | 修改内容 |
|------|----------|
| `apps-common/includes/deepstream_sources.h` | `NvDsSourceConfig` 新增配置字段 |
| `apps-common/includes/deepstream_config_file_parser.h` | 新增配置字段声明 |
| `apps-common/src/deepstream_config_file_parser.c` | CSV 解析新配置项 |
| `apps-common/src/deepstream-yaml/deepstream_source_yaml.cpp` | YAML 解析新配置项 |
| `src/deepstream-app/deepstream_app_main.c` | 核心逻辑、状态管理、初始化/清理 |
| `src/deepstream-app/configs/yml/sources.csv` | 新增配置列（示例） |

## 触发条件完整逻辑

### 无跟踪模式

```
检测到目标 (has_detection_target)
    ↓
更新滑动窗口
    ↓
window_size > 0?
    ├─ 是 → 计算命中率 → hit_ratio >= trigger_ratio → 触发录像
    └─ 否 → 直接触发录像（原有行为）
```

### 单目标跟踪模式（SOT）

不受影响，继续使用原有的 3 秒连续跟踪确认机制。

### 多目标跟踪模式（MOT）

采用与无跟踪模式相同的滑动窗口机制。

## 向后兼容

1. `smart-rec-window-size` 默认值为 0，不配置时保持原有直接触发行为
2. 现有 `sources.csv` 无需修改，新列为可选
3. SOT 模式行为不变

## 验证方法

1. **基本功能**
   - 设置 `smart-rec-window-size=30, smart-rec-trigger-ratio=0.8`
   - 确认无目标时不产生录像
   - 确认持续检测到目标（超过阈值）时产生录像

2. **误检过滤**
   - 模拟瞬态误检（1-5 帧误检），确认不触发录像
   - 确认短暂目标丢失（如遮挡 3-5 帧）不中断录像触发

3. **多源独立**
   - 配置多个视频源，各自设置不同参数
   - 确认各源独立触发，互不影响

4. **向后兼容**
   - 不配置新参数，确认行为与修改前一致

## 风险评估

- **改动范围**：中等，涉及 5 个文件
- **向后兼容**：是，新参数默认值保持原有行为
- **性能影响**：极小，每个源仅维护一个小的队列结构
- **测试建议**：重点验证多源场景和边界条件
