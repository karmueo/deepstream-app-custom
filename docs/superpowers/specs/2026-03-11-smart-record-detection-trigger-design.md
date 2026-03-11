# Smart Record 检测触发录像修复设计

## 问题描述

当 `sources.csv` 中 `smart-record` 设置为 2 时，期望行为是只在检测到目标时存录像，但实际行为是无条件一直录像。

## 根因分析

### 当前代码逻辑

1. **`apps-common/src/deepstream_source_bin.c:1452-1462`**
   - 当 `smart_record == 2` 时，启动定时器调用 `smart_record_event_generator`
   - 该定时器不检查是否有目标检测到，只是周期性开关录像

2. **`src/deepstream-app/deepstream_app_main.c:749`**
   - 检测触发录像的条件是 `smart_record == 3`
   - 当检测到目标且满足条件时调用 `smart_record_event_generator`

### 当前行为

| smart_record | 行为 |
|--------------|------|
| 0 | 禁用智能录像 |
| 1 | 启用智能录像，需要外部触发 |
| 2 | 启动定时器，无条件周期性开关录像 |
| 3 | 只在检测到目标时触发录像 |

## 解决方案

### 方案 A：合并 2 和 3 的行为

将 `smart_record == 2` 的行为改为与 `smart_record == 3` 相同：只在检测到目标时触发录像。

### 修改内容

#### 1. 移除定时器逻辑

**文件：** `apps-common/src/deepstream_source_bin.c`

**位置：** 第 1452-1462 行

**删除代码：**
```c
// Enable local start / stop events in addition to the one
// received from the server.
if (config->smart_record == 2)
{
    if (bin->config->smart_rec_interval)
        g_timeout_add(bin->config->smart_rec_interval * 1000,
                      smart_record_event_generator, bin);
    else
        g_timeout_add(10000, smart_record_event_generator, bin);
}
```

#### 2. 扩展检测触发条件

**文件：** `src/deepstream-app/deepstream_app_main.c`

**位置：** 第 749 行

**修改前：**
```c
if (src_bin->config->smart_record == 3 &&
```

**修改后：**
```c
if ((src_bin->config->smart_record == 2 || src_bin->config->smart_record == 3) &&
```

### 行为变化

| smart_record | 修改前 | 修改后 |
|--------------|--------|--------|
| 2 | 定时器无条件周期录像 | 检测到目标时触发录像 |
| 3 | 检测到目标时触发录像 | 检测到目标时触发录像（不变）|

### 触发条件

检测触发录像需要同时满足以下条件：
1. `smart_record` 为 2 或 3
2. `is_detect_record_enabled(appCtx)` 返回 TRUE
3. `has_detection_target(obj_meta)` 返回 TRUE（目标置信度 >= 0.5，且有标签或类别 ID 为 0）
4. 如果启用了单目标跟踪，需要连续跟踪同一目标超过 3 秒

## 验证方法

1. 设置 `sources.csv` 中 `smart-record` 为 2
2. 运行应用，确保没有检测到目标时不产生录像文件
3. 检测到目标后，确认录像文件被创建
4. 确认录像文件存储在配置的目录中

## 风险评估

- **改动范围：** 小，仅涉及 2 个文件、2 处修改
- **向后兼容：** 是，`smart_record == 3` 行为不变
- **测试建议：** 验证 smart-record 值为 2 和 3 时的录像行为
