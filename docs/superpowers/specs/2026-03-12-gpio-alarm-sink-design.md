# GPIO 报警 Sink 插件设计文档

## 概述

创建 `gst-gpio-alarm-sink` GStreamer 插件，实现检测到指定类别目标时通过 GPIO 闪烁报警。

## 需求

| 项目 | 需求 |
|------|------|
| 触发条件 | 按类别过滤（指定 class_id 列表） |
| 输出形式 | GPIO 信号 |
| 信号模式 | 闪烁 |
| 闪烁频率 | 可配置 |
| GPIO 引脚 | 可配置 |
| 防抖机制 | 触发防抖（连续 N 帧后触发）+ 消失防抖（连续 M 帧消失后停止） |
| 防抖帧数 | 可配置 |

## 架构设计

### 流水线位置

```
视频源 → NVDEC → 流复用器 → 主推理 → 跟踪 → ... → gpio-alarm-sink
                                                         ↓
                                                    GPIO 闪烁输出
```

### 核心组件

| 组件 | 职责 |
|------|------|
| `GstGPIOAlarmSink` | GStreamer sink 元素，处理 buffer 元数据 |
| `GPIOController` | GPIO 控制器，管理闪烁逻辑和防抖 |

### 多路视频源支持

当存在多路视频源时，GPIO 报警行为如下：
- **单 GPIO 模式**：任意一路检测到目标即触发报警（逻辑或）
- **源 ID 过滤**：通过配置 `alarm-source-ids` 限定触发报警的源（如 "0,2" 表示仅第 0 和第 2 路触发）
- 默认行为：所有源均可触发报警

## 模块设计

### 1. GPIOController 类

负责 GPIO 硬件控制和闪烁逻辑。

```cpp
// gpio_controller.h

#include <atomic>
#include <thread>
#include <fstream>
#include <string>

class GPIOController {
public:
    GPIOController(int pin, float freqHz, float dutyCycle,
                   int debounceOnFrames, int debounceOffFrames);
    ~GPIOController();

    // 更新检测状态（每帧调用，线程安全）
    void updateDetectionState(bool hasTarget);

    // 启动/停止闪烁线程
    bool start();  // 返回 false 表示 GPIO 初始化失败
    void stop();

private:
    int m_pin;                  // GPIO 引脚号
    float m_freqHz;             // 闪烁频率 (Hz)
    float m_dutyCycle;          // 占空比 (0.0-1.0)
    int m_debounceOnFrames;     // 触发防抖帧数
    int m_debounceOffFrames;    // 消失防抖帧数

    // 原子变量保证线程安全
    std::atomic<int> m_consecutiveOnFrames{0};   // 连续检测帧计数
    std::atomic<int> m_consecutiveOffFrames{0};  // 连续消失帧计数
    std::atomic<bool> m_isAlarming{false};       // 当前是否在报警状态
    std::atomic<bool> m_running{false};          // 线程运行标志
    std::thread m_blinkThread;                   // 闪烁线程

    // GPIO 文件描述符（保持打开以提高性能）
    std::ofstream m_gpioValueFile;

    bool exportGPIO();          // 导出 GPIO
    bool setGPIODirection();    // 设置为输出模式
    void blinkLoop();           // 闪烁循环
    void setGPIO(bool high);    // 设置 GPIO 电平
};
```

#### 状态机

```
[空闲] --连续N帧检测到目标--> [报警中]
[报警中] --连续M帧目标消失--> [空闲]
```

- 空闲状态：GPIO 保持低电平
- 报警状态：独立线程按频率闪烁 GPIO

#### GPIO 操作与权限

使用 sysfs 方式操作 GPIO：

**初始化流程**：
1. 检查 `/sys/class/gpio/gpio{pin}` 是否已导出
2. 若未导出，写入 `{pin}` 到 `/sys/class/gpio/export`
3. 等待 sysfs 文件系统就绪（最多 100ms）
4. 设置 `direction` 为 `out`
5. 打开 `value` 文件并保持文件描述符

**权限要求**：
- 需要 sudo 权限运行，或
- 将用户加入 `gpio` 用户组（需系统配置）

```cpp
bool GPIOController::exportGPIO() {
    std::string gpioPath = "/sys/class/gpio/gpio" + std::to_string(m_pin);

    // 检查是否已导出
    struct stat st;
    if (stat(gpioPath.c_str(), &st) == 0) {
        return true;  // 已导出
    }

    // 导出 GPIO
    std::ofstream exportFile("/sys/class/gpio/export");
    if (!exportFile) {
        g_error("Failed to export GPIO %d: %s", m_pin, strerror(errno));
        return false;
    }
    exportFile << m_pin;

    // 等待 sysfs 就绪
    for (int i = 0; i < 10; i++) {
        usleep(10000);  // 10ms
        if (stat(gpioPath.c_str(), &st) == 0) {
            break;
        }
    }

    // 设置方向为输出
    std::ofstream dirFile(gpioPath + "/direction");
    dirFile << "out";

    // 打开 value 文件并保持
    m_gpioValueFile.open(gpioPath + "/value");
    return m_gpioValueFile.good();
}

void GPIOController::setGPIO(bool high) {
    m_gpioValueFile.seekp(0);
    m_gpioValueFile << (high ? "1" : "0");
    m_gpioValueFile.flush();
}
```

### 2. GstGPIOAlarmSink 插件

GStreamer sink 元素，从 buffer 元数据中提取检测结果。

```cpp
// gstgpio_alarm_sink.h

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <set>
#include <string>

G_BEGIN_DECLS

#define GST_TYPE_GPIO_ALARM_SINK (gst_gpio_alarm_sink_get_type())
G_DECLARE_FINAL_TYPE(GstGPIOAlarmSink, gst_gpio_alarm_sink,
                     GST, GPIO_ALARM_SINK, GstBaseSink)

struct _GstGPIOAlarmSink {
    GstBaseSink parent;

    // 配置属性
    gint gpio_pin;              // GPIO 引脚
    gfloat blink_freq;          // 闪烁频率
    gfloat duty_cycle;          // 占空比
    gint debounce_on_frames;    // 触发防抖帧数
    gint debounce_off_frames;   // 消失防抖帧数
    gchar *alarm_class_ids;     // 类别ID字符串 (如 "0,1,2")
    gchar *alarm_source_ids;    // 源ID字符串 (如 "0,2"，NULL 表示所有源)

    // 运行时状态
    GPIOController *controller;
    std::set<gint> m_classIdSet;   // 解析后的类别ID集合
    std::set<gint> m_sourceIdSet;  // 解析后的源ID集合（空表示所有源）
};

G_END_DECLS
```

#### 属性定义

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gpio-pin` | int | 216 | GPIO 引脚号（Jetson Orin NX 默认值，其他平台需调整） |
| `blink-freq` | float | 2.0 | 闪烁频率 (Hz)，建议范围 0.5-10.0 |
| `duty-cycle` | float | 0.5 | 占空比 (0.0-1.0) |
| `debounce-on-frames` | int | 10 | 触发防抖帧数（连续 N 帧检测到目标才触发） |
| `debounce-off-frames` | int | 5 | 消失防抖帧数（连续 M 帧消失才停止） |
| `alarm-class-ids` | string | "0" | 触发类别 (逗号分隔) |
| `alarm-source-ids` | string | NULL | 触发源 (逗号分隔，NULL 表示所有源) |

#### 核心回调函数

```cpp
// 启动时初始化 GPIO
static gboolean gst_gpio_alarm_sink_start(GstBaseSink *sink);

// 停止时清理 GPIO
static gboolean gst_gpio_alarm_sink_stop(GstBaseSink *sink);

// 每帧处理
static GstFlowReturn gst_gpio_alarm_sink_render(
    GstBaseSink *sink, GstBuffer *buffer);
```

#### render 函数逻辑

```cpp
GstFlowReturn gst_gpio_alarm_sink_render(GstBaseSink *sink, GstBuffer *buf) {
    GstGPIOAlarmSink *self = GST_GPIO_ALARM_SINK(sink);

    // 1. 获取 DeepStream 元数据
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_FLOW_OK;
    }

    // 2. 检查是否有目标类别
    bool hasTarget = false;
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // 检查源 ID 过滤
        if (!self->m_sourceIdSet.empty() &&
            self->m_sourceIdSet.count(frame_meta->source_id) == 0) {
            continue;
        }

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;

            if (self->m_classIdSet.count(obj->class_id) > 0) {
                hasTarget = true;
                break;
            }
        }
        if (hasTarget) break;
    }

    // 3. 更新 GPIO 控制器状态
    self->controller->updateDetectionState(hasTarget);

    return GST_FLOW_OK;
}
```

## 配置集成

### 枚举定义

在 `apps-common/includes/deepstream_sinks.h` 的 `NvDsSinkType` 枚举中添加：

```c
typedef enum {
    NV_DS_SINK_FAKE = 1,
    NV_DS_SINK_RENDER_EGL,
    NV_DS_SINK_ENCODE_FILE,
    NV_DS_SINK_UDPSINK,
    NV_DS_SINK_RENDER_DRM,
    NV_DS_SINK_MSG_CONV_BROKER,
    NV_DS_SINK_MYNETWORK,
    NV_DS_SINK_GPIO_ALARM = 8   // 新增：GPIO 报警 sink
} NvDsSinkType;
```

### 配置文件结构

在 `app_config.yml` 中新增 sink 配置：

```yaml
sink3:
  enable: 1
  type: 8  # NV_DS_SINK_GPIO_ALARM
  # GPIO 配置
  gpio-pin: 216
  blink-freq: 2.0
  duty-cycle: 0.5
  debounce-on-frames: 10
  debounce-off-frames: 5
  alarm-class-ids: "0"
  alarm-source-ids: null  # 可选，null 表示所有源
```

### 配置解析修改

在 `apps-common/includes/deepstream_sinks.h` 的 `NvDsSinkSubBinConfig` 结构体中添加字段：

```c
typedef struct {
    // ... 现有字段 ...

    // GPIO Alarm Sink 配置
    guint gpio_pin;
    gfloat blink_freq;
    gfloat duty_cycle;
    guint debounce_on_frames;
    guint debounce_off_frames;
    gchar *alarm_class_ids;
    gchar *alarm_source_ids;
} NvDsSinkSubBinConfig;
```

在 `deepstream_app_config_parser.c` 中添加 YAML 解析逻辑：

```c
// parse_sink_bin_config() 函数中添加
if (g_key_file_has_key(key_file, group, "gpio-pin", NULL)) {
    config->gpio_pin = g_key_file_get_integer(key_file, group, "gpio-pin", NULL);
}
if (g_key_file_has_key(key_file, group, "blink-freq", NULL)) {
    config->blink_freq = g_key_file_get_double(key_file, group, "blink-freq", NULL);
}
// ... 其他字段类似
```

### 流水线构建修改

在 `deepstream_app.c` 的 `create_sink_bin` 函数中添加：

```c
case NV_DS_SINK_GPIO_ALARM:  // 8
{
    GstElement *gpio_sink = gst_element_factory_make("gpioalarmsink", NULL);
    if (!gpio_sink) {
        NVGSTDS_ERR_MSG_V("Failed to create gpioalarmsink element");
        return FALSE;
    }
    g_object_set(gpio_sink,
        "gpio-pin", config->gpio_pin,
        "blink-freq", config->blink_freq,
        "duty-cycle", config->duty_cycle,
        "debounce-on-frames", config->debounce_on_frames,
        "debounce-off-frames", config->debounce_off_frames,
        "alarm-class-ids", config->alarm_class_ids,
        "alarm-source-ids", config->alarm_source_ids,
        NULL);
    gst_bin_add(GST_BIN(sink_bin), gpio_sink);
    // 连接到 tee（参考其他 sink 的实现）
    ...
    break;
}
```

### 插件注册

在 `deepstream_app_main.c` 中添加插件初始化：

```c
// 插件初始化声明
extern gboolean gst_gpio_alarm_sink_plugin_init(GstPlugin *plugin);

// 在 main() 的 gst_init() 之后注册
gst_plugin_register_static(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    "gpioalarmsink",
    "GPIO Alarm Sink Plugin for DeepStream",
    gst_gpio_alarm_sink_plugin_init,
    "1.0.0",
    "LGPL",
    "deepstream-app-custom",
    "DeepStream GPIO Alarm",
    "https://nvidia.com"
);
```

## 文件结构

### 新增文件

```
src/gst-gpio-alarm-sink/
├── CMakeLists.txt              # 构建配置
├── gstgpio_alarm_sink.cpp      # GStreamer sink 主实现
├── gstgpio_alarm_sink.h        # Sink 类定义
├── gpio_controller.cpp         # GPIO 控制器实现
├── gpio_controller.h           # GPIO 控制器定义
└── README.md                   # 使用说明（可选）
```

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `apps-common/includes/deepstream_sinks.h` | NvDsSinkType 枚举添加 NV_DS_SINK_GPIO_ALARM；NvDsSinkSubBinConfig 新增字段 |
| `deepstream_app_config_parser.c` | 新增 sink type 8 配置解析 |
| `deepstream_app.c` | 创建 gpioalarmsink 元素 |
| `deepstream_app_main.c` | 注册插件 |

## 构建配置

### CMakeLists.txt

参考现有 `gst-udpmulticast_sink` 插件的构建配置：

```cmake
cmake_minimum_required(VERSION 3.10)
project(gst-gpio-alarm-sink VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0)
pkg_check_modules(DEEPSTREAM REQUIRED deepstream-6.x)

# 源文件
set(SOURCES
    gstgpio_alarm_sink.cpp
    gpio_controller.cpp
)

# 创建共享库
add_library(gstgpioalarmsink SHARED ${SOURCES})

target_include_directories(gstgpioalarmsink PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../apps-common/includes
)

target_link_libraries(gstgpioalarmsink PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${DEEPSTREAM_LIBRARIES}
    nvdsgst_helper
    nvdsgst_meta
    nvds_meta
    pthread
)

# 安装到 GStreamer 插件目录
install(TARGETS gstgpioalarmsink
    LIBRARY DESTINATION ${GSTREAMER_LIBRARY_DIR}/gstreamer-1.0
)
```

### 编译命令

```bash
cd src/gst-gpio-alarm-sink
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| GPIO 导出失败 | 记录错误日志，插件继续运行但不输出 GPIO 信号 |
| GPIO 文件打开失败 | 记录错误日志，禁用 GPIO 输出 |
| 无效配置值 | 配置解析时验证参数范围，无效值使用默认值并警告 |
| 闪烁线程异常 | 捕获异常，记录日志，尝试重启线程（最多 3 次） |

## 测试计划

1. **单元测试**：GPIOController 状态机逻辑（触发/消失防抖）
2. **集成测试**：端到端流水线测试，验证检测到目标后 GPIO 闪烁
3. **硬件测试**：实际 GPIO 输出验证（示波器/LED）
4. **多路源测试**：验证多路视频源场景下的报警行为
