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
| 防抖机制 | 持续触发防抖（连续 N 帧后触发） |
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

## 模块设计

### 1. GPIOController 类

负责 GPIO 硬件控制和闪烁逻辑。

```cpp
// gpio_controller.h

class GPIOController {
public:
    GPIOController(int pin, float freqHz, float dutyCycle, int debounceFrames);
    ~GPIOController();

    // 更新检测状态（每帧调用）
    void updateDetectionState(bool hasTarget);

    // 启动/停止闪烁线程
    void start();
    void stop();

private:
    int m_pin;              // GPIO 引脚号
    float m_freqHz;         // 闪烁频率 (Hz)
    float m_dutyCycle;      // 占空比 (0.0-1.0)
    int m_debounceFrames;   // 防抖帧数

    int m_consecutiveFrames;    // 连续检测帧计数
    bool m_isAlarming;          // 当前是否在报警状态
    std::atomic<bool> m_running; // 线程运行标志
    std::thread m_blinkThread;  // 闪烁线程

    void blinkLoop();           // 闪烁循环
    void setGPIO(bool high);    // 设置 GPIO 电平
};
```

#### 状态机

```
[空闲] --连续N帧检测到目标--> [报警中]
[报警中] --目标消失--> [空闲]
```

- 空闲状态：GPIO 保持低电平
- 报警状态：独立线程按频率闪烁 GPIO

#### GPIO 操作

使用 sysfs 方式，无需额外依赖：

```cpp
void GPIOController::setGPIO(bool high) {
    std::ofstream file("/sys/class/gpio/gpio" + std::to_string(m_pin) + "/value");
    file << (high ? "1" : "0");
}
```

### 2. GstGPIOAlarmSink 插件

GStreamer sink 元素，从 buffer 元数据中提取检测结果。

```cpp
// gstgpio_alarm_sink.h

#define GST_TYPE_GPIO_ALARM_SINK (gst_gpio_alarm_sink_get_type())
G_DECLARE_FINAL_TYPE(GstGPIOAlarmSink, gst_gpio_alarm_sink,
                     GST, GPIO_ALARM_SINK, GstBaseSink)

struct _GstGPIOAlarmSink {
    GstBaseSink parent;

    // 配置属性
    gint gpio_pin;           // GPIO 引脚
    gfloat blink_freq;       // 闪烁频率
    gfloat duty_cycle;       // 占空比
    gint debounce_frames;    // 防抖帧数
    gchar *alarm_class_ids;  // 类别ID字符串 (如 "0,1,2")

    // 运行时状态
    GPIOController *controller;
    std::set<gint> m_classIdSet;  // 解析后的类别ID集合
};
```

#### 属性定义

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gpio-pin` | int | 216 | GPIO 引脚号 |
| `blink-freq` | float | 2.0 | 闪烁频率 (Hz) |
| `duty-cycle` | float | 0.5 | 占空比 (0.0-1.0) |
| `debounce-frames` | int | 10 | 防抖帧数 |
| `alarm-class-ids` | string | "0" | 触发类别 (逗号分隔) |

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

    // 2. 检查是否有目标类别
    bool hasTarget = false;
    for (frame in batch_meta->frame_meta_list) {
        for (obj in frame->obj_meta_list) {
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

### 配置文件结构

在 `app_config.yml` 中新增 sink 配置：

```yaml
sink3:
  enable: 1
  type: 8  # 新类型：GPIO 报警 sink
  # GPIO 配置
  gpio-pin: 216
  blink-freq: 2.0
  duty-cycle: 0.5
  debounce-frames: 10
  alarm-class-ids: "0"
```

### 配置解析修改

在 `deepstream_app_config_parser.c` 中添加：

```c
// NvDsSinkSubBinConfig 结构体新增字段
typedef struct {
    // ... 现有字段 ...

    // GPIO Alarm Sink 配置
    guint gpio_pin;
    gfloat blink_freq;
    gfloat duty_cycle;
    guint debounce_frames;
    gchar *alarm_class_ids;
} NvDsSinkSubBinConfig;
```

### 流水线构建修改

在 `deepstream_app.c` 的 `create_sink_bin` 函数中添加：

```c
case 8:  // GPIO Alarm Sink
{
    GstElement *gpio_sink = gst_element_factory_make("gpioalarmsink", NULL);
    g_object_set(gpio_sink,
        "gpio-pin", config->gpio_pin,
        "blink-freq", config->blink_freq,
        "duty-cycle", config->duty_cycle,
        "debounce-frames", config->debounce_frames,
        "alarm-class-ids", config->alarm_class_ids,
        NULL);
    gst_bin_add(GST_BIN(sink_bin), gpio_sink);
    // 连接到 tee
    ...
    break;
}
```

### 插件注册

在 `deepstream_app_main.c` 中添加插件初始化：

```c
// 插件初始化
extern gboolean gst_gpio_alarm_sink_plugin_init(GstPlugin *plugin);

// 在 main() 中注册
gst_plugin_register_static(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    "gpioalarmsink", "GPIO Alarm Sink Plugin",
    gst_gpio_alarm_sink_plugin_init,
    "1.0", "LGPL", "deepstream-app-custom", ...
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
| `apps-common/includes/deepstream_app.h` | NvDsSinkSubBinConfig 新增字段 |
| `deepstream_app_config_parser.c` | 新增 sink type 8 配置解析 |
| `deepstream_app.c` | 创建 gpio-alarm-sink 元素 |
| `deepstream_app_main.c` | 注册插件 |

## 构建配置

### CMakeLists.txt

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
)

target_link_libraries(gstgpioalarmsink PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${DEEPSTREAM_LIBRARIES}
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

1. **GPIO 初始化失败**：记录错误日志，插件继续运行但不输出 GPIO 信号
2. **无效配置**：配置解析时验证参数范围，无效值使用默认值并警告
3. **线程异常**：闪烁线程异常退出时自动重启

## 测试计划

1. **单元测试**：GPIOController 状态机逻辑
2. **集成测试**：端到端流水线测试
3. **硬件测试**：实际 GPIO 输出验证（示波器/LED）
