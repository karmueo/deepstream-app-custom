# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 提供代码仓库操作指导。

## 项目概述

基于 NVIDIA DeepStream SDK 的视频分析应用，支持实时目标检测、单目标跟踪（SOT）和多帧动作识别，采用 GStreamer + TensorRT 技术栈。

## 编译命令

```bash
# 编译所有插件和主程序
cd src/deepstream-app && make

# 或分步编译（需要 sudo 安装插件）
cd DeepStream-Yolo && make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
cd src/gst-udpmulticast_sink && cmake -B build -S . && cmake --build build && sudo cmake --install build
cd src/gst-videorecognition && cmake -B build -S . && cmake --build build && sudo cmake --install build
cd sot_plugin && cmake -B build -S . && cmake --build build && sudo cmake --install build
cd src/deepstream-app && cmake -B build -S . && cmake --build build && sudo cmake --install build
```

## 运行命令

```bash
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$GST_PLUGIN_PATH
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
./build/deepstream-app -c src/deepstream-app/configs/yml/app_config.yml
```

## 架构设计

### 流水线阶段

```
视频源 → NVDEC (GPU 解码) → 流复用器 → 主推理 (YOLOv11) →
目标跟踪 (SOT) → 多帧识别 (X3D) → UDP/MQTT 输出
```

### 核心组件

| 组件 | 路径 | 用途 |
|------|------|------|
| 主程序 | `src/deepstream-app/` | 入口点、流水线构建、配置解析 |
| UDP 多播Sink | `src/gst-udpmulticast_sink/` | JSON 元数据输出的 GStreamer 插件 |
| 视频识别 | `src/gst-videorecognition/` | 多帧动作识别插件 |
| SOT 插件 | `sot_plugin/` | 单目标跟踪 (MixFormerv2) |
| YOLO 自定义实现 | `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/` | TensorRT YOLO 推理 |
| 消息转换器 | `src/nvmsgconv/` | NvDsEventMsgMeta 转 JSON |

### 数据流向

1. **NvDsBatchMeta**: 缓冲区携带此元数据通过流水线，在流复用器处附加
2. **NvDsObjectMeta**: 目标检测结果附加到帧元数据
3. **NvDsFrameTensorMeta**: 预处理张量供下游推理使用
4. **自定义探针**: `deepstream_app_probes.c` 中的 `parse_bbox_from_tensor_meta`、`pre_process`、`post_process`

### 关键文件

- `src/deepstream-app/deepstream_app_main.c:31` - 生命周期管理（创建→准备→启动→停止）
- `src/deepstream-app/deepstream_app.c:1` - 流水线设置和流复用配置
- `src/deepstream-app/deepstream_app_probes.c:1` - 元数据处理（探针→解析→处理→附加）
- `apps-common/includes/deepstream_app.h` - 应用程序上下文结构体定义

## 代码风格

- **格式**: `.clang-format`，使用 Allman 括号风格 (`BreakBeforeBraces: Allman`)
- **缩进**: 4 空格，80 列限制
- **注释**: 函数/结构体使用中文注释，技术说明使用英文

## 配置说明

- 主配置: `src/deepstream-app/configs/yml/app_config.yml`
- 推理配置: `config_infer_primary_yoloV11_rgb.yml`
- 视频源: `file_sources.csv`、`sources.csv`
- 模型文件: 放置于各插件的 `models/` 目录，使用 `./convert2trt.sh` 转换

## 通用开发约束

1. 不得采用只解决局部问题的补丁式修改而忽视整体设计与全局优化
2. 不得引入过多用于中间通信的中间状态以免降低可读性并形成循环依赖
3. 不得为过渡场景编写大量防御性代码以免掩盖主逻辑并增加维护成本
4. 不得只追求功能完成而忽略架构设计
5. 不得省略必要注释，代码必须对他人和未来维护者可理解
6. 不得编写难以阅读的代码，必须保持结构简单清晰并添加解释性注释
7. 不得违反 SOLID 与 DRY 原则，必须保持职责单一并避免逻辑重复
8. 不得维护复杂的中间状态，仅允许保留最小必要的核心数据
10. 不得通过隐式或间接方式变更状态，状态变化应直接更新数据并由框架重新计算
11. 不得编写过量的防御性代码，应通过清晰的数据约束与边界设计解决问题
12. 不得保留未被使用的变量和函数
16. 不得形成隐式依赖，如依赖调用顺序、全局初始化或副作用时序
17. 不得吞掉异常或使用空 catch 掩盖错误
18. 不得将异常作为正常控制流的一部分
19. 不得返回语义不清或混用的错误结果（如 null / undefined / false）
20. 不得在多个位置同时维护同一份事实数据
22. 不得跨请求共享可变状态，除非明确设计为并发安全
23. 不得使用语义模糊或误导性的命名
24. 不得让单个函数或模块承担多个不相关语义
25. 不得引入非必要的时间耦合或隐含时间假设
26. 不得在关键路径中引入不可控的复杂度或隐式状态机
27. 不得臆测接口行为，必须先查询文档、定义或源码
28. 不得在需求、边界或输入输出不清晰的情况下直接实现
29. 不得基于猜测实现业务逻辑，必须与人类确认需求并留痕
30. 不得在未评估现有实现的情况下新增接口或模块
31. 不得跳过验证流程，必须编写并执行测试用例
32. 不得触碰架构红线或绕过既有设计规范
33. 不得假装理解需求或技术细节，不清楚时必须明确说明
34. 不得在缺乏上下文理解的情况下直接修改代码，必须基于整体结构审慎重构
