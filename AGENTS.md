# Repository Guidelines

本指南用于统一本仓库（DeepStream 流水线、自定义 GStreamer 插件、Triton 客户端）的贡献方式。

## Project Structure & Module Organization
- 主应用：`src/deepstream-app/`（配置、DeepStream pipeline 源码、模型）。
- 插件：`src/gst-udpmulticast_sink/`（UDP 多播 sink）、`src/gst-videorecognition/`（多帧识别）、`src/nvdspreprocess_lib/`、`src/nvmsgconv/`。
- 公共代码：`apps-common/` 头文件与工具函数。
- YOLO 自定义推理：`DeepStream-Yolo/nvdsinfer_custom_impl_Yolo`。
- 其他：`sot_plugin/`（单目标跟踪）、`smart_rec_rgb/`、`triton_model/`（模型文件）、`test_triton.py`、`test_video_triton.py`（客户端验证）、`start_*.sh` 启动脚本。

## Build, Test, and Development Commands
```bash
# 编译 YOLO 自定义推理
cd DeepStream-Yolo && make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
# 编译/安装 UDP 多播 sink
cd src/gst-udpmulticast_sink && cmake -B build -S . && cmake --build build && sudo cmake --install build
# 编译/安装多帧识别插件
cd src/gst-videorecognition && cmake -B build -S . && cmake --build build && sudo cmake --install build
# 编译/安装 SOT 插件
cd sot_plugin && cmake -B build -S . && cmake --build build && sudo cmake --install build
# 编译主程序
cmake -B build -S ./src/deepstream-app/ && cmake --build build && sudo cmake --install build
```
- 运行前设置：`export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$GST_PLUGIN_PATH`。
- 启动：`./start_rgb_app.sh`、`./start_ir_app.sh`、`./start_both_apps.sh`；Triton：`./start_triton_server.sh`。
- 快速验证：`python3 test_triton.py --image <img> --protocol grpc` 或 `python3 test_video_triton.py --video <mp4>`。

## Coding Style & Naming Conventions
- C/C++：4 空格缩进，风格参考`.clang-format` ；函数/变量用 `snake_case`；新头文件加 include guard，常量放头文件。
- CMake：使用 out-of-source 构建（`cmake -B build -S .`），目标命名清晰。
- Python：遵循 PEP 8，能写类型注解就写，零散脚本命名为 `test_<area>.py`。
- 配置：放在 `src/deepstream-app/configs/`，命名与现有 YAML/INI 对齐。

## Testing Guidelines
- 无完整测试框架，主要依赖流水线和 Triton 客户端冒烟测试，注意收集 GStreamer WARN/ERR 日志。
- 新增模型/配置时，提供最小示例资源路径，确认解码 → 推理 → 输出链路可跑通。
- 临时脚本沿用 `test_<area>.py` 命名方便发现。

## Commit & Pull Request Guidelines
- 提交信息参考历史：简短中文主题，可按编号列要点（如 `1. ... 2. ...`），点出组件与影响。
- PR 需写明改动内容、运行方式（具体启动脚本/测试命令）、新增环境变量或配置、变化前后表现；有输出变更最好附日志或截图。

## Security & Configuration Tips
- 不要提交模型二进制或密钥，需放入本地 `triton_model/` 目录自行引用。
- 调试前确认驱动/CUDA/TensorRT 版本满足 README 要求。
- 将多播/RTSP 等地址放配置文件，提交时勿暴露内部地址。
