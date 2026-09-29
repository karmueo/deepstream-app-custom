# DeepStream EmotiEff 情绪识别插件

`emotieff` 接收 DeepStream NVMM 批量帧，只处理上游 `facedetect` 产生的人脸对象。它使用原始人脸框裁剪图像，通过 TensorRT 识别八类情绪；原有身份、边框、检测置信度保持在同一个对象上。结果同时写入 `NvDsClassifierMeta` 和公开的 `EMOTIEFF_META_TYPE` 用户元数据，OSD 显示“姓名 身份相似度 情绪 情绪分数”。无人脸时不会转换画面或推理。

## 目录与构建

- `model/`：原始 ONNX、当前 Jetson 的 FP16 引擎、转换脚本及环境描述。
- `configs/`：模型和帧间隔配置；`includes/`、`src/`：接口与实现。
- `tools/`：引擎构建及单张人脸裁剪诊断；`tests/`：核心、元数据、精度和实机测试。

需要 Jetson DeepStream 7.1、CUDA 12.6、TensorRT 10.3、GStreamer、yaml-cpp、JSON-GLib 和 OpenCV 4.5 运行库。配置 CMake 时会将 Ubuntu 4.5 开发头文件解压到构建目录，不会替换系统头文件；若已有匹配头文件，可指定 `-DEMOTIEFF_OPENCV_INCLUDE_DIR=/path/to/opencv4`。

```bash
src/emotieff_plugin/model/prepare_models.sh
cmake -S src/emotieff_plugin -B src/emotieff_plugin/build -DCMAKE_BUILD_TYPE=Debug
cmake --build src/emotieff_plugin/build -j4
ctest --test-dir src/emotieff_plugin/build --output-on-failure
src/emotieff_plugin/model/convert2trt.sh
cmake -S src/deepstream-app -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
```

`prepare_models.sh` 默认从本机 EmotiEffLib 原始模型目录复制 ONNX；也可用 `EMOTIEFF_MODEL_SOURCE` 指定已取得模型的目录。模型、引擎及层报告放在 `model/`，Git 忽略这些二进制和生成报告。引擎仅用于构建它的 GPU/TensorRT 环境，换设备或版本后应重新运行 `convert2trt.sh`。

## 模型和配置

默认 `enet_b0_8_best_vgaf` 使用 224×224 RGB、ImageNet 均值/标准差；`mbf_va_mtl` 使用 112×112 RGB、均值及标准差均为 0.5。两者输入均为 NCHW。类别顺序为愤怒、轻蔑、厌恶、恐惧、高兴、中性、悲伤、惊讶。MBF 的第 9、10 项是 VA 输出，仅保留在原始输出中，不参与八类 softmax。

`configs/config_emotieff.yml` 的 `emotion-engine` 路径相对配置文件解析；切换模型时须同时修改 `emotion-model` 和 `emotion-engine`。`interval: 0` 表示逐帧分析；`N` 表示每源分析一次后跳过 N 帧。跳过的帧不附加情绪结果。无效裁剪附加 `valid=false`，OSD 显示“无法判断”。

应用配置中的 `emotieff.enable: 1` 默认开启，要求 `face-detect.enable: 1`，且两者 GPU 相同、组件 ID 不同。主程序把人脸 ID 自动传入 `operate-on-gie-id`。直接使用 GStreamer 时可设置 `config-file`、`gpu-id`、`unique-id` 和 `operate-on-gie-id` 属性。公开头文件 `includes/emotieff_metadata.h` 包含分类 ID、最高概率、八类概率以及 8/10 个原始输出。

## 测试和安装

```bash
export EMOTIEFF_TEST_VIDEO=/path/to/face-video.mp4
src/emotieff_plugin/tests/run_test_video.sh
python3 src/emotieff_plugin/tests/compare_reference.py \
  --reference-bin /path/to/jetson-video-emotion \
  --video "$EMOTIEFF_TEST_VIDEO" \
  --report src/emotieff_plugin/build/reference-report.json
```

实机脚本会在 `build/test-video/` 生成独立配置和带 OSD 的 `annotated.mkv`，不修改生产配置。可设置 `EMOTIEFF_TEST_MODEL=mbf_va_mtl`、`EMOTIEFF_TEST_SOURCES=2`、`EMOTIEFF_TEST_INTERVAL=N`、`EMOTIEFF_TEST_LOOP=1`、`EMOTIEFF_TEST_SINK=fake|egl|file` 或 `EMOTIEFF_TEST_WITH_YOLO=1`（同时启用预处理和 SOT）。需要停止循环时手动中断。使用 `GST_DEBUG=emotieff:5` 查看分类日志。精度脚本需要本地参考 C++ 程序、ONNX Runtime、Python OpenCV 和含人脸的视频。

当前主机的手机测试视频含 270° 旋转元数据：OpenCV 自动转正，但 DeepStream 文件源不转正，直接运行时人脸方向错误。实机验收应使用已转正且画面确有正面人脸的视频。本机还在仅启用人脸和 YOLO/SOT 时复现了 `Could not map EglImage from NvBufSurface`，因此该组合需待现有 NVIDIA EGL 环境恢复后复验。

构建及模型准备后可安装：

```bash
sudo cmake --install src/emotieff_plugin/build
sudo cmake --install build
```

安装内容包括动态库、工具、配置、两套模型和本机引擎。部署后默认应用配置路径指向 `/opt/deepstream-app-custom/emotieff_plugin/configs/config_emotieff.yml`。
