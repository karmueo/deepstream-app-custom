# DeepStream 人脸检测与识别插件

`facedetect` 是接收 DeepStream NVMM 帧的 GStreamer 元素。它独立检测人脸，使用
SCRFD-500MF 和 MobileFaceNet/ArcFace 做五点对齐与身份匹配，向每张脸写入
`NvDsObjectMeta` 与 `FACE_DETECT_META_TYPE` 用户元数据。插件不会改写视频帧。

## 目录与依赖

- `includes/`、`src/`：接口和实现；`tools/`：引擎构建与人员照片导入。
- `model/`：`det_500m.onnx`、`w600k_mbf.onnx`、本机生成的引擎及转换脚本。
- `data/`：每个一级子目录对应一人，目录名作为姓名；照片和 `gallery.sqlite` 留在本机。
- `tests/`：核心测试及 `test/test.mp4` 实机验证入口。

需要 Jetson DeepStream 7.1、CUDA 12.6、TensorRT 10、GStreamer、SQLite、
JSON-GLib、yaml-cpp 和 Ubuntu 22.04 的 OpenCV 4.5 运行库。当前机器的
`/usr/include/opencv4` 来自 NVIDIA OpenCV 4.8，不能用于与 SOT 一起构建。
CMake 首次配置时会调用 `tools/prepare_opencv_headers.sh`，将 Ubuntu 4.5
开发包**只解压到当前构建目录**，不替换系统文件。此步骤需要能访问配置的
APT 软件源；若已有匹配的 4.5 头文件，可通过
`-DFACE_OPENCV_INCLUDE_DIR=/path/to/opencv4` 指定。

```bash
cd /home/nvidia/work/deepstream-app-custom
cmake -S src/face_detect_plugin -B src/face_detect_plugin/build -DCMAKE_BUILD_TYPE=Debug
cmake --build src/face_detect_plugin/build -j4
ctest --test-dir src/face_detect_plugin/build --output-on-failure
```

## 模型与人员库

若本地没有 ONNX，先运行 `model/download_models.sh` 从 InsightFace 官方模型
发布页下载 `buffalo_sc`。`model/convert2trt.sh` 在当前 Jetson 上把两个 ONNX 模型转换为 FP16 引擎，
同时写入 SHA-256、运行环境、输入形状、输出名称和层精度文件；成功后会加载
两个引擎预热。TensorRT 引擎不可直接复制到其他 GPU/运行环境使用。

```bash
src/face_detect_plugin/model/download_models.sh  # 本地已有 ONNX 时跳过
src/face_detect_plugin/model/convert2trt.sh
cmake -S src/deepstream-app -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
```

如需身份匹配，再导入本地人员照片：

```bash
src/face_detect_plugin/build/face-gallery-import \
  --config src/face_detect_plugin/config_face_detect.yml \
  --database src/face_detect_plugin/data/gallery.sqlite
```

默认扫描 `data/` 下一级人员目录，并递归读取其中的 JPG、PNG、BMP、WebP
照片。单张图片必须恰好有一张可检测且至少 32 像素的人脸。重复照片按
人员、文件哈希和识别模型去重；无效图片会逐条报告。导入为增量写入，删除
照片不会删除旧特征。完全重建时使用新的数据库文件。插件启动时只读载入
当前模型的人员库；修改照片并导入后，重启流水线才能看到新特征。

`config_face_detect.yml` 中 `interval: 0` 表示逐帧识别；`N` 表示每次识别后
跳过该视频源的 N 帧。阈值分别控制检测、NMS 和余弦匹配。`gallery-file`
默认留空，全部人脸显示为“陌生人”；导入人员后可将其设为
`data/gallery.sqlite`。路径非空但数据库不存在会启动失败。

## 接入与测试

主程序 `app_config.yml` 的 `face-detect.enable` 控制是否创建元素，
`config-file` 指向本插件配置。插件位于 YOLO/SOT 等公共分析元素之后，
OSD 之前；也可关闭预处理、YOLO 和跟踪器，单独运行人脸分析。

```bash
export DISPLAY=:10.0
src/face_detect_plugin/tests/run_test_video.sh
```

脚本基于主配置生成一次性纯人脸测试配置，在构建目录运行，不修改生产
YAML；文件源默认使用 DeepStream 自带的 `sample_720p.mp4`，也可用
`FACE_TEST_VIDEO=/path/to/video.mp4` 指定，播完即停。默认保存带框视频到
`build/test-video/annotated.mkv`；`FACE_TEST_SINK=egl` 切换到窗口显示，
`FACE_TEST_SINK=fake` 仅验证元数据。当前机器在 `DISPLAY=:10.0` 上的
DeepStream EGL sink 报 CUDA-GL 注册错误，因此使用文件输出验收画面。
可用 `GST_DEBUG=facedetect:5` 增加检测及身份日志。运行前先完成模型转换；
无人员库时可设置 `FACE_TEST_EMPTY_GALLERY=1`。

在需要安装到主程序目录时运行：

```bash
sudo cmake --install src/face_detect_plugin/build
sudo cmake --install build
```

安装包含插件动态库、工具、配置与本机已有模型文件，不包含人员照片和
`gallery.sqlite`。部署时若需要身份识别，将数据库复制到可读的位置并
修改 `config_face_detect.yml` 的 `gallery-file` 路径。主程序的启动脚本
已经包含 `/opt/deepstream-app-custom/gst-plugins` 的搜索路径。

下游可通过 `unique_component_id` 区分人脸对象。公开头文件
`includes/face_metadata.h` 定义了完整 UTF-8 姓名、人员 ID、匹配标志、
余弦相似度和五点坐标；`object_id` 为未跟踪状态。该相似度不是概率，
应使用实际人员及陌生人片段校准阈值。InsightFace 预训练模型的许可要求
由使用者单独确认。
