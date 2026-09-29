# Jetson 构建与发布

当前分支不使用离线许可证。Debug、RelWithDebInfo 和 Release 构建均无需签名公钥或许可证文件。
首次构建人脸插件前若没有 ONNX，按[插件说明](../src/face_detect_plugin/README.md)下载模型。

## 本机开发构建

```bash
cmake -B src/face_detect_plugin/build -S src/face_detect_plugin -DCMAKE_BUILD_TYPE=Debug
cmake --build src/face_detect_plugin/build --parallel 2
src/face_detect_plugin/model/convert2trt.sh
cmake -B build -S src/deepstream-app -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
./build/deepstream-app-custom.bin -c src/deepstream-app/configs/yml/app_config.yml
```

## Release 编译与 DEB 打包

发布前以 Release 模式重建要打包的本地插件及模型。路径映射参数避免 `__FILE__` 将工作目录写入 ELF：

```bash
repo_root=$(pwd)
release_cxx_flags="-O2 -DNDEBUG -ffile-prefix-map=${repo_root}=."
cmake -B sot_plugin/build -S sot_plugin -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="${release_cxx_flags}"
cmake --build sot_plugin/build --clean-first --parallel
cmake -B src/face_detect_plugin/build -S src/face_detect_plugin \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="${release_cxx_flags}"
cmake --build src/face_detect_plugin/build --clean-first --parallel
src/face_detect_plugin/model/convert2trt.sh
make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo clean
make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
make -C src/nvdspreprocess_lib clean
make -C src/nvdspreprocess_lib
cmake -B build -S src/deepstream-app -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first --parallel
cpack --config build/CPackConfig.cmake
```

SOT 与人脸插件构建会校验 OpenCV 4.5 头文件及 Ubuntu 22.04 的运行库。人脸引擎转换会生成相对于引擎目录的 ONNX 路径。Release 构建仍要求所需运行库和模型齐备，但不打包测试视频。`deepstream-app-custom.bin.debug` 是内部调试符号，不进入 DEB。包内包含主程序、人脸插件、运行库、配置、标签、选定的 engine、对应的 ONNX 模型以及 YOLO 模型转换脚本。人员库属于部署数据，不进入 DEB；默认留空时人脸会显示为“陌生人”。默认文件源使用 DeepStream 安装包自带的 `sample_720p.mp4`。DeepStream 7.1、L4T 36.4.x、CUDA 12.6 和 TensorRT 由目标 Jetson 预装。

包内运行配置声明为 Debian conffile，升级时由 `dpkg` 处理本地配置变更。

## 发布验收

```bash
staging_dir=$(mktemp -d)
dpkg-deb -x deepstream-app-custom_1.0.0_arm64.deb "$staging_dir"
python3 tools/verify_release.py \
  "$staging_dir/opt/deepstream-app-custom" --runtime-checks
```

`--runtime-checks` 需在相容 Jetson 上执行，额外检查 `ldd` 和两个 GStreamer 插件。静态检查验证模型与转换脚本齐全，并排查源码、头文件、Python 脚本、ELF 调试节、工作目录路径及 RTSP 明文账号。
