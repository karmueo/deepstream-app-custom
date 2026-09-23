# Jetson 离线许可与发布

本项目的 Release 交付程序使用一份通用 ARM64 程序和每台 Jetson 一份的 Ed25519 离线许可证。运行程序只内置公钥，私钥仅保留在发布环境。许可证绑定 Jetson 模组序列号的 SHA-256 指纹，不记录原始序列号。非 Release 开发构建默认关闭许可校验，仅供本机调试。

## 1. 一次性生成签名密钥

在脱离 Jetson 和源码仓库的发布机密钥目录执行：
签发机需安装 Python `cryptography`包（Ubuntu 可使用 `python3-cryptography`）。

```bash
python3 tools/license_issuer.py keygen \
  --private-key ~/secure/license/prod-2026-01.pem \
  --public-key ~/secure/license/prod-2026-01.pub.pem \
  --key-id prod-2026-01
```

工具会输出 CMake 所需的 64 位十六进制公钥。私钥和默认签发台账均为 `0600`，不得提交、打包或复制到 Jetson。

## 2. 本机开发构建

开发和调试统一使用 `build` 目录，不需要许可证：

```bash
cmake -B build -S src/deepstream-app \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_LICENSE_TESTS=ON
cmake --build build --parallel 2
./build/deepstream-app-custom.bin \
  -c src/deepstream-app/configs/yml/app_config.yml
```

`Debug`、`RelWithDebInfo` 和未指定 `CMAKE_BUILD_TYPE` 的构建都会跳过正常启动时的许可校验。执行 `--license-info` 会显示 `License enforcement: disabled (development build)`。`--license-request` 仍然可用。

## 3. 生产编译并生成 DEB

Release 构建会拒绝使用仓库中无对应私钥的占位公钥：

发布前应以 Release 模式重建会被打入包内的本地插件。路径映射参数用于避免
`__FILE__` 将发布机的工作目录写入 ELF：

```bash
repo_root=$(pwd)
release_cxx_flags="-O2 -DNDEBUG -ffile-prefix-map=${repo_root}=."
cmake -B sot_plugin/build -S sot_plugin -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="${release_cxx_flags}"
cmake --build sot_plugin/build --clean-first --parallel
cmake -B src/gst-udpjson_meta/build -S src/gst-udpjson_meta \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="${release_cxx_flags}"
cmake --build src/gst-udpjson_meta/build --clean-first --parallel
cmake -B src/gst-udpmulticast_sink/build -S src/gst-udpmulticast_sink \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="${release_cxx_flags}"
cmake --build src/gst-udpmulticast_sink/build --clean-first --parallel
make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo clean
make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
make -C src/nvdspreprocess_lib clean
make -C src/nvdspreprocess_lib
```

SOT 构建会同时校验 OpenCV 4.5 头文件以及 Ubuntu 22.04 的
`libopencv_core.so.4.5d`、`libopencv_imgproc.so.4.5d`，避免误链接
`/usr/local` 下的其他 OpenCV ABI。

随后编译主程序并生成 DEB：

```bash
cmake -B build -S src/deepstream-app \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LICENSE_TESTS=ON \
  -DLICENSE_ED25519_PUBLIC_KEY_HEX=<keygen输出的公钥> \
  -DLICENSE_KEY_ID=prod-2026-01
cmake --build build --clean-first --parallel
ctest --test-dir build --output-on-failure
cpack --config build/CPackConfig.cmake
```

Release 构建必须提供正式公钥，并始终启用 fail-closed 许可校验。只有 Release 会生成 `build/CPackConfig.cmake`。`deepstream-app-custom.bin.debug` 是内部调试符号，不会进入 DEB。DEB 包含主程序、运行库/插件、配置、标签、已选 engine、对应的 ONNX 模型及静态模型转换脚本；DeepStream 7.1、L4T 36.4.x、CUDA 12.6 和 TensorRT 由目标 Jetson 预装。SOT 使用 Ubuntu 22.04 提供的 OpenCV 4.5 ABI，DEB 会显式安装 `libopencv-core4.5d` 和 `libopencv-imgproc4.5d`。

包内运行配置被声明为 Debian conffile。客户修改视频源或流水线参数后，升级
DEB 时 `dpkg` 会保留修改或提示处理配置冲突，不会静默恢复成示例配置。

发布后切回开发模式：

```bash
cmake -B build -S src/deepstream-app \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --clean-first --parallel
```

非 Release 配置会把 `build` 缓存中的生产公钥和 `key_id` 重置为开发占位值，并删除旧的 CPack 配置。因此下次切换到 Release 时必须再次显式传入公钥和 `key_id`。

## 4. 离线申请与签发

安装 DEB 后，客户 Jetson 上生成申请：

```bash
/opt/deepstream-app-custom/bin/deepstream-app-custom.bin \
  --license-request license-request.json
```

将申请文件带回发布机，离线签发：

```bash
python3 tools/license_issuer.py issue \
  --request license-request.json \
  --private-key /secure/license/prod-2026-01.pem \
  --key-id prod-2026-01 \
  --customer "客户名称" \
  --output license.lic
```

可在发布机验证和查看已签名载荷：

```bash
python3 tools/license_issuer.py inspect \
  --license license.lic \
  --public-key /secure/license/prod-2026-01.pub.pem
```

将许可证安装到原 Jetson：

```bash
sudo install -D -m 0644 license.lic \
  /etc/deepstream-app-custom/license.lic
/opt/deepstream-app-custom/bin/deepstream-app-custom.bin --license-info
```

可使用 `--license-file FILE` 临时覆盖默认路径。Release 程序正常启动时会在解析配置、加载 engine 和创建 CUDA/GStreamer pipeline 之前 fail-closed 校验许可证，失败返回码为 77。

## 5. 发布验收

解包 DEB 并执行静态检查：

```bash
staging_dir=$(mktemp -d)
dpkg-deb -x deepstream-app-custom_1.0.0_arm64.deb "$staging_dir"
python3 tools/verify_release.py \
  "$staging_dir/opt/deepstream-app-custom" --runtime-checks
```

`--runtime-checks` 需在相容 Jetson 上运行，额外检查 `ldd` 和两个 GStreamer 插件。静态检查要求包内包含指定 ONNX 模型和 `models/convert2trt.sh`，并拒绝源码、头文件、Python 脚本、其他 shell 脚本、ELF 调试节、`/home/nvidia` 路径和 RTSP 明文账号。

## 安全边界

首版为永久离线许可，不检查系统时间，不支持撤销，engine 保持明文。该方案防止直接复制安装目录和许可证到另一台 Jetson，不承诺抵抗 root 用户修改二进制、伪造设备树或物理攻击。
