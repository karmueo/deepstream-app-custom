
# 1. 环境安装

## Jetson环境

### 安装依赖

```bash
sudo apt install \
libssl3 \
libssl-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev \
ninja-build
```

### 安装Deepstream

从官网下载deepstream_sdk_v7.1.0_jetson.tbz2

```bash
sudo tar -xvf deepstream_sdk_v7.1.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-7.1
sudo ./install.sh
sudo ldconfig
```

### 配置环境变量

```bash
export CUDA_VER=12.6
```
## 服务器环境

### 安装依赖包

```bash
sudo apt install \
libssl3 \
libssl-dev \
libgles2-mesa-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev \
libjsoncpp-dev \
protobuf-compiler \
gcc \
make \
git \
python3 \
python3-pip \
libjson-glib-dev \
libgstreamer1.0-dev \
libgstrtspserver-1.0-dev \
libx11-dev \
libgbm1 \
libglapi-mesa
```
> 注：安装时不要在conda环境下安装，如果在conda环境则执行`conda deactivate`来退出conda虚拟环境。

### 安装显卡驱动
pass

### 安装CUDA Toolkit
历史版本下载地址: https://developer.nvidia.com/cuda-toolkit-archive。历史版本下载地址: https://developer.nvidia.com/cuda-toolkit-archive 这里使用的版本是: cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb。

```bash
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

安装完查看环境变量:
```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# deepstream
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${LD_LIBRARY_PATH}
"
```

### 安装TensorRT
下载地址: https://developer.nvidia.com/tensorrt/download/10x。这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

### 安装Deepstream SDK
下载地址: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream?version=8.0, 这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

# 2.编译安装

## 编译DeepStream-Yolo

```sh
cd DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

## 编译报文发送插件

```sh
cd src/gst-udpmulticast_sink
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 编译报文接收插件

```sh
cd src/gst-udpjson_meta
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 编译单目标跟踪插件
```sh
cd sot_plugin
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 编译多帧目标识别插件
```sh
cd src/gst-videorecognition
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 编译主程序

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 添加环境变量
```sh
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$GST_PLUGIN_PATH
```

# 3. 准备模型
## 目标检测模型
把目标检测模型onnx文件放入src/deepstream-app/models目录下，根据实际的模型名称修改下面的参数：
动态 batch: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
然后根据实际的engine文件名修改`src/deepstream-app/configs/yml/config_infer_primary_yoloV11_rgb.yml`中`model-engine-file`的值

## 单目标跟踪模型
把onnx模型文件放入`src/sot_plugin/models`目录下，

```sh
# 用法: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
# 例如: 
./convert2trt.sh nanotrack_head.onnx nanotrack_head_fp16.engine fp16
./convert2trt.sh nanotrack_backbone.onnx nanotrack_backbone_fp16.engine fp16
./convert2trt.sh nanotrack_backbone_search.onnx nanotrack_backbone_search_fp16.engine fp16
```

## 多帧识别模型
把onnx模型如放到`src/gst-videorecognition/models`目录下，使用`./convert2trt.sh`转换，类似前面的转换操作

# 4.开机自启动

## 程序开机自启动
当前 `sink1.type: 2` 使用的是 `EglSink`，推荐通过用户级 `systemd` 服务在桌面会话中自动启动：

```bash
/opt/nvidia/deepstream/deepstream/deepstream-app-custom/start_rgb_app.sh
```

## 步骤如下：

1) 确保系统进入图形目标，并为桌面用户开启自动登录。

```bash
systemctl get-default
systemctl status gdm --no-pager
```

2) 安装 GUI 启动脚本与用户级 service

```bash
sudo install -m 755 ./start_rgb_app.sh \
  /opt/nvidia/deepstream/deepstream/deepstream-app-custom/start_rgb_app.sh
mkdir -p ~/.config/systemd/user
install -m 644 ./systemd/user/deepstream-app-rgb.service \
  ~/.config/systemd/user/deepstream-app-rgb.service
```

3) 若之前启用过旧的系统级服务，先停掉它

```bash
sudo systemctl disable --now deepstream-app-rgb.service 2>/dev/null || true
sudo systemctl disable --now deepstream-app-rgb-drm.service 2>/dev/null || true
```

4) 重新加载并启用 GUI 用户服务

```bash
systemctl --user daemon-reload
systemctl --user enable deepstream-app-rgb.service
systemctl --user start deepstream-app-rgb.service
```

5) 查看状态与日志

```bash
systemctl --user status deepstream-app-rgb.service
journalctl --user -u deepstream-app-rgb.service -f
```

6) 停止和取消自启动

```bash
systemctl --user stop deepstream-app-rgb.service
systemctl --user disable deepstream-app-rgb.service
```

说明：
- `start_rgb_app.sh` 会保留图形会话环境，并在缺失时兜底设置 `DISPLAY`、`XDG_RUNTIME_DIR`、`XAUTHORITY`。
- GUI 模式依赖桌面会话；如果系统没有自动登录，用户级服务不会真正启动窗口。
- 如果你的应用依赖其他服务（如 MQTT），可在 `systemd/user/deepstream-app-rgb.service` 里继续追加 `After=`/`Wants=`。
- 如果 `/opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs` 还没建立，请先按主程序编译步骤执行一次 `cmake --install build`。
- Jetson 桌面环境默认会启动 `nvpmodel_indicator`。如果启动后持续弹出 `System throttled due to over-current`，通常不是 DeepStream 本身报错，而是设备处于高功耗模式时触发了过流保护。
- 当前设备建议优先使用 `25W` 模式，不建议长期使用 `MAXN_SUPER` 跑 GUI 推理：

```bash
sudo nvpmodel -m 1
sudo reboot
```

- 如果只是想关闭过流弹窗提示，而不改功耗模式，可以禁用桌面自启动的 `nvpmodel_indicator`：

```bash
mkdir -p ~/.config/autostart
cp /etc/xdg/autostart/nvpmodel_indicator.desktop ~/.config/autostart/
printf '\nHidden=true\n' >> ~/.config/autostart/nvpmodel_indicator.desktop
pkill -f nvpmodel_indicator.py
```

## 旧版 DRM 兼容方式

如果你要回到无桌面本地显示的 `nvdrmvideosink` 模式，需要先把 `sink1.type`
改回 `5`，然后使用系统级 DRM 服务：

```bash
sudo install -m 755 ./start_rgb_drm_app.sh \
  /opt/nvidia/deepstream/deepstream/deepstream-app-custom/start_rgb_drm_app.sh
sudo install -m 644 ./systemd/deepstream-app-rgb-drm.service \
  /etc/systemd/system/deepstream-app-rgb-drm.service
sudo systemctl daemon-reload
sudo systemctl enable deepstream-app-rgb-drm.service
sudo systemctl start deepstream-app-rgb-drm.service
```

注意：
- `deepstream-app-rgb-drm.service` 只适合 Jetson 本地屏直出，不适合桌面 GUI 窗口。
- DRM 模式通常要求系统运行在 `multi-user.target`，并停用桌面显示管理器，否则 DRM 设备可能被占用。
- 调试时不要从 SSH/X11 转发会话直接运行，否则会混入 `DISPLAY/XAUTHORITY` 导致额外的 EGL/X11 报错。

## (可选)定时关闭、启动服务（由此可以定时切换模型，比如夜间和白天用不同的模型）

> 注意: 先停止前面的服务：`systemctl --user stop deepstream-app-rgb.service` 或 `sudo systemctl stop deepstream-app-rgb-drm.service`

1) 创建白天服务文件 `/etc/systemd/system/deepstream-day.service`

```ini
[Unit]
Description=DeepStream Day App (07:00 - 19:00)
After=network-online.target mosquitto.service
Wants=network-online.target mosquitto.service
# 当本服务启动时，强制停止夜间服务
Conflicts=deepstream-night.service

[Service]
Type=simple
User=tl
Group=tl
WorkingDirectory=/opt/nvidia/deepstream/deepstream
# 白天使用的 RGB 配置文件
ExecStart=/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/rgb_app_config.txt
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```
2) 创建夜晚服务文件 `/etc/systemd/system/deepstream-night.service`

```ini
[Unit]
Description=DeepStream Night App (19:00 - 07:00)
After=network-online.target mosquitto.service
Wants=network-online.target mosquitto.service
# 当本服务启动时，强制停止白天服务
Conflicts=deepstream-day.service

[Service]
Type=simple
User=tl
Group=tl
WorkingDirectory=/opt/nvidia/deepstream/deepstream
# 晚上使用的 Night 配置文件
ExecStart=/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/night_app_config.txt
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

3) 创建白天定时器文件 `/etc/systemd/system/deepstream-day.timer`

```ini
[Unit]
Description=Start Day App at 07:00 daily

[Timer]
# 每天 07:00:00 触发
OnCalendar=*-*-* 07:00:00
Unit=deepstream-day.service
# 如果关机错过了时间，开机后是否补发？(可选，建议 false 以免逻辑混乱)
Persistent=false

[Install]
WantedBy=timers.target
```

4) 创建夜晚定时器文件 `/etc/systemd/system/deepstream-night.timer`

```ini
[Unit]
Description=Start Night App at 19:00 daily

[Timer]
# 每天 19:00:00 触发
OnCalendar=*-*-* 19:00:00
Unit=deepstream-night.service
Persistent=false

[Install]
WantedBy=timers.target
```

5) 部署
```sh
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启用定时器（不是服务！）
sudo systemctl enable deepstream-day.timer
sudo systemctl enable deepstream-night.timer

# 启动定时器
sudo systemctl start deepstream-day.timer
sudo systemctl start deepstream-night.timer

# 检查定时器状态
sudo systemctl list-timers --all
```

# 5. (可选)MQTT报文服务

## 安装
```sh
# 可选
# 如果要使用MQTT发送结果，安装mosquitto，可以安装在docker中，也可以安装在宿主机或者局域网其他服务器中
sudo apt-get install libglib2.0 libglib2.0-dev libcjson-dev
wget https://mosquitto.org/files/source/mosquitto-2.0.15.tar.gz
tar -xvf mosquitto-2.0.15.tar.gz
cd mosquitto-2.0.15
make
make install
sudo cp /usr/local/lib/libmosquitto* /opt/nvidia/deepstream/deepstream/lib/
sudo ldconfig
```

运行mosquitto
```sh
adduser --system mosquitto
mosquitto
```

mosquitto配置文件，比如创建一个my_config.conf如下
```conf
allow_anonymous true
listener 1883 0.0.0.0
```

启动
```sh
mosquitto -v -c ./my_config.conf &
```
然后就可以使用mqtt发送和接收消息了

## 作为服务安装并开机自启动
如果要将 mosquitto 作为系统服务运行并设置开机自启动，请按照以下步骤操作：

1. 创建配置文件目录并放置配置文件：
```sh
sudo mkdir -p /etc/mosquitto
sudo cp my_config.conf /etc/mosquitto/
```

2. 创建 systemd 服务文件 `/etc/systemd/system/mosquitto.service`：
```
[Unit]
Description=Mosquitto MQTT Broker
After=network.target

[Service]
Type=simple
User=mosquitto
ExecStart=/usr/local/sbin/mosquitto -v -c /etc/mosquitto/my_config.conf
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

3. 重新加载 systemd 配置并启用服务：
```sh
sudo systemctl daemon-reload
sudo systemctl enable mosquitto.service
sudo systemctl start mosquitto.service
```

4. 检查服务状态：
```sh
sudo systemctl status mosquitto.service
```

5. 如果需要停止服务：
```sh
sudo systemctl stop mosquitto.service
```

6. 彻底取消自动重启（本次与下次开机都不拉起）：
```sh
sudo systemctl stop mosquitto.service
sudo systemctl disable mosquitto.service
```

7. 查看日志
```bash
sudo journalctl -u mosquitto.service
```

# 6 (可选)服务可视化

```bash
sudo apt install cockpit -y
# 启动并启用服务
sudo systemctl enable cockpit.socket
```
浏览器访问 https://服务器IP:9090，使用系统用户名和密码登录即可进入管理界面。
