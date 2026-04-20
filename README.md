
# 1. 环境安装

## 1.1 Jetson环境

### 1.1.1 安装依赖

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

### 1.1.2 安装Deepstream

从官网下载deepstream_sdk_v7.1.0_jetson.tbz2

```bash
sudo tar -xvf deepstream_sdk_v7.1.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-7.1
sudo ./install.sh
sudo ldconfig
```

### 1.1.3 配置环境变量

```bash
export CUDA_VER=12.6
```
## 1.2 服务器环境

### 1.2.1 安装依赖包

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

### 1.2.2 安装显卡驱动
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

### 1.2.3 安装TensorRT
下载地址: https://developer.nvidia.com/tensorrt/download/10x。这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

### 1.2.4 安装Deepstream SDK
下载地址: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream?version=8.0, 这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

# 2. 编译安装

## 2.1 编译DeepStream-Yolo

```sh
cd DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

## 2.2 编译报文发送插件

```sh
cd src/gst-udpmulticast_sink
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 2.3 编译报文接收插件

```sh
cd src/gst-udpjson_meta
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 2.4 编译单目标跟踪插件
```sh
cd sot_plugin
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
sudo cmake --install .
```

## 2.5 编译多帧目标识别插件
```sh
cd src/gst-videorecognition
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build .
sudo cmake --install .
```

## 2.6 编译717设备控制插件
```sh
cd src/gst-cuavcontrolsink
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build .
sudo cmake --install .
```

## 2.7 编译主程序

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../src/deepstream-app
cmake --build .
sudo cmake --install .
```

## 2.8 添加环境变量
```sh
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$GST_PLUGIN_PATH
```

# 3. 准备模型

## 3.1 目标检测模型
把目标检测模型onnx文件放入src/deepstream-app/models目录下，根据实际的模型名称(这里模型名称为yolo26n_rgb_352_2c_v3.onnx)执行下面的命令：

```bash
# 假设模型名称为yolo26n_rgb_352_2c_v3.onnx
./convert2trt.sh yolo26n_rgb_352_2c_v3.onnx yolo26n_rgb_352_2c.engine fp16
```

## 3.2 单目标跟踪模型
把onnx模型文件放入`src/sot_plugin/models`目录下，

```sh
# 用法: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
# 例如: 
./convert2trt.sh nanotrack_head.onnx nanotrack_head_fp16.engine fp16
./convert2trt.sh nanotrack_backbone.onnx nanotrack_backbone_fp16.engine fp16
./convert2trt.sh nanotrack_backbone_search.onnx nanotrack_backbone_search_fp16.engine fp16
```

## 3.3 多帧识别模型
把onnx模型如放到`src/gst-videorecognition/models`目录下，使用`./convert2trt.sh`转换，类似前面的转换操作

```bash
# 假设onnx模型名称为x3d_v9_3cls_simplified.onnx
./convert2trt.sh x3d_v9_3cls_simplified.onnx x3d.engine fp16
```

# 4.运行程序

## 4.1 检查文件是否准备好
确保下面命令运行没有问题，正确输出
```bash
ls /opt/nvidia/deepstream/deepstream/bin/deepstream-app
ls /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml
ls /opt/nvidia/deepstream/deepstream/deepstream-app-ls /opt/nvidia/deepstream/deepstream/sot_plugin/models/nanotrack_backbone_fp16.enginecustom/models/yolo26n_rgb_352_2c.engine
ls /opt/nvidia/deepstream/deepstream/deepstream-app-custom/models/yolo26n_rgb_352_2c.engine
ls /opt/nvidia/deepstream/deepstream/sot_plugin/models/nanotrack_head_fp16.engine
ls /opt/nvidia/deepstream/deepstream/sot_plugin/models/nanotrack_backbone_search_fp16.engine
```

## 4.2 配置文件

### 4.2.1 主配置文件

默认配置文件为`/opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml`，该文件其实是`src/deepstream-app/configs/yml/app_config.yml`的软链接。
常用修改一下配置：

| 配置项 | 说明 |
| --- | --- |
| application | 应用程序全局配置，主要修改smart-rec-detect-default，0关闭录像，1开启录像，同时在source的csv配置中需要把smart-record置为3 |
| source | 视频源CSV配置文件路径（包含RTSP/文件流地址等）|
| sink0 | RTSP输出配置，enable设1开启，设0关闭，只有在NX的板子或者服务器上才能开启 |
| sink1 | 本地窗口显示配置，如果是nano，把 enable设1，可以在hdmi输出中显示窗口 |
| sink2 | UDP组播输出配置，enable设1开启，设0关闭 |
| sink5 | 717设备闭环控制配置，enable设1开启，设0关闭 |
| primary-gie | 检测配置，详见配置文件注释 |
| tracker | 跟踪配置，详见配置文件注释 |
| videorecognition | 视频识别模块配置（多帧动作/行为识别），详见配置文件注释 |
| udpjsonmeta | UDP 组播json报文接收配置，注意不同机器/板子网卡名称需要根据实际情况修改 |

更多配置情参考[Deepstream 7.1 官方文档](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_ref_app_deepstream.html)

### 4.2.1 视频源配置

视频源通过 CSV 文件配置，每行对应一个视频源（对应 DeepStream 的 `[source0]`、`[source1]`... 分组）。CSV 表头及各字段说明如下：

| 字段 | 说明 | 类型与取值 | 示例 |
| --- | --- | --- | --- |
| enable | 启用或禁用该视频源 | Boolean: `0`=禁用, `1`=启用 | `1` |
| type | 视频源类型 | Integer: `1`=Camera(V4L2), `2`=URI(文件/HTTP), `3`=MultiURI, `4`=RTSP, `5`=Camera(CSI,仅Jetson) | `4` |
| uri | 视频流地址。支持文件路径(`file:///`)、HTTP、RTSP；MultiURI 时可用 `%d` 格式指定多源 | String | `rtsp://192.168.1.12`, `file:///home/user/video.mp4` |
| num-sources | 源数量，仅 type=3(MultiURI) 时有效 | Integer, ≥0 | `1` |
| gpu-id | 指定使用的 GPU 编号（多 GPU 环境） | Integer, ≥0 | `0` |
| cudadec-memtype | CUDA 解码内存类型。`0`=设备内存, `1`=主机锁页内存, `2`=统一内存。仅 type=2/3/4 有效 | Integer: 0, 1, 2 | `0` |
| rtsp-reconnect-interval-sec | RTSP 重连间隔（秒）。设为 `0` 禁用重连。仅 type=4 有效 | Integer, ≥0 | `3` |
| rtsp-reconnect-attempts | RTSP 最大重连次数。`-1`=无限重试。仅 type=4 且 reconnect-interval>0 时有效 | Integer, ≥-1 | `-1` |
| select-rtp-protocol | RTP 传输协议。`0`=UDP+UDP组播+TCP, `4`=仅TCP。仅 type=4 有效 | Integer: 0, 4 | `0` |
| smart-record | 智能录像触发方式。`0`=禁用, `3`检测到目标后自动记录 | Integer: 0, 3 | `3` |
| smart-rec-dir-path | 智能录像文件保存目录 | String | `/home/tl/data2/smart_rec_rgb` |
| smart-rec-duration | 智能录像时长（秒） | Integer, ≥0 | `30` |
| smart-rec-start-time | 智能录像回溯起始时间（秒），从当前时间往前推算 | Integer, ≥0 | `3` |

> 更多属性（如 `camera-width`、`camera-height`、`latency`、`drop-frame-interval`、`nvbuf-memory-type` 等）请参考官方文档：[Source Group](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_ref_app_deepstream.html#source-group)

配置示例参考`src/deepstream-app/configs/yml/sources.csv`和`src/deepstream-app/configs/yml/file_sources.csv`。

## 4.3 启动程序

执行以下命令启动程序

```bash
/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml
```

## 4.4 程序开机自启动

1) 创建服务文件 `/etc/systemd/system/deepstream-app-rgb.service`

```ini
[Unit]
Description=DeepStream RGB App
# 网络就绪后再启动
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# 指定运行用户和组
User=nvidia
Group=nvidia
WorkingDirectory=/opt/nvidia/deepstream/deepstream
ExecStart=/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

2) 重新加载并启用/启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepstream-app-rgb.service
sudo systemctl start deepstream-app-rgb.service
```

3) 查看状态与日志

```bash
sudo systemctl status deepstream-app-rgb.service
sudo journalctl -u deepstream-app-rgb.service -f
```

4) 停止和取消自启动

```bash
sudo systemctl stop deepstream-app-rgb.service
sudo systemctl disable deepstream-app-rgb.service
```