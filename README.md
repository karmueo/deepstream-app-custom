
# 1. 安装依赖

## 安装依赖包

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
## 安装显卡驱动
pass

## 安装CUDA Toolkit
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
"
```

## 安装TensorRT
下载地址: https://developer.nvidia.com/tensorrt/download/10x。这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

## 安装Deepstream SDK
下载地址: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream?version=8.0, 这里使用的版本是: nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2404-10.10.0-cuda-12.9/nv-tensorrt-local-CD20EDBE-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
```

# 2.安装
## 获取项目
```sh
git clone --recurse-submodules git@github.com:karmueo/deepstream-app-custom.git
# git submodule init
# git submodule update
```

<!-- ## 安装nvdsinfer_yolo_efficient_nms
```bash
export CUDA_VER=12.6
cd /nvdsinfer_yolo_efficient_nms

# Set the CUDA_VER environment variable
make
make install
``` -->

## 编译DeepStream-Yolo
```sh
cd DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

## 编译报文发送插件
```sh
cd src/gst-udpmulticast_sink
mkdir build && cd build
cmake ..
cmake --build .
sudo cmake --install .
```

## 编译单目标跟踪插件
```sh
cd sot_plugin
mkdir build && cd build
cmake ..
cmake --build .
sudo cmake --install .
```

## (可选)MQTT报文服务
安装
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

## 编译多帧识别插件
```sh
cd /workspace/deepstream-app-custom/src/gst-videorecognition
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

## 编译主工程
```sh
cd /workspace/deepstream-app-custom/src/deepstream-app
make
```
如果要使用vscode Makefiles-tools插件进行调试开发，在`.vscode/settings.json`中添加如下：
```json
{
    ...
    "makefile.makeDirectory": "src/deepstream-app",
    "makefile.launchConfigurations": [
        {
            "cwd": "/workspace/deepstream-app-custom/src/deepstream-app",
            "binaryPath": "/workspace/deepstream-app-custom/src/deepstream-app/deepstream-app",
            "binaryArgs": [
                "-c",
                "/workspace/deepstream-app-custom/src/deepstream-app/configs/ir_app_config.txt"
            ]
        }
    ],
    ...
}
```
# 3. 准备模型
## 目标检测模型
把目标检测模型onnx文件放入src/deepstream-app/models目录下，根据实际的模型名称修改下面的参数：
动态 batch: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> dynamic [min_batch] [opt_batch] [max_batch] [fp16]
```sh
./convert2trt.sh yolov11m_detect_ir_640_v2.onnx yolov11m_detect_ir_640_b4_v2_fp16.engine fp16
./convert2trt.sh yolov11m_detect_rgb_640_v7.onnx yolov11m_detect_rgb_640_v7_b5_fp16.engine fp16
```
然后根据实际的engine文件名修改src/deepstream-app/configs/config_infer_primary_yoloV11.txt中model-engine-file的值

<!-- 参考https://github.com/laugh12321/TensorRT-YOLO/tree/main安装trtyolo cli
转换为end2end.onnx模型

```bash 
# 把通过ultralytics训练的yolov11模型转换为end2end.onnx的模型
trtyolo export -w yolov11.pt -v ultralytics -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1
```
生成的模型在output目录下，把模型名字改为yolov11_ir_drones_p2_single_target_end2end.onnx，然后放到triton_model/Primary_Detect/1目录下。
使用脚本转换为.engine格式的模型。
```bash
./convert2trt.sh
```
 -->

## 二次分类模型
把分类模型比如yolov11m_classify_rgb_b4_v2.onnx放到src/deepstream-app/models目录下，根据实际的模型名称修改下面的参数：
```bash
./convert2trt.sh yolov11m_classify_rgb_b4_v2.onnx yolov11m_classify_rgb_b4_v2_fp16.engine fp16
./convert2trt.sh yolov11m_classify_ir_b4_v2.onnx yolov11m_classify_ir_b4_v2_fp16.engine fp16
```

## 单目标跟踪模型
把模型sutrack.onnx文件放入src/sot_plugin/models目录下，

```sh
# 用法: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
# 例如: 
./convert2trt.sh ostrack-384-ep300-ce.onnx ostrack-384-ep300-ce_fp16.engine fp16
./convert2trt.sh sutrack.onnx sutrack_fp32.engine
./convert2trt.sh mixformerv2_online_base.onnx mixformerv2_online_base_fp32.engine
./convert2trt.sh mixformerv2_online_small.onnx mixformerv2_online_base_fp16.engine fp16
./convert2trt.sh mixformerv2_online_small.onnx mixformerv2_online_small_fp32.engine
```

<!-- ### 视频识别模型
把模型uniformerv2_softmax.onnx文件放入src/gst-videorecognition/models目录下，根据实际的onnx文件名修改convert2trt.sh
```sh
./convert2trt.sh
``` -->

# 4.开机自启动

## 程序开机自启动
将如下命令作为 systemd 服务开机自启动：

```bash
/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/rgb_app_config.txt
```

步骤如下：

1) 创建服务文件 `/etc/systemd/system/deepstream-app-rgb.service`

```ini
[Unit]
Description=DeepStream RGB App
# 网络就绪后再启动，如依赖 MQTT，请追加 mosquitto.service
After=network-online.target mosquitto.service
Wants=network-online.target mosquitto.service

[Service]
Type=simple
WorkingDirectory=/opt/nvidia/deepstream/deepstream
ExecStart=/opt/nvidia/deepstream/deepstream/bin/deepstream-app -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/rgb_app_config.txt
Restart=on-failure
RestartSec=5

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

注意：
- 如果你的应用依赖其他服务（如 MQTT），可在 `[Unit]` 中追加：`After=mosquitto.service` 与/或 `Wants=mosquitto.service`。
- 若启用 `User=...` 以非 root 运行，请确保该用户有 GPU 与摄像头、模型及日志目录等资源的访问权限。

## *可选，服务可视化
```bash
sudo apt install cockpit -y
# 启动并启用服务
sudo systemctl enable cockpit.socket
```
浏览器访问 https://服务器IP:9090，使用系统用户名和密码登录即可进入管理界面。