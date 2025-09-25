## 原始依赖容器
`nvcr.io/nvidia/deepstream:7.1-gc-triton-devel`

## 环境变量配置
```
export CUDA_VER=12.6
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
```

## 时区设置
```
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
echo "Asia/Shanghai" > /etc/timezone
```

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

## 编译
### 安装必要的依赖
```sh
bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh
apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libopencv-dev
apt reinstall libxvidcore4
apt reinstall libmp3lame0
# 中文显示
apt-get install ttf-wqy-microhei
```

### 编译DeepStream-Yolo
```sh
cd DeepStream-Yolo
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

### 编译报文发送插件
```sh
cd /workspace/deepstream-app-custom/src/gst-mynetwork
make
make install
```

### (可选)MQTT报文服务
安装
```sh
# 可选
# 如果要使用MQTT发送结果，安装mosquitto，可以安装在docker中，也可以安装在宿主机或者局域网其他服务器中
apt-get install libglib2.0 libglib2.0-dev
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

### 编译单目标跟踪插件
```sh
cd /workspace/deepstream-app-custom/src/Mixformer_plugin
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

### 编译多帧识别插件
```sh
cd /workspace/deepstream-app-custom/src/gst-videorecognition
mkdir build && cd build
cmake ..
cmake --build .
cmake --install .
```

### 编译主工程
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


## 准备模型
### 目标检测模型
把目标检测模型onnx文件放入src/deepstream-app/models目录下，根据实际的模型名称修改下面的参数：
动态 batch: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> dynamic [min_batch] [opt_batch] [max_batch] [fp16]
```sh
./convert2trt.sh yolov11m_detect_ir_640_v2.onnx yolov11m_detect_ir_640_b4_v2_fp16.engine dynamic 1 4 4 fp16
./convert2trt.sh yolov11m_detect_rgb_640_v5.onnx yolov11m_detect_rgb_640_v5_b4_fp16.engine dynamic 1 4 4 fp16
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

### 二次分类模型
把分类模型比如yolov11m_classify_rgb_b4_v2.onnx放到src/deepstream-app/models目录下，根据实际的模型名称修改下面的参数：
```bash
./convert2trt.sh yolov11m_classify_rgb_b4_v2.onnx yolov11m_classify_rgb_b4_v2_fp16.engine fp16
```

### 单目标跟踪模型
把模型sutrack.onnx文件放入src/Mixformer_plugin/models目录下，

```sh
# 用法: ./convert2trt.sh <ONNX_PATH> <ENGINE_PATH> [fp16]
# 例如: 
./convert2trt.sh ostrack-384-ep300-ce.onnx ostrack-384-ep300-ce_fp16.engine fp16
./convert2trt.sh sutrack.onnx sutrack_fp32.engine
```

<!-- ### 视频识别模型
把模型uniformerv2_softmax.onnx文件放入src/gst-videorecognition/models目录下，根据实际的onnx文件名修改convert2trt.sh
```sh
./convert2trt.sh
``` -->

## 开机自启动
创建和编辑文件/etc/systemd/system/deepstream-compose.service

```
[Unit]
Description=DeepStream Compose Stack
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=oneshot
WorkingDirectory=/home/tl/work/workspace/deepstream-app-custom
RemainAfterExit=yes
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

启用:
```bash
sudo systemctl daemon-reload
sudo systemctl enable deepstream-compose.service
sudo systemctl start deepstream-compose.service
```

状态检查：
```
systemctl status deepstream-compose.service
```

停止:
```bash
docker compose down
```

彻底取消自动重启（本次与下次开机都不拉起）
```
systemctl stop deepstream-compose.service
systemctl disable deepstream-compose.service
```