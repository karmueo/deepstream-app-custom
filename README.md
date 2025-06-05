## 原始依赖容器
`nvcr.io/nvidia/deepstream:7.1-gc-triton-devel`

## 环境变量配置
```
export CUDA_VER=12.6
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
```

## 获取项目
```sh
git clone --recurse-submodules git@github.com:karmueo/deepstream-app-custom.git
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
                "/workspace/deepstream-app-custom/src/deepstream-app/configs/deepstream_app_config.txt"
            ]
        }
    ],
    ...
}
```


## 准备模型
### 目标检测模型
把目标检测模型onnx文件放入src/deepstream-app/models目录下，根据实际的onnx文件名修改convert2trt.sh
```sh
./convert2trt.sh
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
### 分类模型
把分类模型efficientnet_110_with_softmax.onnx放到triton_model/Secondary_Classify/1目录下 -->

### 单目标跟踪模型
把模型sutrack.onnx文件放入src/Mixformer_plugin/models目录下，根据实际的onnx文件名修改convert2trt.sh
```sh
./convert2trt.sh
```

### 视频识别模型
把模型uniformerv2_softmax.onnx文件放入src/gst-videorecognition/models目录下，根据实际的onnx文件名修改convert2trt.sh
```sh
./convert2trt.sh
```

## 运行程序
