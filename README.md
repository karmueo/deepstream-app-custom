## 安装nvdsinfer_yolo_efficient_nms
```bash
cd /workspace
# Clone the repository
git clone https://github.com/karmueo/nvdsinfer_yolo_efficient_nms.git

# Set the CUDA_VER environment variable
export CUDA_VER=12.6

make
make install

```

## 准备模型
### 目标检测模型
参考https://github.com/laugh12321/TensorRT-YOLO/tree/main安装trtyolo cli
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
把分类模型efficientnet_110_with_softmax.onnx放到triton_model/Secondary_Classify/1目录下

