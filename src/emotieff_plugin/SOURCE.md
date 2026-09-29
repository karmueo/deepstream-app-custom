# 来源与限制

情绪预处理、裁剪方式、类别顺序、softmax 和 TensorRT FP16 构建参数依据 EmotiEffLib 的 Jetson C++ 示例实现。GPU 视频表面处理与 DeepStream 元数据接入参考本仓库 `src/face_detect_plugin`。运行时不访问参考源码目录。

两份 ONNX 来自本机 EmotiEffLib 的 `models/affectnet_emotions/onnx/`。引擎为本机生成的 TensorRT 序列化文件，不可视作跨 GPU 或跨 TensorRT 版本的通用文件。EmotiEffLib 主仓库代码为 Apache-2.0；预训练模型的授权应按模型来源单独确认。
