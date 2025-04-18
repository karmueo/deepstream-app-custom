import argparse
import numpy as np
import cv2
from PIL import Image
import tritonclient.http as http_client
import tritonclient.grpc as grpc_client
from torchvision import transforms

# 定义类别映射（根据模型实际输出调整）
idx_to_class = {i: f"class_{i}" for i in range(400)}  # 假设有400个类别

def preprocess_video(video_path, num_frames=250, input_size=224):
    """
    从视频文件中均匀采样 num_frames 帧，并进行预处理
    返回: shape [1, num_frames, 3, input_size, input_size] 的numpy数组
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频没有帧数据")

    # 计算均匀采样的帧索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    # 预处理变换：使用PIL Image和torchvision
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),   # 转为[0,1]的Tensor且格式为(C,H,W)
    ])
    
    current_frame = 0
    frame_dict = {}
    # 预读所有帧，同时存入字典，避免多次seek（视频较小时适用）
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_dict[current_frame] = frame
        current_frame += 1
    cap.release()

    # 对每个选定帧进行预处理
    for idx in indices:
        if idx in frame_dict:
            frame = frame_dict[idx]
        else:
            # 如果索引超过，需要用最后一帧补齐
            frame = frame_dict[current_frame-1]
        # OpenCV读取的是BGR格式，转成RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        tensor = transform(pil_img)   # 得到 (3, input_size, input_size)
        frames.append(tensor.numpy())
    
    # 组合为 (num_frames, 3, input_size, input_size)，再加上batch维度
    video_tensor = np.stack(frames, axis=0)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    return video_tensor.astype(np.float32)

def triton_grpc_inference(video_data, model_name, url='localhost:8001'):
    """使用gRPC协议进行推理"""
    client = grpc_client.InferenceServerClient(url=url)
    
    # 构造输入
    inputs = [grpc_client.InferInput('input', video_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(video_data)
    
    # 发送请求，注意输出名称需与 config.pbtxt 中配置一致（本例中为 "output"）
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=[grpc_client.InferRequestedOutput('output')]
    )
    return response.as_numpy('output')

def triton_http_inference(video_data, model_name, url='localhost:8000'):
    """使用HTTP协议进行推理"""
    client = http_client.InferenceServerClient(url=url)
    
    # 构造输入
    inputs = [http_client.InferInput('input', video_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(video_data)
    
    # 发送请求
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=[http_client.InferRequestedOutput('output')]
    )
    return response.as_numpy('output')

def print_results(probs, top_k=5):
    # 获取Top-K索引和概率
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    print("\n预测结果：")
    for idx in top_indices:
        print(f"{idx_to_class.get(idx, 'unknown'):<20} {probs[idx]*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triton Video Classification Client Test')
    parser.add_argument('--video', type=str, default='arm_wrestling.mp4', help='测试视频路径')
    parser.add_argument('--model', type=str, default='Video_Classify', 
                        help='模型名称，应与config.pbtxt中一致')
    parser.add_argument('--protocol', choices=['grpc', 'http'], default='grpc',
                        help='通信协议（默认gRPC）')
    args = parser.parse_args()

    # 视频预处理：处理后形状为 [1, 250, 3, 224, 224]
    video_data = preprocess_video(args.video, num_frames=250, input_size=224)

    # 选择协议进行推理
    if args.protocol == 'grpc':
        preds = triton_grpc_inference(video_data, args.model)
    else:
        preds = triton_http_inference(video_data, args.model)

    # 打印结果（此处取batch的第一个输出）
    print_results(preds[0])