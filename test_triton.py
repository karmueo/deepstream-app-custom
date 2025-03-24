# test_triton_client.py
import argparse
import numpy as np
from PIL import Image
import tritonclient.http as http_client
import tritonclient.grpc as grpc_client
from torchvision import transforms


idx_to_class = {1: "bird",
                0: "drone"}


def preprocess_image(image_path, input_size=224):
    """预处理流程（必须与训练时完全一致）"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).numpy()[np.newaxis, ...]  # 添加batch维度


def triton_grpc_inference(image_data, model_name, url='localhost:8001'):
    """gRPC协议推理"""
    client = grpc_client.InferenceServerClient(url=url)
    
    # 构建输入
    inputs = [grpc_client.InferInput('input', image_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_data)
    
    # 发送请求
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=[grpc_client.InferRequestedOutput('probabilities')]
    )
    
    return response.as_numpy('probabilities')

def triton_http_inference(image_data, model_name, url='localhost:8000'):
    """HTTP协议推理"""
    client = http_client.InferenceServerClient(url=url)
    
    # 构建输入
    inputs = [http_client.InferInput('input', image_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(image_data)
    
    # 发送请求
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=[http_client.InferRequestedOutput('probabilities')]
    )
    
    return response.as_numpy('probabilities')

def print_results(probs, top_k=5):
    
    # 获取Top-K索引和概率
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    print("\n预测结果：")
    for idx in top_indices:
        print(f"{idx_to_class[idx]:<20} {probs[idx]*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triton Client Test')
    parser.add_argument('--image', type=str, default='drone_test.jpg', help='测试图片路径')
    parser.add_argument('--model', type=str, default='Secondary_Classify', 
                       help='模型名称（与config.pbtxt一致）')
    parser.add_argument('--protocol', choices=['grpc', 'http'], default='grpc',
                       help='通信协议（默认gRPC）')
    args = parser.parse_args()

    # 预处理图像
    input_data = preprocess_image(args.image)
    
    # 选择协议
    if args.protocol == 'grpc':
        preds = triton_grpc_inference(input_data, args.model)
    else:
        preds = triton_http_inference(input_data, args.model)
    
    # 打印结果
    print_results(preds[0])