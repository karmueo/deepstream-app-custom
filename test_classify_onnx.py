import onnxruntime
from PIL import Image
import torch
from torchvision import transforms

# 配置参数
ONNX_PATH = 'src/deepstream-app/models/efficientnet_110_with_softmax.onnx'
IMG_PATH = 'kite_test.jpg'          # 测试图片路径
INPUT_SIZE = 224                     # 与导出时一致

idx_to_class = {0: "unkonw",
                1: "kite",
                2: "plane",
                3: "bird",
                4: "drone"}

# 数据预处理（必须与训练完全一致！）


def preprocess(image):
    image_size = 224
    tfms = transforms.Compose([transforms.Resize((image_size, image_size)),
                               transforms.ToTensor()])

    return tfms(image).unsqueeze(0).numpy()  # 转为numpy数组


# 加载图像
img = Image.open(IMG_PATH).convert('RGB')

# 预处理
input_tensor = preprocess(img)

# 创建ONNX Runtime会话
ort_session = onnxruntime.InferenceSession(
    ONNX_PATH,
    providers=['CPUExecutionProvider']  # 使用GPU改为['CUDAExecutionProvider']
)

# 运行推理
ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
probabilities = ort_session.run(None, ort_inputs)[0]  # [batch, num_classes]
# Convert probabilities to a PyTorch tensor
probabilities = torch.tensor(probabilities)

# 后处理
# probabilities = torch.softmax(torch.tensor(ort_outputs), dim=1)
top5_probs, top5_ids = torch.topk(probabilities, 5)

# 打印结果
print('----- Predictions -----')
for i in range(5):
    class_name = idx_to_class[top5_ids[0][i].item()]
    prob = top5_probs[0][i].item() * 100
    print(f'{class_name:<25} {prob:.2f}%')
