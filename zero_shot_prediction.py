
import os
import clip
import torch
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# === 1. 环境准备与模型加载 ===
# 自动检测是否有 GPU，有则使用 cuda，否则使用 cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备: {device}")

# 加载 CLIP 模型 (ViT-B/32)
# model: 神经网络本身
# preprocess: 图片预处理流水线 (Resize -> Crop -> Stay -> Normalize)
print("正在加载 CLIP 模型...")
model, preprocess = clip.load('ViT-B/32', device)
print("模型加载完成！")

# === 2. 数据集加载 (CIFAR-100) ===
# CIFAR-100 包含 100 个类别的 60000 张 32x32 彩色图像
# root: 数据集下载/保存路径
# download: 如果本地没有，是否自动下载
# train: True=训练集, False=测试集 (我们这里只用测试集做验证)
print("正在检查/下载 CIFAR-100 数据集 (第一次运行会自动下载，请耐心等待)...")
# 用户目录下的 .cache 文件夹
data_root = os.path.expanduser("~/.cache") 
cifar100 = CIFAR100(root=data_root, download=True, train=False)

# === 3. 准备 Prompt (提示词工程) ===
# CLIP 的神奇之处在于，我们不能只给它一个单词 "dog"，
# 而是要给它一个完整的句子 "a photo of a dog"。
# 这样模型能理解这是在描述一张照片，准确率会显著提升。
print("正在构建文本 Prompt...")
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
print(f"已构建 {len(cifar100.classes)} 个类别的 Prompt。")

# === 4. 挑选一张图片进行测试 ===
# 我们随机挑一张图，比如第 3637 张 (原论文附录里的那张蛇的图片)
# 你可以修改 image_index 来测试不同的图片
image_index = 3637
image, class_id = cifar100[image_index]
class_name = cifar100.classes[class_id]
print(f"\n选中图片 ID: {image_index}, 真实标签: {class_name}")

# 对图片进行预处理 (变成 tensor 并加到 device)
# unsqueeze(0) 是因为模型需要 batch 维度 [1, 3, 224, 224]
image_input = preprocess(image).unsqueeze(0).to(device)

# === 5. 开始推理 (Zero-Shot Prediction) ===
print("正在进行推理...")
with torch.no_grad():
    # 5.1 计算图片特征 [1, 512]
    image_features = model.encode_image(image_input)
    # 5.2 计算所有 100 个类别的文本特征 [100, 512]
    text_features = model.encode_text(text_inputs)

    # 5.3 归一化 (Cosine Similarity 的前提)
    # 就像我们在第四课学到的，必须除以范数
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 5.4 计算相似度
    # image_features @ text_features.T -> [1, 512] @ [512, 100] -> [1, 100]
    # 得到这张图和 100 个类别的相似度分数
    # * 100.0 是温度系数 (Logit Scale) 的一种手动模拟，让 Softmax 分布更尖锐
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # 5.5 取出前 5 名 (Top-5)
    values, indices = similarity[0].topk(5)

# === 6. 打印结果 ===
print(f"\n=== 预测结果 (真实标签: {class_name}) ===")
print(f"{'预测类别':>16s} : {'置信度':>8s}")
print("-" * 30)
for value, index in zip(values, indices):
    predicted_class = cifar100.classes[index]
    confidence = 100 * value.item()
    # 打印格式：类别名称 (右对齐) : 置信度
    print(f"{predicted_class:>16s}: {confidence:.2f}%")

