"""
Linear-probe 评估实验
=====================
核心思想：
  CLIP 的视觉编码器 (encode_image) 就像一台超级 X 光机。
  它可以把任何图片照出一张 512 维的"透视图"（特征向量）。
  
  Linear Probe 要做的是：
  1. 用这台 X 光机把 CIFAR-100 的 50000 张训练图和 10000 张测试图全部"拍片"。
  2. 然后训练一个极其简单的分类器（逻辑回归，本质就是一层全连接）来区分这些片子。
  3. 如果一个简单的分类器就能在 CLIP 的特征上取得很高的准确率，
     说明 CLIP 提取的特征非常好，信息含量极高。

  这就是 Linear Probe 的意义：
  用最弱的分类器，来测量最强的特征。
"""

import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# === 1. 环境准备 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前运行设备: {device}")

# 加载 CLIP 模型
print("正在加载 CLIP 模型...")
model, preprocess = clip.load('ViT-B/32', device)
print("模型加载完成！")

# === 2. 加载数据集 ===
# 注意：这里的 transform 参数直接用 CLIP 自带的 preprocess！
# 这样每张图片在被 DataLoader 取出时，已经被自动预处理好了。
root = os.path.expanduser("~/.cache")
print("正在加载 CIFAR-100 数据集...")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)
print(f"训练集: {len(train)} 张, 测试集: {len(test)} 张")


# === 3. 提取特征 (最耗时的步骤) ===
def get_features(dataset, desc="提取特征"):
    """
    用 CLIP 的视觉编码器批量提取图片特征。
    
    这个函数的核心逻辑很简单：
    1. 用 DataLoader 把数据集按 batch 送入模型。
    2. model.encode_image() 会输出 [batch_size, 512] 的特征。
    3. 把所有 batch 的特征和标签收集起来，拼成大矩阵返回。
    """
    all_features = []
    all_labels = []
    
    # torch.no_grad() 关闭梯度计算，节省显存并加速推理
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100), desc=desc):
            # images: [100, 3, 224, 224] -> 已预处理好的图片批次
            features = model.encode_image(images.to(device))
            # features: [100, 512] -> 每张图的 512 维特征向量

            all_features.append(features)
            all_labels.append(labels)

    # torch.cat: 把所有小批次的结果拼接成一个大张量
    # .cpu().numpy(): 从 GPU 移回 CPU，并转成 NumPy 数组（sklearn 需要 NumPy）
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


print("\n=== 开始提取特征 ===")
print("（在 RTX 4050 上大约需要 1-2 分钟）")

train_features, train_labels = get_features(train, "训练集")
test_features, test_labels = get_features(test, "测试集")

print(f"\n训练集特征形状: {train_features.shape}")  # 预期: (50000, 512)
print(f"测试集特征形状: {test_features.shape}")    # 预期: (10000, 512)


# === 4. 训练逻辑回归分类器 ===
# LogisticRegression 本质就是 y = softmax(Wx + b)，一个全连接层。
# 参数说明：
#   - C=0.316: 正则化强度的倒数。C 越大，正则化越弱，模型越容易过拟合。
#              0.316 是 OpenAI 在论文中通过超参数搜索找到的最优值。
#   - max_iter=1000: 最大迭代次数，保证收敛。
#   - verbose=1: 打印训练过程日志。
print("\n=== 开始训练逻辑回归分类器 ===")
print("（这一步可能需要几分钟，取决于 CPU 性能）")

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)


# === 5. 评估 ===
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"\n{'='*40}")
print(f"  Linear Probe 准确率 = {accuracy:.2f}%")
print(f"{'='*40}")

# 对比参考：
# - Zero-Shot CLIP (ViT-B/32):   ~63% (不训练任何东西)
# - Linear Probe CLIP (ViT-B/32): ~78% (只训练一个线性层)
# - 从头训练 ResNet-50:           ~79% (需要大量 GPU 时间)
# 
# 结论：CLIP 的特征质量非常高！
# 只加一个简单的线性层，就能接近从头训练专业模型的水平！
