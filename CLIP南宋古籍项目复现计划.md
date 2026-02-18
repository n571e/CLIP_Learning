# CLIP × 南宋文化多模态检索 · 详细项目计划

> **一句话**：基于 Chinese-CLIP，通过 LoRA 微调 + LLM 辅助古文处理，构建南宋文化领域的图文检索系统。

---

## 一、技术路线总览

```
古籍原文 (梦梁录/武林旧事)
      ↓  通义千问 API (免费)
结构化标注 (实体抽取 + 多语言翻译)
      ↓
图文对数据集 (tsv + jsonl → LMDB)
      ↓
Chinese-CLIP (ViT-B-16 + RoBERTa-wwm)
      ↓  注入 LoRA 适配层
对比学习微调 (冻结骨干，仅训练 ~1% 参数)
      ↓
检索评估 (Recall@K) + Gradio Demo
```

### 为什么用 Chinese-CLIP 而非原版 CLIP？

原版 CLIP 的 BPE tokenizer 将每个汉字拆为 **3 个 UTF-8 字节 token**，语义完全丢失。  
Chinese-CLIP 用 **RoBERTa-wwm** 替换文本编码器，原生支持中文分词。  
视觉编码器 ViT 架构与原版一致，你之前学的 `model.py` 知识完全可复用。

---

## 二、阶段一：环境搭建（第 1-3 天）

### 2.1 Fork 与克隆

```bash
# 1. 在 GitHub 上 Fork: https://github.com/OFA-Sys/Chinese-CLIP
# 2. 克隆到本地
git clone https://github.com/你的用户名/Chinese-CLIP.git
cd Chinese-CLIP

# 3. 创建 conda 环境
conda create -n clip-song python=3.10 -y
conda activate clip-song
pip install -r requirements.txt
pip install -e .   # 安装 cn_clip 包，之后可以 import cn_clip
```

### 2.2 下载预训练权重

推荐 **ViT-B-16**（适合 6GB 显存）：

```bash
# 创建数据目录（Chinese-CLIP 的工作区约定）
mkdir -p ../clip_data/pretrained_weights
mkdir -p ../clip_data/datasets
mkdir -p ../clip_data/experiments

# 下载 ViT-B-16 预训练 ckpt（约 1GB）
# 方式一：从 HuggingFace 下载
# https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16
# 方式二：从 ModelScope 下载（国内推荐）
pip install modelscope
```

### 2.3 验证安装

```python
# test_chinese_clip.py
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image

print("可用模型:", available_models())
# ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../clip_data/pretrained_weights/')
model.eval()

# 测试中文编码！
text = clip.tokenize(["西湖美景", "德寿宫遗址", "南宋古画"]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    print(f"文本特征形状: {text_features.shape}")  # 期望: [3, 512]
    print("✅ 中文编码成功！")
```

### 2.4 代码对比学习

对照阅读，理解差异：

| 组件 | 原版 CLIP (`clip/model.py`) | Chinese-CLIP (`cn_clip/clip/model.py`) |
|------|----------------------------|----------------------------------------|
| 视觉编码器 | VisionTransformer | **相同**的 ViT 架构 |
| 文本编码器 | 自定义 Transformer + BPE | **RoBERTa-wwm** (来自 HuggingFace) |
| Tokenizer | `simple_tokenizer.py` (英文 BPE) | `bert_tokenizer.py` (中文 WordPiece) |
| 对比损失 | `forward()` 中的 cosine sim | 相同逻辑，在 `training/main.py` |

---

## 三、阶段二：LLM 古文数据处理（第 4-8 天）

### 3.1 注册通义千问 API（免费）

1. 访问 [阿里云百炼](https://bailian.console.aliyun.com/)
2. 开通百炼服务（免费额度，180 天有效）
3. 创建 API Key

> 备选免费方案：智谱 GLM-4-flash（免费额度）、DeepSeek API（低价）

### 3.2 古文文本收集

以下南宋古籍全文网上公开可查：

| 文献 | 内容 | 来源 |
|------|------|------|
| 《梦梁录》 | 临安城市生活、建筑、地理 | 中国哲学书电子化计划 (ctext.org) |
| 《武林旧事》 | 宫廷、园林、节日、地标 | 同上 |
| 《咸淳临安志》 | 方志，详细地理信息 | 同上 |

手动复制 20-30 段包含地理信息的段落即可（不需要全文），存为 `raw_texts.json`：

```json
[
  {
    "id": 1,
    "source": "梦梁录",
    "chapter": "卷二·都城",
    "text": "德寿宫在望仙桥东，元系秦太师赐第。"
  },
  {
    "id": 2,
    "source": "武林旧事",
    "chapter": "卷五·西湖游幸",
    "text": "西湖之胜，画船箫鼓，游人如织。"
  }
]
```

### 3.3 LLM 批处理脚本

```python
# scripts/process_ancient_text.py
import json
import os
from openai import OpenAI  # 通义千问兼容 OpenAI 接口

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

SYSTEM_PROMPT = """你是一位南宋历史与古典文献专家。请从给定的古文段落中：
1. 抽取所有地理实体（地名、建筑名、山水名），标注类型
2. 分析空间关系（方位、距离、相对位置）
3. 翻译为现代中文白话文
4. 翻译为英文
5. 生成一条适合图像检索的描述（中文Prompt）

输出严格JSON格式。"""

def process_text(text_entry):
    response = client.chat.completions.create(
        model="qwen-turbo",  # 免费额度支持的模型
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"古文原文：{text_entry['text']}\n来源：{text_entry['source']}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# 批量处理
with open("data/raw_texts.json", "r", encoding="utf-8") as f:
    raw_texts = json.load(f)

results = []
for entry in raw_texts:
    try:
        result = process_text(entry)
        result["id"] = entry["id"]
        result["original_text"] = entry["text"]
        result["source"] = entry["source"]
        results.append(result)
        print(f"✅ 处理完成: {entry['id']}")
    except Exception as e:
        print(f"❌ 处理失败: {entry['id']}, 错误: {e}")

with open("data/annotations.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 3.4 预期输出格式

```json
{
  "id": 1,
  "original_text": "德寿宫在望仙桥东，元系秦太师赐第。",
  "source": "梦梁录",
  "entities": [
    {"name": "德寿宫", "type": "宫殿"},
    {"name": "望仙桥", "type": "桥梁"}
  ],
  "spatial_relation": "德寿宫位于望仙桥东侧",
  "modern_chinese": "德寿宫位于望仙桥东面，原本是秦桧的赐第。",
  "english": "Deshou Palace is east of Wangxian Bridge, originally the granted residence of Prime Minister Qin.",
  "search_prompt": "南宋临安古画中的德寿宫，宫殿建筑，望仙桥东侧"
}
```

---

## 四、阶段三：图像采集 + 数据集构建（第 9-13 天）

### 4.1 图像来源（全部免费公开）

| 来源 | 内容 | 获取方式 | 目标数量 |
|------|------|----------|----------|
| [故宫博物院数字文物库](https://digicol.dpm.org.cn/) | 南宋绘画 | 网站截图/下载 | 40-60 张 |
| [台北故宫 Open Data](https://theme.npm.edu.tw/opendata/) | 宋代书画 | 公开下载 | 20-40 张 |
| 谭其骧《中国历史地图集》 | 南宋地图 | 公开扫描版裁剪 | 15-25 张 |
| 百度/维基百科 | 现代杭州地标照 | CC协议图片 | 25-40 张 |

### 4.2 图文配对逻辑

每张图像配对多条文本（来自 LLM 处理结果）：

```
德寿宫遗址照片.jpg
├── 古文: "德寿宫在望仙桥东，元系秦太师赐第"
├── 白话: "德寿宫位于望仙桥东面"
├── 英文: "Deshou Palace is east of Wangxian Bridge"
└── Prompt: "南宋临安古画中的德寿宫"

西湖古画.jpg
├── 古文: "西湖之胜，画船箫鼓，游人如织"
├── 白话: "西湖景色优美，画船与箫鼓声相映"
└── Prompt: "南宋古画中描绘的西湖胜景"
```

### 4.3 转换为 Chinese-CLIP 数据格式

Chinese-CLIP 要求 `tsv` (图像) + `jsonl` (文本) 格式：

```python
# scripts/build_dataset.py
import base64, json
from PIL import Image
from io import BytesIO

def image_to_base64(image_path):
    """将图片转为 base64 字符串"""
    img = Image.open(image_path)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 生成 train_imgs.tsv
# 格式: image_id\tbase64_string
with open("train_imgs.tsv", "w") as f:
    for img_id, img_path in enumerate(image_paths):
        b64 = image_to_base64(img_path)
        f.write(f"{img_id}\t{b64}\n")

# 生成 train_texts.jsonl
# 格式: {"text_id": int, "text": str, "image_ids": [int]}
with open("train_texts.jsonl", "w", encoding="utf-8") as f:
    for item in annotations:
        for text_type in ["original_text", "modern_chinese", "search_prompt"]:
            entry = {
                "text_id": next_text_id(),
                "text": item[text_type],
                "image_ids": [item["image_id"]]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

```bash
# 转换为 LMDB（Chinese-CLIP 要求）
python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir ../clip_data/datasets/SongDynasty \
    --splits train,valid
```

---

## 五、阶段四：LoRA 微调实现（第 14-19 天）

### 5.1 LoRA 模块

> **核心代码**：Chinese-CLIP 官方暂不支持 LoRA，需要自己实现。  
> 这正是项目的技术亮点——你帮 Chinese-CLIP 补上了 LoRA 能力。

```python
# cn_clip/clip/lora.py
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """在冻结的 Linear 层上叠加 LoRA 低秩更新"""
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # 冻结原始权重
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # A 用 Kaiming 初始化，B 用零初始化 → 保证初始 ΔW = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / rank

    def forward(self, x):
        # 原始输出 + LoRA 增量
        original_output = self.original_linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output


def inject_lora(model, rank=4, alpha=16.0):
    """
    将 LoRA 注入到 Chinese-CLIP 的 Attention 层的 Q 和 V 投影。
    
    Chinese-CLIP 使用 nn.MultiheadAttention，其 Q/K/V 合并在 in_proj_weight 中。
    为简化实现，我们对 out_proj 注入 LoRA（效果类似）。
    或者遍历找到 Attention 的 Linear 层。
    """
    lora_params = []
    
    for name, module in model.named_modules():
        # 针对视觉编码器和文本编码器的 Attention output projection
        if isinstance(module, nn.MultiheadAttention):
            # 替换 out_proj
            old_proj = module.out_proj
            new_proj = LoRALinear(old_proj, rank=rank, alpha=alpha)
            module.out_proj = new_proj
            lora_params.extend([new_proj.lora_A, new_proj.lora_B])
    
    return lora_params


def get_lora_state_dict(model):
    """提取仅 LoRA 相关的参数，用于保存轻量 checkpoint"""
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}
```

### 5.2 训练脚本

```python
# train_lora.py
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from cn_clip.clip.lora import inject_lora
from torch.utils.data import DataLoader

# 1. 加载模型
device = "cuda"
model, preprocess = load_from_name("ViT-B-16", device=device)

# 2. 注入 LoRA
lora_params = inject_lora(model, rank=4, alpha=16.0)

# 3. 冻结所有非 LoRA 参数
for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False

# 统计可训练参数
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total:,} | 可训练(LoRA): {trainable:,} | 占比: {trainable/total*100:.2f}%")

# 4. 优化器 (只优化 LoRA 参数)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

# 5. 训练循环（简化版，实际请参考 Chinese-CLIP 的 training/main.py）
model.train()
for epoch in range(30):
    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)
        
        # 前向
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # InfoNCE 对比损失
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(len(images), device=device)
        loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
        loss_t2i = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 5.3 显存优化（6GB RTX 4050）

```bash
# 关键参数调整：
batch_size=8          # 小 batch
--grad-checkpointing  # 重计算，省显存
--fp16                # 半精度训练
accum_freq=4          # 梯度累积，等效 batch_size=32
```

---

## 六、阶段五：实验 + Demo + 包装（第 20-28 天）

### 6.1 消融实验设计

| 编号 | 配置 | 说明 |
|------|------|------|
| Exp1 | Chinese-CLIP Zero-Shot (原始) | 基线：不做任何训练 |
| Exp2 | Chinese-CLIP + 全参数微调 | 对比项：训练全部参数 |
| Exp3 | Chinese-CLIP + LoRA (rank=4) | **核心实验** |
| Exp4 | Exp3 + Prompt 工程 | 验证 Prompt 模板效果 |
| Exp5 | LoRA rank=2/4/8/16 对比 | LoRA 超参分析 |

### 6.2 评估指标

```python
# evaluate.py
def compute_recall_at_k(image_features, text_features, k_list=[1, 5, 10]):
    """计算 Text→Image 和 Image→Text 的 Recall@K"""
    sim_matrix = text_features @ image_features.T  # [N_text, N_image]
    
    results = {}
    for k in k_list:
        # Text → Image
        topk_indices = sim_matrix.topk(k, dim=1).indices
        correct = sum(i in topk_indices[i] for i in range(len(sim_matrix)))
        results[f"T2I_R@{k}"] = correct / len(sim_matrix) * 100
        
        # Image → Text
        topk_indices_t = sim_matrix.T.topk(k, dim=1).indices
        correct_t = sum(i in topk_indices_t[i] for i in range(sim_matrix.shape[1]))
        results[f"I2T_R@{k}"] = correct_t / sim_matrix.shape[1] * 100
    
    return results
```

### 6.3 Gradio 检索 Demo

```python
# demo.py
import gradio as gr
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from PIL import Image

model, preprocess = load_from_name("ViT-B-16", device="cuda")
# ... 加载 LoRA 权重 ...

def text_to_image_search(query_text, top_k=5):
    """古文描述 → 检索最相似图像"""
    text_token = clip.tokenize([query_text]).to("cuda")
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    
    # 与预计算的图像特征比较
    sims = (text_feat @ image_features_db.T).squeeze()
    topk = sims.topk(top_k)
    return [(image_paths[i], f"相似度: {s:.3f}") for s, i in zip(topk.values, topk.indices)]

demo = gr.Interface(
    fn=text_to_image_search,
    inputs=gr.Textbox(label="输入古文描述", placeholder="西湖之胜，画船箫鼓"),
    outputs=gr.Gallery(label="检索结果"),
    title="南宋文化多模态检索系统",
    description="基于 Chinese-CLIP + LoRA 的古文→图像跨模态检索"
)
demo.launch()
```

### 6.4 README 模板

```markdown
# 🏯 SongCLIP: 南宋文化多模态检索系统

基于 Chinese-CLIP + LoRA 微调的南宋文化图文检索系统。

## 🔬 技术亮点
- **LoRA 微调**：仅训练 ~1% 参数，适配南宋文化垂直领域
- **LLM 辅助标注**：利用通义千问自动抽取古文地理实体
- **跨时态检索**：支持古文描述 → 现代照片的检索

## 📊 实验结果
| 模型 | T→I R@1 | T→I R@5 | I→T R@1 |
|------|---------|---------|---------|
| Chinese-CLIP (Zero-Shot) | xx% | xx% | xx% |
| + LoRA 微调 (ours) | xx% | xx% | xx% |
| + Prompt 工程 (ours) | xx% | xx% | xx% |

## 🚀 快速开始
...
```

---

## 七、关键文件清单

```
Chinese-CLIP/                          # Fork 后的项目根目录
├── cn_clip/
│   └── clip/
│       └── lora.py                    # [新增] LoRA 模块
├── scripts/
│   ├── process_ancient_text.py        # [新增] LLM 古文处理
│   └── build_dataset.py              # [新增] 数据集构建
├── train_lora.py                      # [新增] LoRA 微调训练
├── evaluate.py                        # [新增] 检索评估
├── demo.py                            # [新增] Gradio Demo
├── data/
│   ├── raw_texts.json                 # 古文原始段落
│   └── annotations.json              # LLM 处理后的标注
└── README.md                          # [修改] 项目说明
```

---

## 八、时间线

```
第 1-3 天    ██ 搭环境、Fork Chinese-CLIP、跑通推理
第 4-8 天    ████ LLM 处理古文、生成结构化标注
第 9-13 天   ████ 图像采集、图文配对、LMDB 数据集
第 14-19 天  ████ LoRA 模块编写、训练脚本、本地调试
第 20-28 天  ██████ 消融实验 + Gradio Demo + README
```

## 九、简历推荐表述

> 基于 Chinese-CLIP 构建面向南宋文化的多模态检索系统。利用通义千问大模型实现古汉语地理实体自动抽取与结构化标注，构建包含古文原文、白话翻译及 Prompt 模板的图文对数据集；在 ViT-B-16 视觉编码器与 RoBERTa 文本编码器中注入 LoRA 适配层，仅训练约 1% 参数实现领域适配，Text→Image Recall@1 较 Zero-Shot 基线提升 XX%。

**关键词**：Chinese-CLIP · ViT · RoBERTa · LoRA (PEFT) · 多模态对齐 · 对比学习 · NLP 实体抽取 · Prompt 工程
