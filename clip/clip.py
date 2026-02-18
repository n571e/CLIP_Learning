import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),# 1. 缩放：将图像的短边缩放到 n_px 分辨率，使用 BICUBIC（双三次插值）算法，保留更多细节
        CenterCrop(n_px), # 2. 中心裁剪：从缩放后的图像中心裁取一个 n_px * n_px 的正方形区域
        _convert_image_to_rgb,# 3. 格式转换：确保图像被转换为 RGB 模式（滤开透明通道 A 或灰度模式的干扰）
        ToTensor(), # 4. 转为张量：将 PIL 图片（0-255 像素值）转换为 PyTorch Tensor，并自动缩放到 0.0 - 1.0 之间
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),# 5. 归一化：使用 CLIP 特定的均值(mean)和标准差(std)对三个色彩通道进行标准化处理
        # 这组数值是基于 OpenAI 原始训练数据计算得出的，目的是让输入分布更利于模型理解
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """加载 CLIP 模型

    参数
    ----------
    name : str
        模型名称（见 `clip.available_models()`）或者指向包含 state_dict 的本地文件路径

    device : Union[str, torch.device]
        模型加载后的运行设备（如 'cuda' 或 'cpu'）

    jit : bool
        是否加载优化过的 JIT 模型（默认 False，加载更易修改的非 JIT 模型）

    download_root: str
        模型文件的下载保存路径；默认为 "~/.cache/clip"

    返回
    -------
    model : torch.nn.Module
        CLIP 模型实例

    preprocess : Callable[[PIL.Image], torch.Tensor]
        一个 torchvision transform 处理流水线，用于将 PIL 图像转换为模型能接受的张量输入
    """
    # 如果名称在预定义的模型字典中，则从网络下载或从缓存获取模型路径
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    # 如果名称是一个存在的本地文件路径，则直接使用
    elif os.path.isfile(name):
        model_path = name
    # 否则抛出运行错误，告知可用模型列表
    else:
        raise RuntimeError(f"未找到该模型 {name}; 可用模型列表 = {available_models()}")

    # 以二进制读取模式打开模型文件
    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试作为 JIT 模型归档加载
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 如果加载 JIT 失败，则预期它是普通的权重字典（state dict）
            if jit:
                warnings.warn(f"文件 {model_path} 不是 JIT 归档文件。将作为权重字典（state dict）加载。")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    # 如果运行在非 JIT 模式（或加载失败回退到了非 JIT 模式）
    if not jit:
        # 使用权重字典或 JIT 模型中的权重重新构建模型架构，并移动到目标设备
        model = build_model(state_dict or model.state_dict()).to(device)
        # 如果是 CPU 设备，确保模型权重转换为 float32（防止半精度推理在某些 CPU 上不支持）
        if str(device) == "cpu":
            model.float()
        # 返回模型和对应的图像预处理变换（n_px 取自模型的分辨率配置）
        return model, _transform(model.visual.input_resolution)

    # --- 以下代码仅针对 JIT 模型进行设备补丁处理（Patching） ---
    # 创建一个用于追踪设备信息的辅助节点
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    # 在计算图中找到表示设备的 Constant 节点
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """获取节点的属性，处理返回类型多态的情况"""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        """递归修正 JIT 图中的设备常量，确保它们指向正确的设备"""
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                # 如果常量属性中包含 "cuda" 字符串，则将其属性替换为当前目标设备的属性
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    # 对模型及其主要预测方法应用设备补丁
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 如果目标设备是 CPU，还需要将 JIT 图中的半精度（float16）补丁为 float32
    if str(device) == "cpu":
        # 创建一个用于追踪 float32 类型信息的节点
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            """递归修正 JIT 图中的数据类型（dtype），将 float16 类型映射强制修改为 float32"""
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype 可能是 aten::to() 的第二个或第三个参数
                        # 值 5 代表 torch.float16 的枚举
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        # 应用浮点数转换补丁
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # 最终确保整个模型参数全部转为 float32
        model.float()

    # 返回 JIT 模型和对应的图像预处理方法
    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    对输入的字符串进行分词处理，返回 tokens 序列

    参数
    ----------
    texts : Union[str, List[str]]
        输入的字符串或字符串列表

    context_length : int
        上下文长度；CLIP 模型默认统一使用 77

    truncate: bool
        如果编码后的长度超过 context_length，是否进行截断处理

    返回
    -------
    一个二维张量，包含生成的 tokens，形状为 [字符串数量, context_length]
    如果 torch 版本 < 1.8.0，返回 LongTensor，否则返回 IntTensor
    """
    # 如果输入是单个字符串，将其包装进列表中统一处理
    if isinstance(texts, str):
        texts = [texts]

    # 获取分词器的起始符号 (SOT) 和 结束符号 (EOT) 的 ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 遍历每行文本，添加起始/结束标记，并进行编码
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 根据 PyTorch 版本初始化结果张量（旧版本对于张量索引要求必须是 long）
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 将编码后的 token 列表填充到结果张量的对应位置
    for i, tokens in enumerate(all_tokens):
        # 检查是否超过了上下文定义的最大长度
        if len(tokens) > context_length:
            if truncate:
                # 截断到最大长度并在最后一个槽位强制放置 EOT 标记
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                # 否则抛出异常，提示文本过长
                raise RuntimeError(f"输入文本 \"{texts[i]}\" 太长，超过了上下文长度 {context_length}")
        
        # 将 token 数值填入结果张量的对应行
        result[i, :len(tokens)] = torch.tensor(tokens)

    # 返回所有处理完毕的 tokens 张量
    return result
