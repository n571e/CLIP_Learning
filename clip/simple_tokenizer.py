import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

 
@lru_cache()
def default_bpe():
    # 返回默认的 BPE 词表文件路径
    # 该文件通常位于当前脚本同级目录下，文件名为 "bpe_simple_vocab_16e6.txt.gz"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    返回一个字典，将 utf-8 字节映射到对应的 unicode 字符串。
    
    原因：
    可逆的 BPE 编码通常是在 unicode 字符串上工作的。
    如果直接使用 unicode 字符，通过 BPE 避免 unknown (UNK) token 需要非常大的词表。
    对于像 10B token 规模的数据集，为了覆盖率可能需要 5000 多个特殊字符，
    这占据了标准 32K BPE 词表的很大一部分。
    
    为了避免这种情况，我们建立了 utf-8 字节（0-255）到 unicode 字符串的映射表。
    这个映射表还需要避免映射到 BPE 代码无法处理的空白字符或控制字符。
    """
    # 生成基础的可打印字符范围列表：
    # range(ord("!"), ord("~")+1) -> ASCII 33-126 (常见标点、数字、字母)
    # range(ord("¡"), ord("¬")+1) -> Latin-1 增补字符一部分
    # range(ord("®"), ord("ÿ")+1) -> Latin-1 增补字符另一部分
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    
    # 复制一份作为对应的 unicode 字符列表
    cs = bs[:]
    
    # 初始化计数器，用于生成那些不在上面的 bs 列表里的字节的映射字符
    n = 0
    
    # 遍历所有可能的 256 个字节 (0-255)
    for b in range(2**8):
        # 如果这个字节不在上面的常用可打印字符列表中
        if b not in bs:
            # 将其加入 bs 列表
            bs.append(b)
            # 将其映射为一个从 256 开始递增的字符 ID (避免与标准 ASCII 冲突)
            cs.append(2**8+n)
            n += 1
    
    # 将整数列表转换为对应的字符列表
    cs = [chr(n) for n in cs]
    
    # 返回一个字典，key 是字节值，value 是对应的 unicode 字符
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回一个单词中所有相邻字符组成的符号对 (symbol pairs)。
    
    参数 word 是一个元组，包含了一系列符号（符号可以是变长的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    # 遍历单词中的每一个字符（从第二个开始）
    for char in word[1:]:
        # 将 (前一个字符, 当前字符) 作为一个对加入集合
        pairs.add((prev_char, char))
        prev_char = char
    # 返回所有找到的字符对集合
    return pairs


def basic_clean(text):
    # 使用 ftfy 修复文本中的编码问题（如乱码）
    text = ftfy.fix_text(text)
    # 两次 html 下一转换义，确保将 &amp; 等 HTML 实体转换为对应的字符
    text = html.unescape(html.unescape(text))
    # 去除首尾空白
    return text.strip()


def whitespace_clean(text):
    # 将所有的空白字符（包括换行、制表符等）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        # 初始化字节到 unicode 的编码器
        self.byte_encoder = bytes_to_unicode()
        # 初始化 unicode 到字节的解码器（反转上面的字典）
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 读取并解压 gzip 格式的 BPE 词表文件（merges 文件）
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        
        # 截取 merges 列表的一部分
        # 1:49152-256-2+1 是一个切片操作，具体的范围依赖于词表的设计
        # 49152 是总的 vocab size，减去 256 (基础字节) 和 2 (特殊 token) 等得到 merge 规则的数量
        merges = merges[1:49152-256-2+1]
        
        # 将每一行 merge 规则（例如 "e r"）分割并转换为元组 (('e', 'r'))
        merges = [tuple(merge.split()) for merge in merges]
        
        # 初始化词表，首先包含所有基础的字节对应的 unicode 字符
        vocab = list(bytes_to_unicode().values())
        
        # 将基础词表复制一份，并给每个词加上 '</w>' 后缀，表示单词结尾
        # 这通常用于区分作为单词一部分的字符和作为单词结尾的字符
        vocab = vocab + [v+'</w>' for v in vocab]
        
        # 将所有的 merge 规则合并后的词加入词表
        for merge in merges:
            vocab.append(''.join(merge))
            
        # 添加特殊的起始和结束 token
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        
        # 创建词表到 ID 的映射字典 (Encoder)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # 创建 ID 到词表的映射字典 (Decoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 创建 BPE merge 规则到其优先级的映射字典
        # 列表索引越小，优先级越高
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # 初始化缓存，预存特殊 token
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        
        # 编译正则表达式，用于分词
        # 该正则包含了：
        # - 特殊 token (<|startoftext|>, <|endoftext|>)
        # - 常见的缩写 ( 's, 't, 're 等)
        # - 字母序列 ( [\p{L}]+ )
        # - 数字 ( [\p{N}] )
        # - 非空白的特殊符号 ( [^\s\p{L}\p{N}]+ )
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        # 如果该 token 已经在缓存中，直接返回
        if token in self.cache:
            return self.cache[token]
            
        # 将 token 转换为字符元组，除了最后一个字符外
        # 最后一个字符加上 '</w>' 后缀，标记单词边界
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        
        # 获取当前 word 中所有的相邻字符对
        pairs = get_pairs(word)

        # 如果没有字符对（即单个字符），直接返回
        if not pairs:
            return token+'</w>'

        while True:
            # 找到当前 pairs 中优先级最高（rank 值最小 / 在 merges 中出现最早）的 pair
            # 如果 pair 不在 bpe_ranks 中，给无穷大的 rank
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # 如果找到的 pair 不在我们的合并规则中，说明无法继续合并，退出循环
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            
            # 遍历当前的 word，尝试执行合并操作
            while i < len(word):
                try:
                    # 在 word 中查找 first 出现的位置
                    j = word.index(first, i)
                    # 将 i 到 j 之间的部分（即未匹配部分）直接加入 new_word
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 如果找不到 first 了，将剩余部分加入 new_word 并退出内层循环
                    new_word.extend(word[i:])
                    break

                # 检查是否匹配到了 (first, second) 这个 bigram
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    # 如果匹配成功，合并它们并加入 new_word
                    new_word.append(first+second)
                    # 跳过这两个已合并的元素
                    i += 2
                else:
                    # 如果没匹配上（比如只有 first 但后面不是 second），保留原样
                    new_word.append(word[i])
                    i += 1
            
            # 更新 word 为合并后新的 tuple
            new_word = tuple(new_word)
            word = new_word
            
            # 如果 word 已经合并成一个符号了，就不需要继续了
            if len(word) == 1:
                break
            else:
                # 重新计算新 word 的 pairs，准备下一轮合并
                pairs = get_pairs(word)
        
        # 将最终的 word元组 用空格连接成字符串
        word = ' '.join(word)
        # 存入缓存
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        # 对文本进行基础清洗和空格清洗，并转为小写
        text = whitespace_clean(basic_clean(text)).lower()
        
        # 使用正则表达式将文本分割成一个个 token (如单词、标点等)
        for token in re.findall(self.pat, text):
            # 第一步：将 utf-8 字符串转为 bytes，再映射为特定的 unicode 字符
            # 目的是为了获得可逆且安全的 BPE 输入
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            # 第二步：对每个 mapped token 执行 BPE 算法
            # self.bpe(token) 返回的是用空格分隔的子词字符串，如 "w or d </w>"
            # split(' ') 后变成列表，再通过 self.encoder 查表转为 ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            
        # 返回最终的 token ID 列表
        return bpe_tokens

    def decode(self, tokens):
        # 将 token ID 序列转回字符串序列，并拼接
        text = ''.join([self.decoder[token] for token in tokens])
        
        # 将字符串序列中的每个字符通过 byte_decoder 转回原始的 byte 值
        # bytearray 接收这个 byte 列表
        # decode('utf-8') 将 bytearray 解码回 utf-8 字符串
        # 最后把 '</w>' 替换为空格
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        return text
