"""
test.py - 加载已训练的 RNN 词性标注模型进行预测
用法: python test.py
"""

import os
import re
import pickle
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ===================== 配置区=====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "199801")
MODEL_DIR = os.path.dirname(__file__)
MAX_LEN = 60
MAX_VOCAB = 50000
VOCAB_CACHE = os.path.join(MODEL_DIR, "vocab.pkl")
# ===========================================================================
# -------------------------- 词典缓存读写 --------------------------
def save_vocab(word2idx, idx2tag):
    with open(VOCAB_CACHE, "wb") as f:
        pickle.dump((word2idx, idx2tag), f)


def load_vocab():
    with open(VOCAB_CACHE, "rb") as f:
        return pickle.load(f)

# -------------------------- 语料解析（与 embedding.py 一致）--------------------------
def read_text(path):
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件: {path}")

def parse_line(line):
    pairs = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "[":
            j = line.find("]", i)
            if j == -1:
                i += 1
                continue
            inner = line[i + 1 : j]
            after = line[j + 1 :]
            outer_tag_m = re.match(r"[A-Za-z]+", after)
            outer_tag = outer_tag_m.group(0) if outer_tag_m else "n"
            if outer_tag_m:
                i = j + 1 + len(outer_tag)
            else:
                i = j + 1
            words = []
            for tok in re.split(r"\s+", inner.strip()):
                if not tok:
                    continue
                slash = tok.rfind("/")
                if slash == -1:
                    words.append(tok)
                else:
                    words.append(tok[:slash])
            combined = "".join(words)
            if combined:
                pairs.append((combined, outer_tag))
        else:
            j = line.find("[", i)
            if j == -1:
                j = n
            seg = line[i:j].strip()
            i = j
            for tok in re.split(r"\s+", seg):
                if not tok:
                    continue
                slash = tok.rfind("/")
                if slash == -1:
                    pairs.append((tok, "n"))
                else:
                    word, tag = tok[:slash], tok[slash + 1 :]
                    if word and tag:
                        pairs.append((word, tag))
    return pairs


def load_corpus(data_dir, max_lines_per_file=None):
    sentences, tags = [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    print(f"找到 {len(files)} 个语料文件: {[os.path.basename(f) for f in files]}")
    for fpath in files:
        text = read_text(fpath)
        count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            pairs = parse_line(line)
            if pairs:
                sentences.append([p[0] for p in pairs])
                tags.append([p[1] for p in pairs])
                count += 1
                if max_lines_per_file and count >= max_lines_per_file:
                    break
    print(f"共加载 {len(sentences)} 条句子")
    return sentences, tags


# -------------------------- 构建词典（与 embedding.py 一致）--------------------------
def build_vocab(sentences, tags, max_vocab=50000):
    word_freq = {}
    for s in sentences:
        for w in s:
            word_freq[w] = word_freq.get(w, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in sorted_words[: max_vocab - 2]]
    word2idx = {w: i for i, w in enumerate(vocab)}
    tag_set = sorted(set(t for ts in tags for t in ts))
    tag_list = ["<PAD>"] + tag_set
    idx2tag = {i: t for i, t in enumerate(tag_list)}
    print(f"词典大小: {len(vocab)} (含<PAD>和<UNK>)")
    print(f"标签数量: {len(tag_list)} (含<PAD>)")
    return word2idx, idx2tag


# -------------------------- 文本编码 --------------------------
def encode_sentence(words, word2idx, max_len=MAX_LEN):
    ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words[:max_len]]
    pad_len = max_len - len(ids)
    return ids + [word2idx["<PAD>"]] * pad_len


# -------------------------- 模型加载 --------------------------
def find_model():
    for name in ("rnn_pos_model.keras", "rnn_pos_model.h5"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(MODEL_DIR, "*_model.*"))
    if hits:
        return hits[0]
    raise FileNotFoundError("未找到模型文件，请确认 rnn_pos_model.keras/.h5 在同目录")


# -------------------------- 预测 --------------------------
def predict(model, words, word2idx, idx2tag, max_len=MAX_LEN):
    unk_count = sum(1 for w in words if w not in word2idx)
    ids = encode_sentence(words, word2idx, max_len)
    x = np.array([ids])
    probs = model(x, training=False)[0]
    preds = []
    for k, w in enumerate(words[:max_len]):
        tag = idx2tag[int(np.argmax(probs[k]))]
        preds.append((w, tag))
    return preds, unk_count


# -------------------------- 主流程 --------------------------
def main():
    # 1. 加载/重建词典
    if os.path.exists(VOCAB_CACHE):
        print("加载缓存词典 vocab.pkl ...")
        word2idx, idx2tag = load_vocab()
    else:
        print("未找到 vocab.pkl，从语料重建词典（首次运行较慢）...")
        sentences, tags = load_corpus(DATA_DIR)
        word2idx, idx2tag = build_vocab(sentences, tags, MAX_VOCAB)
        save_vocab(word2idx, idx2tag)
        print(f"词典已缓存到 {VOCAB_CACHE}")

    # 2. 加载模型
    model_path = find_model()
    print(f"加载模型: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    # 3. 预设测试
    test_cases = [
        "迈向 充满 希望 的 新 世纪",
        "中国 经济 持续 快速 发展",
        "我们 坚决 拥护 党 的 领导",
    ]
    for raw in test_cases:
        words = raw.split()
        preds, unk = predict(model, words, word2idx, idx2tag)
        print("\n" + " ".join(f"{w}/{t}" for w, t in preds))
        if unk:
            print(f"  提示: 句中有 {unk} 个 <UNK> 词，可能影响标注效果")

    # 4. 自由验证
    line="在 新 时代 征程 续写 写经 经济 快速 发展 和 社会 会长 长期 稳定 两 大 奇迹 新篇 新篇章 篇章 中共 中共中央 中央 举行 新闻 发布 发布会 解读 党 的 二十 十届 四中 四中全会 全会 全会精神 精神"
    words = line.split()
    preds, unk = predict(model, words, word2idx, idx2tag)
    print(" ".join(f"{w}/{t}" for w, t in preds))
    if unk:
        print(f"  提示: 句中有 {unk} 个 <UNK> 词，可能影响标注效果")

main()