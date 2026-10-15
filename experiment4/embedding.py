import os
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# 1. 语料预处理
# ============================================================

def parse_line(line):
    line = line.strip()
    if not line:
        return []

    tokens = line.split()
    pairs = []

    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]

        # 跳过段首编号（如 19980101-01-001-001/m）
        if re.match(r"^\d{8}-\d{2}-\d{3}-\d{3}/", token):
            i += 1
            continue

        # 处理合并标注 [xxx/tag yyy/tag ...]tag
        if token.startswith("["):
            merged_words = []
            merged_tag = None
            while i < n:
                t = tokens[i]
                i += 1  # 关键：内层循环一定前进

                if "]" in t:
                    # 形如 电台/n]nt —— ]后面才是整体标签
                    inner, _, outer = t.partition("]")
                    inner = inner.replace("[", "")
                    if outer:  # ] 后面有标签，作为整体标签
                        merged_tag = outer
                    if "/" in inner:
                        w = inner.rsplit("/", 1)[0]
                        if w:
                            merged_words.append(w)
                    break

                t_clean = t.replace("[", "").replace("]", "")
                if "/" in t_clean:
                    w = t_clean.rsplit("/", 1)[0]
                    if w:
                        merged_words.append(w)

            if merged_words:
                merged_word = "".join(merged_words)
                pairs.append((merged_word, merged_tag if merged_tag else "nt"))
            continue

        # 普通词/词性
        if "/" in token:
            parts = token.rsplit("/", 1)
            if len(parts) == 2:
                word, tag = parts
                word = word.replace("[", "").replace("]", "")
                if word and tag:
                    pairs.append((word, tag))

        i += 1  # 关键修复：普通分支必须前进，否则死循环 -> MemoryError

    return pairs


def load_corpus(data_dir, max_lines_per_file=None):
    sentences = []
    tags = []

    txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    print(f"找到 {len(txt_files)} 个语料文件: {txt_files}")

    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)
        count = 0
        with open(filepath, "r", encoding="gbk", errors="ignore") as f:
            for line in f:
                pairs = parse_line(line)
                if not pairs:
                    continue

                words = [p[0] for p in pairs]
                tgs = [p[1] for p in pairs]

                if len(words) != len(tgs) or len(words) == 0:
                    continue

                sentences.append(words)
                tags.append(tgs)

                count += 1
                if max_lines_per_file and count >= max_lines_per_file:
                    break

    return sentences, tags

# ============================================================
# 2. 构建字典（词表 + 标签表）
# ============================================================

def build_vocab(sentences, tags, max_vocab=50000):
    word_count = {}
    tag_set = set()

    for words, tgs in zip(sentences, tags):
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1
        for t in tgs:
            tag_set.add(t)

    # 按词频排序，保留高频词
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    sorted_words = sorted_words[:max_vocab - 2]  # 预留 <PAD> 和 <UNK>

    # 词字典
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in sorted_words:
        word2idx[w] = len(word2idx)

    idx2word = {v: k for k, v in word2idx.items()}

    # 标签字典
    tag2idx = {"<PAD>": 0}
    for t in sorted(tag_set):
        tag2idx[t] = len(tag2idx)

    idx2tag = {v: k for k, v in tag2idx.items()}

    print(f"词典大小: {len(word2idx)} (含<PAD>和<UNK>)")
    print(f"标签数量: {len(tag2idx)} (含<PAD>)")
    print(f"标签列表: {list(tag2idx.keys())}")

    return word2idx, idx2word, tag2idx, idx2tag


# ============================================================
# 3. 生成训练样本
# ============================================================

def sentences_to_ids(sentences, tags, word2idx, tag2idx, max_len=60):
    X, y = [], []

    for words, tgs in zip(sentences, tags):
        if len(words) == 0:
            continue

        word_ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
        tag_ids = [tag2idx[t] for t in tgs]

        X.append(word_ids)
        y.append(tag_ids)

    # 填充到统一长度
    X = pad_sequences(X, maxlen=max_len, padding="post", value=word2idx["<PAD>"])
    y = pad_sequences(y, maxlen=max_len, padding="post", value=tag2idx["<PAD>"])

    print(f"训练样本形状: X={X.shape}, y={y.shape}")

    return X, y

# ============================================================
# 4. 构建循环神经网络模型
# ============================================================

def build_model(vocab_size, tag_size, embedding_dim=128, hidden_dim=256, max_len=60):
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True)),
        layers.Dense(tag_size, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. 训练模型
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    # 添加早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history


# ============================================================
# 6. 测试：对已分词句子进行词性标注
# ============================================================

def predict_pos(model, words, word2idx, tag2idx, idx2tag, max_len=60):
    # 将词转换为索引
    ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
    ids_padded = pad_sequences([ids], maxlen=max_len, padding="post", value=word2idx["<PAD>"])

    # 预测
    probs = model.predict(ids_padded, verbose=0)[0]

    # 取每个位置概率最大的标签
    results = []
    for i, w in enumerate(words):
        if i >= probs.shape[0]:
            break
        tag_id = int(np.argmax(probs[i]))
        tag = idx2tag.get(tag_id, "<UNK>")
        results.append((w, tag))

    return results

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    # ---------- 配置 ----------
    DATA_DIR = "./199801"       # 语料目录
    MAX_LEN = 60                # 序列最大长度
    MAX_VOCAB = 50000           # 最大词汇量
    EMBEDDING_DIM = 128         # 词嵌入维度
    HIDDEN_DIM = 256            # LSTM隐藏层维度
    EPOCHS = 1                 # 训练轮数
    BATCH_SIZE = 128            # 批大小
    TRAIN_RATIO = 0.9          # 训练集比例
    MAX_LINES_PER_FILE = None   # 调试时可设 500 快速跑通全流程，验证无误后改回 None 跑全量

    print("=" * 60)
    print("步骤1: 加载语料")
    print("=" * 60)
    sentences, tags = load_corpus(DATA_DIR, max_lines_per_file=MAX_LINES_PER_FILE)
    print(f"共加载 {len(sentences)} 条句子")

    # 打印示例
    print(f"\n示例句子: {' / '.join(sentences[0][:10])}")
    print(f"示例标签: {' / '.join(tags[0][:10])}")

    print("\n" + "=" * 60)
    print("步骤2: 构建字典")
    print("=" * 60)
    word2idx, idx2word, tag2idx, idx2tag = build_vocab(
        sentences, tags, max_vocab=MAX_VOCAB
    )

    print("\n" + "=" * 60)
    print("步骤3: 生成训练样本")
    print("=" * 60)
    X, y = sentences_to_ids(sentences, tags, word2idx, tag2idx, max_len=MAX_LEN)

    # 划分训练集和验证集
    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)
    split = int(TRAIN_RATIO * len(indices))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")

    print("\n" + "=" * 60)
    print("步骤4: 构建模型")
    print("=" * 60)
    model = build_model(
        vocab_size=len(word2idx),
        tag_size=len(tag2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        max_len=MAX_LEN
    )
    model.summary()

    print("\n" + "=" * 60)
    print("步骤5: 训练模型")
    print("=" * 60)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    print("\n" + "=" * 60)
    print("步骤6: 测试词性标注")
    print("=" * 60)

    # 测试用例1：从语料中取一个句子
    test_words_1 = sentences[100] if len(sentences) > 100 else sentences[0]
    test_tags_1 = tags[100] if len(tags) > 100 else tags[0]
    results_1 = predict_pos(model, test_words_1, word2idx, tag2idx, idx2tag, max_len=MAX_LEN)

    print("\n测试用例1（来自语料）:")
    print("预测结果:")
    for w, t in results_1:
        print(f"  {w}/{t}", end="")
    print()
    print("真实标签:")
    for w, t in zip(test_words_1, test_tags_1):
        print(f"  {w}/{t}", end="")
    print()

    # 测试用例2：自定义句子
    test_words_2 = ["中国", "经济", "持续", "快速", "发展"]
    results_2 = predict_pos(model, test_words_2, word2idx, tag2idx, idx2tag, max_len=MAX_LEN)

    print("\n测试用例2（自定义）:")
    print("预测结果:")
    for w, t in results_2:
        print(f"  {w}/{t}", end="")
    print()

    # 测试用例3：另一个自定义句子
    test_words_3 = ["我们", "坚决", "拥护", "党", "的", "领导"]
    results_3 = predict_pos(model, test_words_3, word2idx, tag2idx, idx2tag, max_len=MAX_LEN)

    print("\n测试用例3（自定义）:")
    print("预测结果:")
    for w, t in results_3:
        print(f"  {w}/{t}", end="")
    print()

    print("\n" + "=" * 60)
    print("步骤7: 验证集评估")
    print("=" * 60)
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"验证集 Loss: {loss:.4f}")
    print(f"验证集 Accuracy: {acc:.4f}")

    # 保存模型
    model.save("rnn_pos_model.h5")
    print("\n模型已保存到 rnn_pos_model.h5")