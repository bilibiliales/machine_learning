from pathlib import Path

import numpy as np
from PIL import Image

# =========================
# 1. 配置
# =========================
DATASET_PATH = Path("dataset/att_faces")
MODEL_PATH = Path("pca_model.npz")

NUM_PERSONS = 41
TRAIN_PER_PERSON = 9
IMAGE_WIDTH = 92
IMAGE_HEIGHT = 112
K = 50


# =========================
# 2. 读取训练集
# =========================
train_data = []
train_labels = []

for person_id in range(1, NUM_PERSONS + 1):
    person_dir = DATASET_PATH / f"s{person_id}"

    for image_id in range(1, TRAIN_PER_PERSON + 1):
        image_path = person_dir / f"{image_id}.pgm"

        if not image_path.exists():
            raise FileNotFoundError(f"找不到图片：{image_path}")

        with Image.open(image_path) as img:
            img = img.convert("L")
            if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"{image_path} 尺寸为 {img.size}，"
                    f"预期为 {(IMAGE_WIDTH, IMAGE_HEIGHT)}"
                )
            img = np.asarray(img, dtype=np.float64).reshape(-1)

        train_data.append(img)
        train_labels.append(person_id)

train_data = np.asarray(train_data)
train_labels = np.asarray(train_labels)

# 每一列是一张人脸
X = train_data.T

print("训练数据 shape:", train_data.shape)
print("训练标签 shape:", train_labels.shape)
print("X shape:", X.shape)


# =========================
# 3. 计算平均脸并中心化
# =========================
mean_face = np.mean(X, axis=1)
A = X - mean_face[:, np.newaxis]

print("mean_face shape:", mean_face.shape)
print("A shape:", A.shape)


# =========================
# 4. 计算小型协方差矩阵
# =========================
small_cov = A.T @ A
print("A.T @ A shape:", small_cov.shape)


# =========================
# 5. 求特征值和特征向量
# =========================
eigenvalues, eigenvectors = np.linalg.eigh(small_cov)

# 按特征值从大到小排序
index = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[index]
eigenvectors = eigenvectors[:, index]

# 消除浮点误差造成的微小负特征值
eigenvalues = np.maximum(eigenvalues, 0)

print("前 10 个特征值:")
print(eigenvalues[:10])


# =========================
# 6. 计算并归一化 Eigenfaces
# =========================
eigenfaces = A @ eigenvectors

norms = np.linalg.norm(eigenfaces, axis=0)

E = eigenfaces[:, :K]

# W 是 PCA 投影矩阵，形状为 K × 10304
W = E.T

print("E shape:", E.shape)
print("W shape:", W.shape)


# =========================
# 7. 训练集投影 P = W @ A
# =========================
P = W @ A

print("P shape:", P.shape)
print("训练样本数:", P.shape[1])


# =========================
# 8. 保存模型
# =========================
np.savez(
    MODEL_PATH,
    mean_face=mean_face,
    E=E,
    W=W,
    P=P,
    train_labels=train_labels,
    eigenvalues=eigenvalues,
    K=np.array(K),
)

print(f"\n模型已保存：{MODEL_PATH.resolve()}")
print("训练完成。")
