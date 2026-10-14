from PIL import Image
import numpy as np
import os

# =========================
# 1. 数据集路径
# =========================
DATASET_PATH = "dataset/att_faces"

# 40个人
NUM_PERSONS = 41

# 每个人10张图片
NUM_IMAGES = 10

# 每个人9张训练，最后1张测试
TRAIN_PER_PERSON = 9


# =========================
# 2. 读取图片
# =========================
train_data = []
train_labels = []

test_data = []
test_labels = []

for person_id in range(1, NUM_PERSONS + 1):

    person_dir = os.path.join(DATASET_PATH, f"s{person_id}")

    for image_id in range(1, NUM_IMAGES + 1):

        image_path = os.path.join(
            person_dir,
            f"{image_id}.pgm"
        )

        # 读取PGM
        img = Image.open(image_path)

        # 转成numpy数组
        img = np.array(img)

        # 拉平：
        # (112, 92) -> (10304,)
        img = img.reshape(-1)

        if image_id <= TRAIN_PER_PERSON:
            train_data.append(img)
            train_labels.append(person_id)
        else:
            test_data.append(img)
            test_labels.append(person_id)


# =========================
# 3. 转换成numpy矩阵
# =========================
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)


print("训练数据 shape:", train_data.shape)
print("训练标签 shape:", train_labels.shape)

print("测试数据 shape:", test_data.shape)
print("测试标签 shape:", test_labels.shape)

# =========================
# 4. 转置成 PCA 使用的形式
# =========================
X = train_data.T

T = test_data.T

print()
print("X shape:", X.shape)
print("T shape:", T.shape)


# =========================
# 5. 计算平均脸
# =========================
mean_face = np.mean(X, axis=1)

print()
print("mean_face shape:", mean_face.shape)


# =========================
# 6. 中心化
# =========================
A = X - mean_face[:, np.newaxis]

print("A shape:", A.shape)


# =========================
# 7. 计算小型协方差矩阵
# =========================
small_cov = A.T @ A

print("A^T A shape:", small_cov.shape)

# =========================
# 8. 求 A^T A 的特征值和特征向量
# =========================

eigenvalues, eigenvectors = np.linalg.eigh(small_cov)

print()
print("eigenvalues shape:", eigenvalues.shape)
print("eigenvectors shape:", eigenvectors.shape)

# =========================
# 9. 按特征值从大到小排序
# =========================

index = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[index]
eigenvectors = eigenvectors[:, index]

print()
print("前10个特征值：")
print(eigenvalues[:10])

print()
print("最后10个特征值：")
print(eigenvalues[-10:])

# =========================
# 10. 计算 Eigenfaces
# =========================

eigenfaces = A @ eigenvectors

print()
print("eigenfaces shape:", eigenfaces.shape)

# =========================
# 11. 归一化 Eigenfaces
# =========================

norms = np.linalg.norm(eigenfaces, axis=0)

eigenfaces = eigenfaces / norms

print("归一化后 Eigenfaces shape:", eigenfaces.shape)

print("第一张 Eigenface 的长度:",
      np.linalg.norm(eigenfaces[:, 0]))

import matplotlib.pyplot as plt

# =========================
# 12. PCA 人脸重建实验
# =========================

original = X[:, 0]

centered = original - mean_face

ks = [5, 10, 20, 50, 100, 200]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# 原图
axes[0, 0].imshow(
    original.reshape(112, 92),
    cmap="gray"
)
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")


for ax, k in zip(axes.flat[1:], ks):

    E = eigenfaces[:, :k]

    # PCA编码
    z = E.T @ centered

    # PCA解码 / 重建
    reconstructed = mean_face + E @ z

    face = reconstructed.reshape(112, 92)

    ax.imshow(face, cmap="gray")
    ax.set_title(f"k = {k}")
    ax.axis("off")

plt.tight_layout()

# =========================
# 13. 计算累计解释方差比例
# =========================

# 所有特征值之和
total_variance = np.sum(eigenvalues)

# 累计特征值
cumulative_variance = np.cumsum(eigenvalues)

# 累计解释方差比例
explained_variance_ratio = (
    cumulative_variance / total_variance
)

print()
print("累计解释方差：")

for k in [1, 5, 10, 20, 30, 50, 100, 150, 200, 300, 359]:
    print(
        f"k={k:3d}: "
        f"{explained_variance_ratio[k - 1] * 100:.2f}%"
    )

# =========================
# 14. 训练集 PCA 投影
# =========================

P = E.T @ A

print()
print("E shape:", E.shape)
print("P shape:", P.shape)

T_centered = T - mean_face[:, np.newaxis]

TestT = E.T @ T_centered

print("TestT shape:", TestT.shape)

# =========================
# 15. 人脸识别
# =========================

correct = 0

for test_index in range(41):

    test_vector = TestT[:, test_index]

    distances = np.linalg.norm(
        P - test_vector[:, np.newaxis],
        axis=0
    )

    nearest_index = np.argmin(distances)

    predicted_person = train_labels[nearest_index]
    actual_person = test_labels[test_index]

    print(
        f"测试图片 {test_index + 1:2d}: "
        f"真实 = s{actual_person:2d}, "
        f"识别 = s{predicted_person:2d}, "
        f"距离 = {distances[nearest_index]:.2f}"
    )

    if predicted_person == actual_person:
        correct += 1


accuracy = correct / 41

print()
print(f"正确数量: {correct}/41")
print(f"识别率: {accuracy * 100:.2f}%")

# =========================
# 16. 测试不同 K 对识别率的影响
# =========================

K_list = [10, 20, 30, 50, 100, 150, 200, 300]

for K in K_list:

    # 选择前 K 个 Eigenface
    E = eigenfaces[:, :K]

    # 训练集投影
    P = E.T @ A

    # 测试集投影
    TestT = E.T @ T_centered

    correct = 0

    for test_index in range(41):

        test_vector = TestT[:, test_index]

        distances = np.linalg.norm(
            P - test_vector[:, np.newaxis],
            axis=0
        )

        nearest_index = np.argmin(distances)

        predicted_person = train_labels[nearest_index]
        actual_person = test_labels[test_index]

        if predicted_person == actual_person:
            correct += 1

    accuracy = correct / 41

    print(
        f"K={K:3d}, "
        f"解释方差={explained_variance_ratio[K - 1] * 100:.2f}%, "
        f"正确={correct}/41, "
        f"识别率={accuracy * 100:.2f}%"
    )

def show_projection_comparison(image, mean_face, eigenfaces, k=50):
    """
    显示 PCA 投影前后的图片。

    image: 原始人脸图像的一维向量，形状为 (10304,)
    mean_face: 平均脸，形状为 (10304,)
    eigenfaces: 特征脸矩阵，形状为 (10304, n_components)
    k: 使用前 k 个特征脸进行投影和重建
    """
    # 1. 中心化
    centered = image - mean_face

    # 2. 选择前 k 个特征脸
    E = eigenfaces[:, :k]

    # 3. 投影到低维 PCA 空间
    z = E.T @ centered

    # 4. 从低维空间重建图像
    reconstructed = mean_face + E @ z

    # 5. 显示投影前后的图片
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image.reshape(112, 92), cmap="gray")
    axes[0].set_title("Before PCA")
    axes[0].axis("off")

    axes[1].imshow(reconstructed.reshape(112, 92), cmap="gray")
    axes[1].set_title(f"After PCA (k={k})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return z, reconstructed

# 显示训练集中第一张人脸投影前后的效果
z, reconstructed = show_projection_comparison(
    image=X[:, 0],
    mean_face=mean_face,
    eigenfaces=eigenfaces,
    k=50
)