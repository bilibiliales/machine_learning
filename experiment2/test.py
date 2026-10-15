from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# 1. 配置
# =========================
DATASET_PATH = Path("dataset/att_faces")
MODEL_PATH = Path("pca_model.npz")

TEST_PERSON = 41
TEST_IMAGE_ID = 10

IMAGE_WIDTH = 92
IMAGE_HEIGHT = 112

# =========================
# 2. 加载 PCA 模型
# =========================
model = np.load(MODEL_PATH)

mean_face = model["mean_face"]
E = model["E"]
W = model["W"]
P = model["P"]
train_labels = model["train_labels"]
K = int(model["K"])

print("投影矩阵 W shape:", W.shape)
print("训练集投影矩阵 P shape:", P.shape)


# =========================
# 3. 加载测试照片
# =========================
test_path = (
    DATASET_PATH
    / f"s{TEST_PERSON}"
    / f"{TEST_IMAGE_ID}.pgm"
)

if not test_path.exists():
    raise FileNotFoundError(f"找不到测试图片：{test_path}")

with Image.open(test_path) as img:
    img = img.convert("L")
    original = np.asarray(img, dtype=np.float64).reshape(-1)

T = original.reshape(-1, 1)

print("测试图片:", test_path)
print("T shape:", T.shape)

# =========================
# 4. 测试图片 PCA 投影
# =========================
T_centered = T - mean_face[:, np.newaxis]

# TestT = W @ (T - mean_face)
TestT = W @ T_centered

print("TestT shape:", TestT.shape)

# =========================
# 5. 最近邻人脸识别
# =========================
test_vector = TestT[:, 0]

distances = np.linalg.norm(
    P - test_vector[:, np.newaxis],
    axis=0
)

nearest_index = np.argmin(distances)

predicted_person = int(train_labels[nearest_index])
actual_person = TEST_PERSON

print("\n========== 识别结果 ==========")
print(f"测试图片: s{actual_person}/{TEST_IMAGE_ID}.pgm")
print(f"真实身份: s{actual_person}")
print(f"预测身份: s{predicted_person}")
print(f"最近邻训练样本: s{predicted_person}/"
      f"{nearest_index % 9 + 1}.pgm")
print(f"最近邻距离: {distances[nearest_index]:.4f}")
print(f"识别结果: {'正确' if predicted_person == actual_person else '错误'}")

# 同时打印距离最近的 5 张训练图片
top_indices = np.argsort(distances)[:5]

print("\n距离最近的 5 张训练图片:")
for rank, idx in enumerate(top_indices, start=1):
    print(
        f"{rank}. s{int(train_labels[idx])}/"
        f"{idx % 9 + 1}.pgm, "
        f"距离={distances[idx]:.4f}"
    )

# =========================
# 6. PCA 重建测试图片
# =========================
z = TestT[:, 0]

reconstructed = mean_face + E @ z

original_image = original.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
reconstructed_image = reconstructed.reshape(
    IMAGE_HEIGHT, IMAGE_WIDTH
)

# 仅用于显示，限制在灰度像素范围
reconstructed_image = np.clip(reconstructed_image, 0, 255)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Before PCA: s41/10")
axes[0].axis("off")

axes[1].imshow(reconstructed_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(f"After PCA: K={K}")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("my_face_pca_comparison.png", dpi=150)
plt.show()
