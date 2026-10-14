import numpy as np
import tensorflow.keras as ka
import tensorflow as tf
import matplotlib.pyplot as plt
import time

np.random.seed(0)
tf.random.set_seed(0)

# 1. 加载 CIFAR-10
(x_train, y_train), (x_test, y_test) = ka.datasets.cifar10.load_data()

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# 2. 数据预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CIFAR-10 标签原本形状为 (N, 1)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. 展示部分数据
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 4. 构建卷积神经网络
model = ka.Sequential([
    ka.layers.Input(shape=(32, 32, 3)),

    # 第一组卷积层
    ka.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    ),
    ka.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    ),
    ka.layers.MaxPooling2D((2, 2)),
    ka.layers.Dropout(0.2),

    # 第二组卷积层
    ka.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    ),
    ka.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    ),
    ka.layers.MaxPooling2D((2, 2)),
    ka.layers.Dropout(0.3),

    # 分类层
    ka.layers.Flatten(),
    ka.layers.Dense(128, activation="relu"),
    ka.layers.Dropout(0.3),
    ka.layers.Dense(10, activation="softmax")
])

model.summary()

# 5. 编译模型
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. 训练模型
start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

train_time = time.time() - start_time

print(f"训练耗时：{train_time:.2f} 秒")

# 7. 测试模型
test_loss, test_acc = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"测试集损失：{test_loss:.4f}")
print(f"测试集准确率：{test_acc:.4f}")
print(f"测试集准确率百分比：{test_acc * 100:.2f}%")

# 8. 绘制训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("cifar10_training.png", dpi=150)
plt.show()

# 9. 保存模型
model.save("cifar10_cnn.keras")
print("Model saved to cifar10_cnn.keras")
