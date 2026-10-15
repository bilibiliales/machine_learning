import numpy as np
import matplotlib.pyplot as plt

# 从图片中提取的数据
epochs = np.arange(1, 10)
train_accuracy = np.array([0.8273, 0.9222, 0.9348, 0.9425, 0.9492, 0.9547, 0.9596, 0.9642, 0.9684])
train_loss = np.array([0.6130, 0.2474, 0.2055, 0.1801, 0.1598, 0.1429, 0.1276, 0.1136, 0.1006])
val_accuracy = np.array([0.9091, 0.9228, 0.9294, 0.9323, 0.9345, 0.9347, 0.9337, 0.9336, 0.9322])
val_loss = np.array([0.2912, 0.2441, 0.2233, 0.2148, 0.2102, 0.2121, 0.2181, 0.2266, 0.2358])

# 绘制训练曲线
plt.figure(figsize=(12, 5))

# 子图1：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'o-', label="Train Loss")
plt.plot(epochs, val_loss, 's-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.grid(True)

# 子图2：Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'o-', label="Train Accuracy")
plt.plot(epochs, val_accuracy, 's-', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.grid(True)

plt.tight_layout()
plt.show()