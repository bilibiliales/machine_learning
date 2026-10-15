from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("dataset/att_faces/s41/1.pgm")

print("图片大小:", img.size)
print("图片模式:", img.mode)

data = np.array(img)

print("矩阵形状:", data.shape)
print("最小灰度:", data.min())
print("最大灰度:", data.max())
print("数据类型:", data.dtype)

plt.imshow(data, cmap="gray")
plt.axis("off")
plt.show()