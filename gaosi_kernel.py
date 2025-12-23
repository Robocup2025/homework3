import numpy as np
import matplotlib.pyplot as plt
import cv2

def gaosi_kernel(size: int, sigma: float):
    assert size % 2 == 1, "kernel size must be odd"
    k = size // 2
    x,y = np.mgrid[-k:k+1, -k:k+1]
    gaussion = np.exp(-(x**2 + y**2) / (2*sigma**2))
    gaussion /= 2 * np.pi * sigma**2
    gaussion /= gaussion.sum()
    return gaussion

# 示例：生成三个不同参数的高斯核
kernels = [
    gaosi_kernel(3, 1),
    gaosi_kernel(5, 1),
    gaosi_kernel(7, 2)
]

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i, k in enumerate(kernels):
    axes[i].imshow(k, cmap='viridis')
    axes[i].set_title(f"size={k.shape[0]}, σ={ [1,1,2][i] }")
    axes[i].axis('off')
plt.show()

size = 5
sigma = 1

# OpenCV 生成二维高斯核
g1d = cv2.getGaussianKernel(size, sigma)
opencv_kernel = g1d @ g1d.T  

my_kernel = gaosi_kernel(size, sigma)

# 比较差异
print("差值（绝对值最大）:", np.abs(opencv_kernel - my_kernel).max())
