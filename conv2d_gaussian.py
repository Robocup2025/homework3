import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size: int, sigma: float):
    assert size % 2 == 1, "kernel size must be odd"
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= 2 * np.pi * sigma**2
    kernel /= kernel.sum()  
    return kernel

def conv2d(image, kernel):
    # 获取尺寸
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 边缘镜像填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)

    # 逐像素卷积
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

if __name__ == "__main__":
    img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("请将 test.jpg 放在当前目录下")

    # 定义不同参数的高斯核
    params = [(3, 1), (5, 1), (7, 2)]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("原图像")
    plt.axis('off')

    # 依次卷积并显示结果
    for idx, (size, sigma) in enumerate(params):
        kernel = gaussian_kernel(size, sigma)
        blurred = conv2d(img, kernel)
        plt.subplot(1, 4, idx + 2)
        plt.imshow(blurred, cmap='gray')
        plt.title(f"size={size}, σ={sigma}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
