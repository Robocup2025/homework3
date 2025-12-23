import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size: int, sigma: float):
    assert size % 2 == 1
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= 2 * np.pi * sigma**2
    kernel /= kernel.sum()
    return kernel

def conv2d(image, kernel, padding_type="zero"):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 手动 padding
    if padding_type == "zero":
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding_type == "reflect":
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:
        raise ValueError("Unsupported padding type")

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

    kernel = gaussian_kernel(5, 1)

    # 两种 padding 方式滤波
    img_zero = conv2d(img, kernel, padding_type="zero")
    img_reflect = conv2d(img, kernel, padding_type="reflect")

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("原图像")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_zero, cmap='gray')
    plt.title("Zero Padding")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_reflect, cmap='gray')
    plt.title("Reflect Padding")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
