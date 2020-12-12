import numpy as np
import math


def create_gaussian_kernel(radius, sigma):
    norm = 0
    size = 2 * radius + 1
    kernel = np.zeros(shape=(size, size), dtype='float32')
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            kernel[i + radius, j + radius] = (math.exp((-1) * (i * i + j * j) / (sigma * sigma)))
            norm += kernel[i + radius, j + radius]
    for i in range(size):
        for j in range(size):
            kernel[i, j] = kernel[i, j] / norm
    return kernel


def calculate_new_color(img, x, y, radius, kernel):
    height, width = img.shape
    g = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            idx = max(min(x + i, width - 1), 0)
            idy = max(min(y + j, height - 1), 0)
            color = img[idy, idx]
            g += color * kernel[i, j]
    return int(max(min(g, 255), 0))


def gauss_filter(img):
    height, width = img.shape
    result = np.zeros(shape=(height, width), dtype='uint8')
    radius = 1
    kernel = create_gaussian_kernel(radius, 2)
    for x in range(width):
        for y in range(height):
            result[y, x] = calculate_new_color(img, x, y, radius, kernel)
    return result
