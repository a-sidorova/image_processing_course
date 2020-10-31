import numpy as np


class Intensity:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.value = 0.36 * color[2] + 0.53 * color[1] + 0.11 * color[0]


def calculate_new_color(img, x, y, radius):
    height, width, _ = img.shape
    intensities = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            idx = max(min(x + i, width - 1), 0)
            idy = max(min(y + j, height - 1), 0)
            intensities.append(Intensity(idx, idy, img[idy, idx]))
    sorted_intensities = sorted(intensities, key=lambda intensity: intensity.value)
    median = sorted_intensities[len(sorted_intensities) // 2]
    return img[median.y, median.x]


def median_filter(img):
    height, width, _ = img.shape
    result = np.zeros(shape=(height, width, 3), dtype='uint8')
    radius = 1
    for x in range(width):
        for y in range(height):
            result[y, x] = calculate_new_color(img, x, y, radius)
    return result
