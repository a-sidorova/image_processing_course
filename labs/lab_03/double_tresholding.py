import numpy as np
import math


def double_tresholding(img, low, high):
    down = int(255 * low)
    up = int(255 * high)

    h, w = img.shape
    res = np.zeros(shape=(h, w), dtype='uint8')
    for y in range(h - 1):
        for x in range(w - 1):
            if img[y, x] >= up:
                res[y, x] = 255
            elif img[y, x] <= down:
                res[y, x] = 0
            else:
                res[y, x] = 127
    return res
