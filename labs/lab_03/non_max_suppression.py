import numpy as np
import math


def non_max_suppression(sobel, grad):
    h, w = sobel.shape
    res = np.zeros(shape=(h, w), dtype='float')
    for y in range(h - 1):
        for x in range(w - 1):
            if (grad[y, x] == -1):
                continue
            dx = sign(math.cos(grad[y, x]))
            dy = -1 * sign(math.sin(grad[y, x]))
            if is_correct_idx(w, h, x + dx, y + dy):
                if sobel[y + dy, x + dx] <= sobel[y, x]:
                    res[y + dy, x + dx] = 0
            if is_correct_idx(w, h, x - dx, y - dy):
                if sobel[y - dy, x - dx] <= sobel[y, x]:
                    res[y - dy, x - dx] = 0
            res[y, x] = sobel[y, x]
    return res


def sign(val):
    return 0 if val == 0 else (1 if val > 0 else -1)


def is_correct_idx(rows, cols, x, y):
    return x >= 0 and y >= 0 and x <= (rows - 1) and y <= (cols - 1)
