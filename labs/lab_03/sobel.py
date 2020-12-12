import numpy as np
import math


def sobel(img):
    m_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    m_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    h, w = img.shape
    res = np.zeros(shape=(h, w), dtype='float')
    res_grad = np.zeros(shape=(h, w), dtype='float')
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g_x = 0
            g_y = 0
            for p in range(-1, 2):
                for q in range(-1, 2):
                    g_x += img[y + p, x + q] * m_x[p + 1, q + 1]
                    g_y += img[y + p, x + q] * m_y[p + 1, q + 1]
            g = math.sqrt(g_x * g_x + g_y * g_y)
            t = int(math.atan2(g_x, g_y) / (math.pi / 4)) * (math.pi / 4) - math.pi / 2 if g != 0 else -1
            res[y, x] = g
            res_grad[y, x] = t
    return res, res_grad
