import numpy as np


def monochrome_img(img):
    h, w, _ = img.shape
    res = np.zeros(shape=(h, w), dtype='uint8')

    for x in range(w):
        for y in range(h):
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]

            s = 0.2952 * r + 0.5547 * g + 0.148 * b
            res[y, x] = s

    return res
