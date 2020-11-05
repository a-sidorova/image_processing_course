import numpy as np
import cv2


def noise(img, A, B):
    h, w, ch = img.shape
    pix = np.zeros(shape=(h, w, ch), dtype='uint8')

    for i in range(w):
        for j in range(h):
            noise = np.random.randint(A, B)
            b = img[j, i, 0] + noise
            g = img[j, i, 1] + noise
            r = img[j, i, 2] + noise

            b = max(min(b, 255), 0)
            g = max(min(g, 255), 0)
            r = max(min(r, 255), 0)

            pix[j, i] = (b, g, r)

    pix = pix.astype(np.uint8)
    return pix
