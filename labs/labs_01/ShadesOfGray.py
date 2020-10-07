import cv2
import numpy as np

def monochromeImg(imgMono):
    h, w, _ = imgMono.shape
    resImg = np.zeros(shape=(h, w, 1), dtype='uint8')

    for x in range(w):
        for y in range(h):
            b = imgMono[y, x, 0]
            g = imgMono[y, x, 1]
            r = imgMono[y, x, 2]

            s = 0.2952 * r + 0.5547 * g + 0.148 * b
            resImg[y, x] = s

    return resImg

def monochromeImgOpenCV(imgMono):
    return cv2.cvtColor(imgMono, cv2.COLOR_BGR2GRAY)