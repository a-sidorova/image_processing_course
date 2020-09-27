import cv2
import numpy as np


def mse(img, img_opencv):
    height, width, _ = img.shape
    mse_value = 0
    img = img.astype(np.float32, copy=False)
    img_opencv = img_opencv.astype(np.float32, copy=False)
    for coord_x in range(width):
        for coord_y in range(height):
            diff = np.sum(abs(img[coord_y, coord_x] - img_opencv[coord_y, coord_x]))
            mse_value = mse_value + diff * diff
    mse_value = mse_value / (3.0 * float(height * width))
    return mse_value