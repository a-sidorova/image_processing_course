import cv2
import numpy as np


def mse(img, img_opencv):
    if len(img.shape) == 2:
        height, width = img.shape
        num_of_channels = 1.0
    else:
        height, width, _ = img.shape
        num_of_channels = 3.0
    img = img.astype(np.float32, copy=False)
    img_opencv = img_opencv.astype(np.float32, copy=False)
    diff = (img - img_opencv) ** 2
    summ = np.sum(diff)
    mse_value = summ / (num_of_channels * float(height * width))
    return mse_value
