import numpy as np
import cv2

def getGaussianNoise(imgGauss, sigma_var):
    h, w, ch = imgGauss.shape
    mean = 0
    sigma = sigma_var
    gauss = np.random.normal(mean, sigma, (h, w, ch))
    gauss = gauss.reshape(h, w, ch)

    noisyImg = imgGauss + gauss
    noisyImg = np.clip(noisyImg, 0, 255)
    noisyImg = noisyImg.astype(np.uint8)
    return noisyImg
