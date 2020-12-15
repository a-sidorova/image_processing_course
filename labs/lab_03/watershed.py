import cv2
from shades_of_gray import monochrome_img
from distance_transform import distanceTransform
import numpy as np


def normalize(img):
    height, width = img.shape
    img1 = img
    for i in range(height):
        for j in range(width):
            img1[i, j] = img1[i, j] / 255

    return img1


def watershed(img):
    bimg_array = img
    bimg1 = monochrome_img(bimg_array)
    _, bimg = cv2.threshold(bimg1, 100, 255, cv2.THRESH_BINARY)

    dist_transf_img = distanceTransform(bimg)
    norm_img = normalize(dist_transf_img)
    bimg1 = cv2.threshold(bimg1, 0.5, 1, cv2.THRESH_BINARY)

    convert_img = np.copy(bimg1, dtype='uint8')
    contours = cv2.findContours(convert_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size_contours = len(contours)
    print(size_contours)
    markers = np.zeros(size_contours, dtype='nx1')

    for i in range(size_contours):
        cv2.drawContours(markers, contours, i, ((i + 1), (i + 1), (i + 1)), -1)

    cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)

    cv2.watershed(img, markers)

    res = np.zeros(len(markers), dtype='uint8')
