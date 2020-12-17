import cv2
import numpy as np
from distance_transform import distance_transform
from shades_of_gray import monochrome_img


def watershed(img, distances):
    image = np.copy(img)
    gray_img = monochrome_img(image)
    _, bin_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    background = cv2.dilate(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
    ret, markers = cv2.connectedComponents(background)

    _, foreground = cv2.threshold(distances, distances.max() * 0.6, 255, cv2.THRESH_BINARY)

    markers = markers + 1
    unknown = cv2.subtract(background, np.uint8(foreground))
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers=markers)

    image[markers == -1] = (0, 0, 255)
    return image
