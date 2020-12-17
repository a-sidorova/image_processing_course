import cv2 as cv
import numpy as np
from distance_transform import distanceTransform


def watershed(img):
    image = np.copy(img)

    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(mb, kernel, iterations=3)

    dist = distanceTransform(mb)

    ret, sure_fg = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(sure_bg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers=markers)

    image[markers == -1] = (0, 0, 255)

    return image
