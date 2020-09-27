import cv2
import numpy as np


def convert_BGR_to_HSV(img):
    height, width, _ = img.shape
    result = np.zeros(shape=(height, width, 3), dtype='uint8')
    for coord_x in range(width):
        for coord_y in range(height):
            r = img[coord_y, coord_x, 2] / 255.0
            g = img[coord_y, coord_x, 1] / 255.0
            b = img[coord_y, coord_x, 0] / 255.0

            min_value = min(r, g, b)
            max_value = max(r, g, b)
            diff = max_value - min_value

            if diff == 0:
                h = 0
            elif max_value == r and g >= b:
                h = 60 * ((g - b) / diff)
            elif max_value == r and g < b:
                h = 60 * ((g - b) / diff) + 360
            elif max_value == g:
                h = 60 * ((b - r) / diff) + 120
            elif max_value == b:
                h = 60 * ((r - g) / diff) + 240

            s = 0 if max_value == 0 else diff / max_value
            v = max_value

            result[coord_y, coord_x] = (h / 2.0, s * 255.0, v * 255.0)
    return result


def convert_HSV_to_BGR(img):
    height, width, _ = img.shape
    result = np.zeros(shape=(height, width, 3), dtype='uint8')
    for coord_x in range(width):
        for coord_y in range(height):
            h = img[coord_y, coord_x, 0] * 2
            s = img[coord_y, coord_x, 1] / 255.0
            v = img[coord_y, coord_x, 2] / 255.0

            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c

            if h < 60:
                bgr = [0, x, c]
            elif h < 120:
                bgr = [0, c, x]
            elif h < 180:
                bgr = [x, c, 0]
            elif h < 240:
                bgr = [c, x, 0]
            elif h < 300:
                bgr = [c, 0, x]
            elif h <= 360:
                bgr = [x, 0, c]

            b = (bgr[0] + m) * 255
            g = (bgr[1] + m) * 255
            r = (bgr[2] + m) * 255

            result[coord_y, coord_x] = (b, g, r)
    return result


def brightness_BGR(img, coeff):
    height, width, _ = img.shape
    result = np.zeros(shape=(height, width, 3), dtype='uint8')
    for coord_x in range(width):
        for coord_y in range(height):
            result[coord_y, coord_x, 2] = max(min(img[coord_y, coord_x, 2] + coeff, 255), 0)
            result[coord_y, coord_x, 1] = max(min(img[coord_y, coord_x, 1] + coeff, 255), 0)
            result[coord_y, coord_x, 0] = max(min(img[coord_y, coord_x, 0] + coeff, 255), 0)
    return result

def brightness_HSV(img, coeff):
    height, width, _ = img.shape
    result = img
    for coord_x in range(width):
        for coord_y in range(height):
            result[coord_y, coord_x, 2] = max(min(img[coord_y, coord_x, 2] + coeff, 255), 0)
    return result