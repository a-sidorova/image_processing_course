import cv2
import sys
from time import perf_counter
from utils import mse, reshape_img
from canny import canny
from distance_transform import distance_transform
from shades_of_gray import monochrome_img
from watershed import watershed


def test_canny(img):
    start = perf_counter()
    res = canny(img)
    finish = perf_counter()
    cv2.imshow('Canny', res)
    print('Time = ' + str(finish - start))

    start = perf_counter()
    res_cv = cv2.Canny(img, 50, 70)
    finish = perf_counter()
    cv2.imshow('OpenCV Canny', res_cv)
    print('Time_opencv = ' + str(finish - start))

    mse_value = mse(res, res_cv)
    print('MSE: ' + str(mse_value) + '\n')


def test_distance_transform(img):
    gray_img = monochrome_img(img)
    _, bin_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

    start = perf_counter()
    res = distance_transform(bin_img)
    finish = perf_counter()
    cv2.imshow('Distance transform', res)
    print('Time = ' + str(finish - start) + '\n')


def test_watershed(img):
    gray_img = monochrome_img(img)
    _, bin_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    distances = distance_transform(bin_img)

    start = perf_counter()
    watershed_img = watershed(img, distances)
    finish = perf_counter()
    cv2.imshow('Watershed', watershed_img)
    print('Time = ' + str(finish - start) + '\n')


def main():
    path_to_img = r'../../resources/money.jpg'

    print('========== HIGHLIGHTING BORDERS ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    img = reshape_img(img)
    cv2.imshow('Src image', img)

    print('I. Canny\n')
    test_canny(img)
    cv2.waitKey()

    print('II. Distance Transform\n')
    test_distance_transform(img)
    cv2.waitKey()

    print('III. Watershed\n')
    test_watershed(img)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
