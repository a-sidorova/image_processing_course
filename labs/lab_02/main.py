import cv2
import sys
from time import perf_counter
from utils import mse, reshape_img
from median_filter import median_filter
from gauss_filter import gauss_filter


def test_noise(img):
    print('\t[Loading...] Calculating..')
    start = perf_counter()
    result = noise(img, -9, 20)
    finish = perf_counter()
    cv2.imshow('Add noise', result)
    print('\tTime: ' + str(finish - start))


def test_median_filter(img):
    print('\t[Loading...] Calculating..')
    start = perf_counter()
    result = median_filter(img)
    finish = perf_counter()
    cv2.imshow('Median Filter', result)
    print('\tTime: ' + str(finish - start))


def test_gauss_filter(img):
    print('\t[Loading...] Calculating..')
    start = perf_counter()
    result = gauss_filter(img)
    finish = perf_counter()
    cv2.imshow('Gauss Filter', result)
    print('\tTime: ' + str(finish - start))


def test_opencv_bilateral_filter(img):
    print('\t[Loading...] Calculating..')
    result = img.copy()
    start = perf_counter()
    cv2.bilateralFilter(result, 5, 50, 100)
    finish = perf_counter()
    cv2.imshow('OpenCV Bilateral Filter', result)
    print('\tTime: ' + str(finish - start))


def main():
    path_to_img = r'..\..\resources\noise.jpg'

    print('========== NOICE SUPPRESSION ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    img = reshape_img(img)
    cv2.imshow('Src image', img)


    print('I. Making noise\n')
    test_noise(img)
    cv2.waitKey()

    print('II. Deleting noise')
    print('\tII.a Median filter')
    test_median_filter(img)
    cv2.waitKey()
    print('\tII.b Gauss filter')
    test_gauss_filter(img)
    cv2.waitKey()
    print('\tII.c OpenCV Bilateral filter')
    test_opencv_bilateral_filter(img)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
   sys.exit(main() or 0)

