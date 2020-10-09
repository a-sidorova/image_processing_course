import cv2

import sys
from time import perf_counter
from utils import *
from color_models import *
from ShadesOfGray import *

def test_shades_of_gray(img):
    t1_start = perf_counter()
    monoImg = monochromeImg(img)
    t1_finish = perf_counter()
    cv2.imshow('Monochrome Image', monoImg)
    print('Time = ' + str(t1_finish - t1_start))

    t2_start = perf_counter()
    monoImgCV = monochromeImgOpenCV(img)
    t2_finish = perf_counter()
    cv2.imshow('OpenCV Monochrome Image', monoImgCV)
    print('Time_opencv = ' + str(t2_finish - t2_start))

def test_color_models(img):
    # Converting BGR to HSV
    print('[Loading...] Converting BGR to HSV')
    start = perf_counter()
    hsv_img = convert_BGR_to_HSV(img)
    finish = perf_counter()
    cv2.imshow('BGR2HSV', hsv_img)
    print('Time: ' + str(finish - start))

    print('[Loading...] Converting BGR to HSV with OpenCV')
    start = perf_counter()
    hsv_img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    finish = perf_counter()
    cv2.imshow('BGR2HSV_OpenCV', hsv_img_cv)
    print('Time: ' + str(finish - start))

    mse_value = mse(hsv_img, hsv_img_cv)
    print('MSE: ' + str(mse_value) + '\n')

    # Converting HSV to BGR
    print('[Loading...] Converting HSV to BGR')
    start = perf_counter()
    bgr_img = convert_HSV_to_BGR(hsv_img)
    finish = perf_counter()
    cv2.imshow('HSV2BGR', bgr_img)
    print('Time: ' + str(finish - start))

    print('[Loading...] Converting HSV to BGR with OpenCV')
    start = perf_counter()
    bgr_img_cv = cv2.cvtColor(hsv_img_cv, cv2.COLOR_HSV2BGR)
    finish = perf_counter()
    cv2.imshow('HSV2BGR_OpenCV', bgr_img_cv)
    print('Time: ' + str(finish - start))

    mse_value = mse(bgr_img, bgr_img_cv)
    print('MSE: ' + str(mse_value) + '\n')

    # Brigthness
    coeff = 30
    print('[Loading...] Increase in brightness for BGR')
    bgr_brigth_img = brightness_BGR(img, coeff)
    print('[Success]')
    cv2.imshow('BGR_Brigthness', bgr_brigth_img)

    print('[Loading...] Increase in brightness for HSV')
    hsv_brigth_img = brightness_HSV(hsv_img, coeff)
    print('[Success]')
    cv2.imshow('HSV_Brigthness', hsv_brigth_img)


def main():
    path_to_img = r'..\..\resources\flower.jpg'

    print('========== SPOTS FILTERS ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    cv2.imshow('Src image', img)

    print('I. Image similarity metric\n')


    print('II. Gray scale\n')
    test_shades_of_gray(img)
    cv2.waitKey()

    print('III. Color models\n')
    test_color_models(img)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
   sys.exit(main() or 0)

