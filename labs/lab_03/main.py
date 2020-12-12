import cv2
import sys
from time import perf_counter
from utils import mse, reshape_img
from canny import canny


def test_canny(img):
    start = perf_counter()
    res = canny(img)
    finish = perf_counter()
    cv2.imshow('Canny', res)
    print('Time = ' + str(finish - start))

    start = perf_counter()
    res_cv = cv2.Canny(img, 0.5, 0.7)
    finish = perf_counter()
    cv2.imshow('OpenCV Canny', res_cv)
    print('Time_opencv = ' + str(finish - start))

    mse_value = mse(res, res_cv)
    print('MSE: ' + str(mse_value) + '\n')


def main():
    path_to_img = r'../../resources/apple.jpg'

    print('========== HIGHLIGHTING BORDERS ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    img = reshape_img(img)
    cv2.imshow('Src image', img)

    print('I. Canny\n')
    test_canny(img)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
   sys.exit(main() or 0)
