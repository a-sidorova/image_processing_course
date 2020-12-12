from shades_of_gray import monochrome_img
from gauss_filter import gauss_filter
from sobel import sobel
from non_max_suppression import non_max_suppression
from double_tresholding import double_tresholding


def canny(img):
    gray_img = monochrome_img(img)
    gauss_img = gauss_filter(gray_img)
    sobel_img, grad_img = sobel(gauss_img)
    non_max_suppression_img = non_max_suppression(sobel_img, grad_img)
    result = double_tresholding(non_max_suppression_img, 0.5, 0.7)
    return result
