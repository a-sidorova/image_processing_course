import cv2
import sys
from time import perf_counter
from utils import mse, reshape_img


def main():
    path_to_img = r'..'

    print('========== HIGHLIGHTING BORDERS ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    img = reshape_img(img)
    cv2.imshow('Src image', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
   sys.exit(main() or 0)
