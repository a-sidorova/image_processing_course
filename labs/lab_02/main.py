import cv2

import sys
from time import perf_counter
from utils import mse

def main():
    path_to_img = r'..\..\resources\flower.jpg'

    print('========== NOICE SUPPRESSION ==========\n')
    print('Src image: ' + path_to_img + '\n')
    img = cv2.imread(path_to_img)
    cv2.imshow('Src image', img)

    print('I. ...\n')

    print('II. ...\n')

    print('II. ...\n')

    cv2.destroyAllWindows()


if __name__ == '__main__':
   sys.exit(main() or 0)

