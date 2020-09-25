import cv2

path_to_img = r'..\..\resources\flower.jpg'

img = cv2.imread(path_to_img)
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()

