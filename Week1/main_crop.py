import cv2

from function import pic_display

img_ori = cv2.imread('lenna.jpg', 1)
img_gray = cv2.imread('lenna.jpg', 0)

pic_display(img_ori[150:300][0:200])
