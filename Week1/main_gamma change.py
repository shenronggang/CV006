import cv2

from function import pic_display, adjust_gamma

img_dark = cv2.imread('dark.jpg', 1)
img_dark_lighter = adjust_gamma(img_dark, 2)
pic_display(img_dark_lighter)
