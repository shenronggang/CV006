import cv2
import matplotlib.pyplot as plt
import numpy as np

from function import pic_display

img = cv2.imread('lenna.jpg', 1)

##g_img = cv2.GaussianBlur(img, (7, 7), 2) #卷积操作

#pic_display(g_img)

#cv2.getGaussianKernel()

#cv2.sepFilter2D()

#pic_display(lap_img)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detec(img)