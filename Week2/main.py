import cv2
import matplotlib.pyplot as plt
import numpy as np

from function import pic_display

img = cv2.imread('lenna.jpg', 1)

g_img = cv2.GaussianBlur(img, (7, 7), 2) #卷积操作

#pic_display(g_img)

#cv2.getGaussianKernel()

#cv2.sepFilter2D()

kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
lap_img = cv2.filter2D(img, -1, kernel) #深度为-1，使新图与原图
pic_display(lap_img)
