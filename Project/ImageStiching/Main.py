#   项目：图像拼接
#   日期：2020.02.15
#   小组成员：沈荣港 张虓 吴瀚宇

import cv2
from Stitching import Stitcher
from matplotlib import pyplot as plt

imageA = cv2.imread('Image_1.jpg')
imageB = cv2.imread('Image_2.jpg')

stitcher = Stitcher()
(result, vis) = stitcher.stitch((imageA, imageB), showMatches=True)
plt.imshow(result), plt.show()
