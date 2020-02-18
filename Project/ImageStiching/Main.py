#   项目：图像拼接
#   日期：2020.02.15
#   小组成员：沈荣港 张虓 吴瀚宇

import StitchClass
import cv2
import numpy as np
from matplotlib import pyplot as plt

LIMITATION = 0.99

a = StitchClass.Stitching('Image_1.jpg','Image_2.jpg',LIMITATION)
a.stitch()