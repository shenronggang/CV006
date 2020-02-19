#   项目：图像拼接
#   日期：2020.02.15
#   小组成员：沈荣港 张虓 吴瀚宇

import DetectClass

LIMITATION = 0.99

a = DetectClass.Detection('Image_1.jpg', 'Image_2.jpg', LIMITATION)
a.detect()
