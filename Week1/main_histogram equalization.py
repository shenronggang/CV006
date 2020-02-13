import cv2
import matplotlib.pyplot as plt

from function import pic_display, adjust_gamma

img_dark = cv2.imread('dark.jpg', 1)
img_dark_lighter = adjust_gamma(img_dark, 3)

plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color='b')
plt.subplot(122)
plt.hist(img_dark_lighter.flatten(), 256, [0, 256], color='r')
plt.show()

img_yuv = cv2.cvtColor(img_dark, cv2.COLOR_BGR2YUV)
print(img_yuv)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
pic_display(img_out)
