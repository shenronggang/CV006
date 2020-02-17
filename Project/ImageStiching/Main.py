#   项目：图像拼接
#   日期：2020.02.15
#   小组成员：沈荣港 张虓 吴瀚宇

import cv2
import numpy as np
from matplotlib import pyplot as plt


img1_original = cv2.imread('Image_1.jpg')
img2_original = cv2.imread('Image_2.jpg')
LIMITATION = 0.99

# 使用orb算法获取特征点及其对应的描述子
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1_original, None)
kp2, des2 = orb.detectAndCompute(img2_original, None)

# 提取较好的特征点
bf = cv2.BFMatcher.create()
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
goodPoints = []
for i in range(len(matches)-1):
    if matches[i].distance < LIMITATION * matches[i+1].distance:
        goodPoints.append(matches[i])

img_match = cv2.drawMatches(img1_original, kp1, img2_original, kp2, goodPoints, flags=2, outImg=None)

# 计算单应性矩阵，并使用RANSAC算法降噪
src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# 对第二幅图像进行单应性变换
h1, w1, p1 = img2_original.shape
h2, w2, p2 = img1_original.shape
h = np.maximum(h1, h2)
w = np.maximum(w1, w2)
dis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))
image2_transform = cv2.warpPerspective(img2_original, M, (w1+w2-dis, h))

# 拼接
M1 = np.float32([[1, 0, 0], [0, 1, 0]])
dst1 = cv2.warpAffine(img1_original, M1, (w1+w2-dis, h))
dst = cv2.add(dst1, image2_transform)
dst_no = np.copy(dst)
img_target = np.maximum(dst1, image2_transform)

plt.subplot(311)
plt.imshow(img_match)
plt.subplot(312)
plt.imshow(dst_no)
plt.subplot(313)
plt.imshow(img_target)
plt.show()