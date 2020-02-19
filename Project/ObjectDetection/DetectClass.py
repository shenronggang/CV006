import cv2
import numpy as np
from matplotlib import pyplot as plt


class Detection:
    def __init__(self, img1, img2, LIMIT):
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)
        self.LIMIT = LIMIT

    def detect(self):
        # 使用orb算法获取特征点及其对应的描述子
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.img1, None)
        kp2, des2 = orb.detectAndCompute(self.img2, None)

        # 提取较好的特征点
        bf = cv2.BFMatcher.create()
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_points = []
        for i in range(len(matches) - 1):
            if matches[i].distance < self.LIMIT * matches[i + 1].distance:
                good_points.append(matches[i])

        # 计算单应性矩阵，并使用RANSAC算法降噪
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # 绘制目标边框
        result = cv2.warpPerspective(self.img2, M, (self.img2.shape[1] + self.img1.shape[1], self.img1.shape[0]))
        img3 = result[0:self.img1.shape[0], 0:self.img1.shape[1]]
        plt.imshow(img3)
        plt.show()