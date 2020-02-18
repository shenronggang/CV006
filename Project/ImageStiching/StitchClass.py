import cv2
import numpy as np
from matplotlib import pyplot as plt


class Stitching:
    def __init__(self,img1,img2, LIMIT):
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)
        self.LIMIT = LIMIT

    def stitch(self):
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

        img_match = cv2.drawMatches(self.img1, kp1, self.img2, kp2, good_points, flags=2, outImg=None)

        # 计算单应性矩阵，并使用RANSAC算法降噪
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # 对第二幅图像进行单应性变换
        h1, w1, p1 = self.img2.shape
        h2, w2, p2 = self.img1.shape
        h = np.maximum(h1, h2)
        w = np.maximum(w1, w2)
        dis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))
        image2_transform = cv2.warpPerspective(self.img2, M, (w1 + w2 - dis, h))

        # 拼接
        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        dst1 = cv2.warpAffine(self.img1, M1, (w1 + w2 - dis, h))
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