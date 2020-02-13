import cv2

from function import pic_display, img_cooler

img_ori = cv2.imread('lenna.jpg', 1)    # read colour picture
img_gray = cv2.imread('lenna.jpg', 0)   # read black-and-white picture

B, G, R = cv2.split(img_ori)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
key = cv2.waitKey(0)
if key == 27:   # '27' represent 'Esc'
    cv2.destroyAllWindows()

cooler_img = img_cooler(img_ori, 30, 10)
pic_display(cooler_img)
