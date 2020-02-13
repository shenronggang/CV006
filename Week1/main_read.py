import cv2
import matplotlib.pyplot as plt

img_ori = cv2.imread('lenna.jpg', 1)    # read colour picture
img_gray = cv2.imread('lenna.jpg', 0)   # read black-and-white picture

print(img_ori.shape)
print(img_gray.shape)

cv2.imshow('lenna_photo', img_ori)
key = cv2.waitKey(0)
if key == 27:   # '27' represent 'Esc'
    cv2.destroyAllWindows()

plt.subplot(121)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))  # convert the default BGR to RGB
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')   # the default image parameter of plt is colorful
plt.show()
