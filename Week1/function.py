import cv2
import matplotlib.pyplot as plt
import numpy as np

def pic_display(img, size=(6, 6)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def img_cooler(img, b_increase, r_decrease):
    b, g, r = cv2.split(img)
    b_lim = 255 - b_increase
    b[b > b_lim] = 255
    b[b <= b_lim] = b_increase + b[b <= b_lim]
    # use 'astype' can guarantee that the data type is floating-point

    r_lim = r_decrease
    r[r < r_lim] = 0
    r[r >= r_lim] = r[r >= r_lim] - r_decrease

    return cv2.merge((r, g, b))

def adjust_gamma(image, gamma=1.0):
    gamma_inv = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** gamma_inv) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(image, table)

