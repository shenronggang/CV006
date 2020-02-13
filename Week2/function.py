import cv2
import matplotlib.pyplot as plt
import numpy as np

def pic_display(img, size=(6, 6)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
