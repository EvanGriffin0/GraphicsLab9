import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ATU1.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()
