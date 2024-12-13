import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ATU1.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()


dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

imgHarris = img.copy()

threshold = 0.1  # Adjust this value
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i, j] > threshold * dst.max():
            cv2.circle(imgHarris, (j, i), 3, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.show()
