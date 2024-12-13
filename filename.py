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

corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)
imgShiTomasi = img.copy()
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imgShiTomasi, (x, y), 3, (255, 0, 0), -1)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.show()

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
imgORB = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(imgORB, cv2.COLOR_BGR2RGB))
plt.show()
