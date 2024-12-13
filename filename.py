import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Step 1: Load and preprocess the image
image_path = 'ATU1.jpeg'  # Replace with your image path
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot grayscale image
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Grayscale')
plt.imshow(gray, cmap='gray')

# Step 2: Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
imgHarris = img.copy()
threshold = 0.1
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i, j] > threshold * dst.max():
            cv2.circle(imgHarris, (j, i), 3, (0, 255, 0), -1)

# Plot Harris corners
plt.subplot(2, 2, 2)
plt.title('Harris Corners')
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))

# Step 3: Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
if corners is not None:
    corners = np.int32(corners)
    imgShiTomasi = img.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(imgShiTomasi, (x, y), 3, (255, 0, 0), -1)

# Plot Shi-Tomasi corners
plt.subplot(2, 2, 3)
plt.title('Shi-Tomasi Corners')
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))

# Step 4: ORB Keypoint Detection
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
imgORB = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot ORB keypoints
plt.subplot(2, 2, 4)
plt.title('ORB Keypoints')
plt.imshow(cv2.cvtColor(imgORB, cv2.COLOR_BGR2RGB))

plt.show()

