import cv2
import numpy as np
import math
import Transforms as ref
import HSV

# Algorithm details:
# 1. Undistort camera
# 2. Map perspective transform
# 3. Convert to hsv and then threshold for red, blue, and yellow
# 4. Find Contours
# 5. Filter based on aspect ratio and minimum area
# 6. Draw contours if needed

img = cv2.imread("new/new1.jpg")

dim = (640, 480)
#img.resize(dim)
img = cv2.resize(img, dim)

# Distortion stuff is handled by logitech here

CONTOUR_THRESH_LOW = 500
CONTOUR_THRESH_HIGH = 4000

# Perspective transform
bounds = [(200, 48), (500, 46), (630, 360),  (60, 330)]

# for bound in bounds:
  # cv2.circle(img, bound, 2, (255, 0, 0), 2)

#cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

# cv2.polylines(img, [np.int32(bounds)], True, (0, 0, 0), 3)

matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
transformed_image = cv2.warpPerspective(img, matrix, dim)

mat = transformed_image.copy()
mat = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)
threshold = cv2.inRange(mat, HSV.blueLow, HSV.blueHigh)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(transformed_image, contours, -1, color=(0, 255, 0), thickness=2)

cv2.imshow("transformed", transformed_image)

cv2.waitKey(0)