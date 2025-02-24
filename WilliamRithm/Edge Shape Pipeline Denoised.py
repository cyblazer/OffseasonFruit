import cv2
import numpy as np
import math
import Transforms as ref

# Algorithm details:
# 1. Undistort camera
# 2. Map perspective transform
# 3. Convert to hsv and then threshold for red, blue, and yellow
# 4. Find Contours
# 5. Filter based on aspect ratio and minimum area
# 6. Draw contours if needed

img = cv2.imread("imgs/img3.jpg")

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

edges = cv2.Canny(img, 100, 350)
edges = cv2.warpPerspective(edges, matrix, dim)

for i in range(0):
  edges = cv2.GaussianBlur(edges,(5,5),0)

#Edge Noise Mask
empty = np.zeros(transformed_image.shape[:2], np.uint8)
edge_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in edge_contours:
   if (len(contour) < 50):
      cv2.drawContours(empty, contour, -1, 255, thickness=cv2.FILLED)
cv2.bitwise_not(empty, empty)
cv2.imshow("empty", empty)

edges = cv2.bitwise_and(edges, empty)
cv2.imshow("post denoise", edges)
# empty is white for empty space, black for stuff you want to remove
# in order to merge you bitwise and 

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

cv2.drawContours(transformed_image, contours, -1, color=(0, 255, 0), thickness=2)

points = []

for contour in contours:
  M = cv2.moments(contour) 
  if M['m00'] != 0.0: 
      x = int(M['m10']/M['m00']) 
      y = int(M['m01']/M['m00']) 
      points.append([x, y])
      cv2.circle(edges, (x, y), 2, 255, 2)


cv2.imshow("edges", edges)
cv2.imshow("transformed", transformed_image)

cv2.waitKey(0)