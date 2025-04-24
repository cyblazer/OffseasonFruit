import cv2
import numpy as np
import math
import HSV
import Transforms as ref

img = cv2.imread("imgs/adam1.jpg")

dim = (640, 480)
img = cv2.resize(img, dim)

# Distortion stuff is handled by logitech here
# Automatically undistorted, so there's no need for us to do that

bounds = ref.adam1

matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
transformed_image = cv2.warpPerspective(img, matrix, dim)

#Find Edges
edges = cv2.Canny(transformed_image, 10, 100)

cv2.imshow("masked edges", edges)
cv2.imshow("transformed", transformed_image)

#Build line segment detector
lsd = cv2.createLineSegmentDetector(0)

# Detect lines
lines = lsd.detect(edges)[0]

mask = np.zeros(img.shape[:2], np.uint8)

# #Draw lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)

cv2.imshow("mask", mask)

drawn_img = lsd.drawSegments(transformed_image, lines)

cv2.imshow("lsd", drawn_img)

shape_contours, _ = cv2.findContours( 
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

# SHAPE DETECTION

coords = []

for contour in shape_contours[1:]:
  # cv2.approxPloyDP() function to approximate the shape 
  approx = cv2.approxPolyDP( 
      contour, 0.01 * cv2.arcLength(contour, True), True) 
    
  size = cv2.contourArea(contour)
  if size < 40:
     continue

  # finding center point of shape 
  M = cv2.moments(contour) 
  if M['m00'] != 0.0: 
      x = int(M['m10']/M['m00']) 
      y = int(M['m01']/M['m00']) 

  cv2.circle(transformed_image, (x, y), 3, (255, 0, 0), 1)
  coords.append([x, y])

for bound in bounds:
  cv2.circle(img, bound, 2, (255, 0, 0), 2)

cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

cv2.imshow("raw", img)
cv2.waitKey(0)