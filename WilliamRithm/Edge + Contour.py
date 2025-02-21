import cv2
import numpy as np
import math

# Algorithm details:
# 1. Undistort camera
# 2. Map perspective transform
# 3. Convert to hsv and then threshold for red, blue, and yellow
# 4. Find Contours
# 5. Filter based on aspect ratio and minimum area
# 6. Draw contours if needed

img = cv2.imread("img3.jpg")

dim = (640, 480)
#img.resize(dim)
img = cv2.resize(img, dim)

# Distortion stuff is handled by logitech here

CONTOUR_THRESH_LOW = 500
CONTOUR_THRESH_HIGH = 4000

# Perspective transform
# bounds = [(140, 48), (500, 46), (710, 360),  (0, 350)]
bounds = [(200, 48), (500, 46), (630, 360),  (60, 330)]

# for bound in bounds:
  # cv2.circle(img, bound, 2, (255, 0, 0), 2)

#cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

# cv2.polylines(img, [np.int32(bounds)], True, (0, 0, 0), 3)

matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
transformed_image = cv2.warpPerspective(img, matrix, dim)

# kernel = np.array([[0, 1, 1, 1, 0],
#                   [1, 1, 1, 1, 1],
#                   [1, 1, 1, 1, 1],
#                   [1, 1, 1, 1, 0],
#                   [0, 1, 1, 1, 1]], dtype=np.int32)
# element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2), (-1, -1))
# transformed_image = cv2.erode(transformed_image, element)
# transformed_image = cv2.dilate(transformed_image, element)
# transformed_image = cv2.morphologyEx(transformed_image, cv2.MORPH_OPEN, element)

#Blur -> For smoothing
# for i in range(2):
#   element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5), (-1, -1))
#   img = cv2.erode(img, element)
#   img = cv2.dilate(img, element)
#   img = cv2.GaussianBlur(img,(5,5),0)

edges = cv2.Canny(img, 100, 350)
edges = cv2.warpPerspective(edges, matrix, dim)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

cv2.drawContours(transformed_image, contours, -1, color=(0, 255, 0), thickness=2)

cv2.imshow("edges", edges)
cv2.imshow("transformed", transformed_image)

cv2.waitKey(0)

#Standard Blue Yellow Red Contour Detection Below

# # --- BLUE CONSTANTS ---
# blueLow = np.array([0, 60, 30])
# blueHigh = np.array([80, 255, 255])

# # --- RED CONSTANTS ---
# redLow = np.array([115, 60, 80])
# redHigh = np.array([180, 255, 2])

# # --- YELLOW CONSTANTS ---
# yellowLow = np.array([20, 80, 80])
# yellowHigh = np.array([30, 255, 255])

# hsv_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2HSV)
# # Make masks
# red_threshold = cv2.inRange(hsv_image, redLow, redHigh)
# blue_threshold = cv2.inRange(hsv_image, blueLow, blueHigh)
# yellow_threshold = cv2.inRange(hsv_image, yellowLow, yellowHigh)

# CONTOUR_THRESH_LOW = 500
# CONTOUR_THRESH_HIGH = 4000

# rc, _ = cv2.findContours(red_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# bc, _ = cv2.findContours(blue_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# yc, _ = cv2.findContours(yellow_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# #contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

# cv2.drawContours(transformed_image, bc, -1, color=(0, 255, 0), thickness=5)

# for bound in bounds:
#   cv2.circle(img, bound, 2, (255, 0, 0), 2)

# cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

# cv2.imshow("raw", img)
# cv2.imshow("transformed", transformed_image)
# cv2.waitKey(0)