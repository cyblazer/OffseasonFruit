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

edges = cv2.Canny(img, 100, 350)
edges = cv2.warpPerspective(edges, matrix, dim)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

def filterByRatio(sample) -> bool:
    if len(sample) != 4:
        return False

    #sample = cv2.boxPoints(cv2.minAreaRect(sample))

    width = np.linalg.norm(sample[0] - sample[1])
    length = np.linalg.norm(sample[1] - sample[2])

    ratio = length / width
    ratioInverted = width / length

    area = length * width

    MIN_AREA = 0
    MAX_AREA = 1000000000
    
    if (area < MIN_AREA) or (area > MAX_AREA):
        return False

    MIN_RATIO = 2
    MAX_RATIO = 1000

    if ((ratio > MIN_RATIO) and (ratio < MAX_RATIO)) or ((ratioInverted > MIN_RATIO) and (ratioInverted < MAX_RATIO)):
        return True
    return False

def mapToQuad(sample):
    arcLength = cv2.arcLength(sample, True)
    epsilon = 0.01

    output = cv2.approxPolyDP(sample, epsilon * arcLength, True)
    while len(output) > 4:
        epsilon += 0.005
        output = cv2.approxPolyDP(sample, epsilon * arcLength, True)

    return output

def filterByRectangleness(sample) -> bool:
    contourArea = cv2.contourArea(sample)
    rectArea = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(sample)))

    ratio = contourArea / rectArea

    if ratio > 1.5 or ratio < 0.5:
        return False
    return True

#contours = [np.intp(cv2.boxPoints(cv2.minAreaRect(sample))) for sample in contours]

#contours = [cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True) for contour in contours]

contours = list(map(mapToQuad, contours))

contours = list(filter(filterByRatio, contours))

contours = list(filter(filterByRectangleness, contours))

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