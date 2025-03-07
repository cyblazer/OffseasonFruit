import cv2
import numpy as np

left = cv2.imread("AllenRithm/stereoleft.png")
right = cv2.imread("AllenRithm/stereoright.png")

cv2.circle(left, (150, 300), 3, 255, 3, 2)

# depth processing



cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.waitKey(0)