import cv2
import numpy as np

asdf = []


img = cv2.imread("imgs/img0.png")
# make the img actually visible
img = cv2.resize(img, (600, 800), img)

# --- RED CONSTANTS ---
lowHSV = np.array([0, 130, 110])
highHSV = np.array([100, 255, 255])
CONTOUR_THRESH_LOW = 500
CONTOUR_THRESH_HIGH = 4000


dim = (600, 800)

# image 1
# bounds = [(123, 453), (428, 321), (621, 421), (302, 664)]

# image 2
bounds = [(0, 60), (500, 40), (620, 620), (-80, 560)]


matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
transformed_image = cv2.warpPerspective(img, matrix, dim)

edges = cv2.Canny(img,100,200)

edges = cv2.warpPerspective(edges, matrix, dim)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgRed = cv2.dilate(edges, kernel, iterations=3)
imgRed = cv2.morphologyEx(imgRed, cv2.MORPH_OPEN,
                           kernel,
                           iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

cv2.drawContours(transformed_image, contours, -1, color=(0, 255, 0), thickness=2)

cv2.imshow("edges", edges)
cv2.imshow("done", transformed_image)

cv2.waitKey(0)