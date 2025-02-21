import cv2
import numpy as np

img = cv2.imread("img3.jpg")
img = cv2.resize(img, (600, 800), img)

dim = (600, 800)



edges = cv2.Canny(img,100, 200, apertureSize=3)

cv2.imshow("Canny", edges)


#matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
#transformed_image = cv2.warpPerspective(img, matrix, dim)

#cv2.imshow("Transformed - Top Down", transformed_image)
cv2.waitKey(0)