import json
import cv2 as cv
import numpy as np
import math


intr = np.array([1221.445,0.0,637.226,0.0,1223.398,502.549,0.0,0.0,1.0]).reshape(3, 3)
dist = np.array([0.17716814022120847,-0.4573406421907695,0.002752733239126054,0.00036002841308005,0.17825941966394934])

w = 320
h = 240

center = [206/2, 313/2]

intr[0, 0] = intr[0, 0] * w / 1280
intr[0, 2] = intr[0, 2] * w / 1280
intr[1, 1] = intr[1, 1] * h / 960
intr[1, 2] = intr[1, 2] * h / 960

newcameramtx, roi = cv.getOptimalNewCameraMatrix(intr, dist, (w, h), 1, (w, h))

def undistort(img):

    dst = cv.undistort(img, intr, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    return dst[y:y + h, x:x + w]

src_points = np.array([
    [85, 4],  # Top-left corner
    [236, 3],  # Top-right corner
    [305, 233],  # Bottom-right corner
    [3, 226]  # Bottom-left corner
], dtype=np.float32)

scale = 15

p_w = 13.75
p_h = 20+(7/8)

# Define the destination points (corners in the bird's eye view)
dst_points = np.array([
    [0, 0],  # Top-left corner
    [int(p_w * scale), 0],  # Top-right corner
    [int(p_w * scale), int(p_h * scale)],  # Bottom-right corner
    [0, int(p_h * scale)]  # Bottom-left corner
], dtype=np.float32)

# Compute the homography matrix
transformation = cv.getPerspectiveTransform(src_points, dst_points)
inv = np.linalg.inv(transformation)
def get_birdseye(img):


    # Apply the homography transformation
    return cv.warpPerspective(img, transformation, (int(p_w * scale), int(p_h * scale)))


kernel = np.ones((3, 3), np.uint8)

def runPipeline(image, llrobot):

    undistorted = undistort(image)


    birdseye = get_birdseye(undistorted)    


    #print(birdseye.shape)

    img_hsv = cv.cvtColor(birdseye, cv.COLOR_BGR2HSV)
    binary = cv.inRange(img_hsv, (40, 0, 25), (60, 255, 125))
    masked = cv.bitwise_and(birdseye, birdseye, mask=binary)

    edges = cv.Canny(masked, 210, 30)

    img_dilation = cv.dilate(edges, kernel, iterations=1)

    #print(img_dilation.shape)


    contours, _ = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda c: (cv.contourArea(c) > 600) and (cv.contourArea(c) < 900) and (cv.minAreaRect(c)[0][1]<260) and (cv.minAreaRect(c)[0][1]>75), contours))


    p_out = []



    try:
        largest = min(contours, key = lambda c: math.dist([cv.minAreaRect(c)[0][0],cv.minAreaRect(c)[0][1]], center))
        #largest = max(contours, key = lambda c: cv.contourArea(c))
        
        rect = cv.minAreaRect(largest)
        #print(rect[0])
        #print(rect[2])
        warped_point = np.array([rect[0][0], rect[0][1]], dtype=np.float32)

        warped_point_homogeneous = np.array([warped_point[0], warped_point[1], 1.0])

        original_point_homogeneous = np.dot(inv, warped_point_homogeneous)

        original_point = original_point_homogeneous / original_point_homogeneous[2]


        # Extract the x and y coordinates
        original_point = original_point[:2]
        original_point = original_point.astype(int)
        print(rect[1][0]/rect[1][1])
        cv.circle(image, (original_point[0], original_point[1]), 2 ,(255, 0, 255), -1)
        p_out = [original_point[0], original_point[1], rect[2], rect[1][0], rect[1][1]]
    except ValueError:
        largest = []
    #print(largest)
    return largest, image, p_out