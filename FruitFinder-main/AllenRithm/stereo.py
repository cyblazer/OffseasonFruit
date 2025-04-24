import numpy as np
import cv2

left = cv2.imread("AllenRithm/stereoleft.png")
right = cv2.imread("AllenRithm/stereoright.png")
left_window_name = "left"
right_window_name = "right"
disparity_map_window_name = "disparity"
cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(right_window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(disparity_map_window_name, cv2.WINDOW_NORMAL)

left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.uint8)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.uint8)

window_size = 15
min_disp = 0
nDispFactor = 1  # adjust this
num_disp = 16 * nDispFactor - min_disp

stereo = cv2.StereoSGBM.create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=1 * 8 * 3 * window_size**2,
    disp12MaxDiff=-1,
    uniquenessRatio=11,
    speckleWindowSize=0,
    speckleRange=1,
    preFilterCap=0,
    mode=cv2.STEREO_SGBM_MODE_HH,
)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
disparity = cv2.normalize(
    disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
).astype(np.uint8)

# disparity = cv2.medianBlur(disparity, 5)

cv2.imshow(left_window_name, left_gray)
cv2.imshow(right_window_name, right_gray)
cv2.imshow(disparity_map_window_name, disparity)

cv2.waitKey(-1)
cv2.destroyAllWindows()
