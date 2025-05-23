import cv2
import numpy as np

asdf = []

#dimensions of image
dim = (600, 800)

# #img = cv2.imread("img3.jpg")
img = cv2.imread("imgs/img1.png")
# # make the img actually visible
img = cv2.resize(img, dim, img)

cv2.imshow("raw image", img)

# #define camera matrix
# cameraMatrix = np.array([[775.79, 0, 400.898], 
#                 [0, 775.79, 300.79], 
#                 [0, 0, 1]])

# #define distortion coefficients
# dist = np.array([0.112507, -0.272067, 0, 0, 0.15775, 0, 0, 0])

# newcameramtx, a_roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, dim, 1, dim)
# img = cv2.undistort(img, cameraMatrix, dist, None, newcameramtx)

# cv2.imshow("undistorted", img)
# cv2.waitKey(0)
#Also we'd need to dewarp camera here

        #     <Calibration
        #     size="800 600"
        #     focalLength="775.79f, 775.79f"
        #     principalPoint="400.898f, 300.79f"
        #     distortionCoefficients="0.112507, -0.272067, 0, 0, 0.15775, 0, 0, 0"
        #     />

# --- BLUE CONSTANTS ---
#lowHSV = np.array([0, 60, 30])
#highHSV = np.array([80, 255, 255])

# --- RED CONSTANTS ---
#lowHSV = np.array([115, 60, 80])
#highHSV = np.array([180, 255, 2

# --- YELLOW CONSTANTS ---
lowHSV = np.array([50, 80, 80])
highHSV = np.array([120, 255, 255])



CONTOUR_THRESH_LOW = 500
CONTOUR_THRESH_HIGH = 4000

# image 1
# bounds = [(123, 453), (428, 321), (621, 421), (302, 664)]

# image 2
#bounds = [(0, 60), (500, 40), (620, 620), (-80, 560)]

# image 3
#bounds = [(120, 60), (520, 180), (520, 480), (120, 520)]

bounds = [(50, 60), (500, 60), (650, 440),  (0, 440)]

matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))

transformed_image = cv2.warpPerspective(img, matrix, dim)

colorImg = transformed_image.copy()
colorImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2HSV)
colorImg = cv2.inRange(colorImg, lowHSV, highHSV)

cv2.waitKey(0)

edges = cv2.Canny(img,100,200, apertureSize=3)

edges = cv2.warpPerspective(edges, matrix, dim)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgRed = cv2.dilate(edges, kernel, iterations=3)
imgRed = cv2.morphologyEx(imgRed, cv2.MORPH_OPEN,
                           kernel,
                           iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

satMask = transformed_image.copy()
satMask = cv2.cvtColor(satMask, cv2.COLOR_RGB2HSV)
satMask = cv2.inRange(satMask, np.array([0, 80, 0]), np.array([255, 255, 255]), satMask)

contours.sort(key=cv2.contourArea)

samples = []

for contour in contours:
    mask = np.zeros(transformed_image.shape[:2], np.uint8)
    cv2.drawContours(mask, [np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))], -1, 255, cv2.FILLED)



    maskedImg = cv2.bitwise_and(colorImg, colorImg, mask=mask)

    vals = cv2.mean(maskedImg)

    area = cv2.contourArea(contour)

    satVal = vals[0] * transformed_image.shape[0] * transformed_image.shape[1] / area

    satVal /= 255.0

    print(satVal)

    if (satVal > 1):
        samples.append(contour)
        

        # remove from satmask
        cv2.bitwise_not(mask, mask)
        satMask = cv2.bitwise_and(satMask, satMask, mask=mask)


def filterByRatio(sample) -> bool:
    width = np.linalg.norm(sample[0] - sample[1])
    length = np.linalg.norm(sample[1] - sample[2])

    ratio = length / width
    ratioInverted = width / length

    area = length * width

    MIN_AREA = 50
    MAX_AREA = 4000
    
    if (area < MIN_AREA) or (area > MAX_AREA):
        return False

    MIN_RATIO = 1.3
    MAX_RATIO = 1000

    if ((ratio > MIN_RATIO) and (ratio < MAX_RATIO)) or ((ratioInverted > MIN_RATIO) and (ratioInverted < MAX_RATIO)):
        return True
    return False

def centerPoint(sample) -> tuple:
    x, y = 0, 0
    for i in range (4):
        x += sample[i][0]
        y += sample[i][1]

    x /= 4
    y /= 4

    return {x, y}

samples = list(map(lambda x: np.intp(cv2.boxPoints(cv2.minAreaRect(x))), samples))
samples = list(filter(filterByRatio, samples))
#samples = np.sort(samples)


# print(samples)
cv2.drawContours(transformed_image, samples, -1, color=(0, 255, 0), thickness=2)

cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

samples = list(map(lambda sample: centerPoint(sample), samples))


# display image
cv2.imshow("main", img)
cv2.imshow("edges", edges)
cv2.imshow("sat", satMask)
cv2.imshow("done", transformed_image)
cv2.imshow("yellow", colorImg)

cv2.waitKey(0)