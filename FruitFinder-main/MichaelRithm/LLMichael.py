import cv2
import numpy as np

asdf = []

# --- BLUE CONSTANTS ---
#lowHSV = np.array([0, 60, 30])
#highHSV = np.array([80, 255, 255])

# --- RED CONSTANTS ---
#lowHSV = np.array([115, 60, 80])
#highHSV = np.array([180, 255, 255])

# --- YELLOW CONSTANTS ---
lowHSV = np.array([90, 150, 100])
highHSV = np.array([110, 255, 255])

CONTOUR_THRESH_LOW = 500
CONTOUR_THRESH_HIGH = 4000

dim = (960, 720)

# image 1
# bounds = [(123, 453), (428, 321), (621, 421), (302, 664)]

# image 2
# bounds = [(0, 60), (500, 40), (620, 620), (-80, 560)]

# image 3
bounds = [(120, 60), (520, 60), (520, 520), (120, 520)]

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):

    #HSV processing
    matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [dim[0], 0], dim, [0, dim[1]]]))
    transformed_image = cv2.warpPerspective(image, matrix, dim)

    color_img = transformed_image.copy()
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    color_img = cv2.inRange(color_img, lowHSV, highHSV)
   
    #Edge detection
    edges = cv2.Canny(image,100,200)

    edges = cv2.warpPerspective(edges, matrix, dim)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgRed = cv2.dilate(edges, kernel, iterations=3)
    imgRed = cv2.morphologyEx(imgRed, cv2.MORPH_OPEN,
                           kernel,
                           iterations=1)
  
    contours = [contour for contour in contours if cv2.contourArea(contour) > CONTOUR_THRESH_LOW and cv2.contourArea(contour) < CONTOUR_THRESH_HIGH]

    satMask = transformed_image.copy()
    satMask = cv2.cvtColor(satMask, cv2.COLOR_RGB2HSV)
    satMask = cv2.inRange(satMask, np.array([0, 110, 0]), np.array([255, 255, 255]), satMask)

    contours.sort(key=cv2.contourArea)

    samples = []

    for contour in contours:
        mask = np.zeros(transformed_image.shape[:2], np.uint8)
        cv2.drawContours(mask, [np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))], -1, 255, cv2.FILLED)

        maskedImg = cv2.bitwise_and(color_img, color_img, mask=mask)

        vals = cv2.mean(maskedImg)

        area = cv2.contourArea(contour)

        satVal = vals[0] * transformed_image.shape[0] * transformed_image.shape[1] / area

        satVal /= 255.0

        # print(satVal)

        if (satVal > 0.5):
            samples.append(contour)

            # remove from satmask
            cv2.bitwise_not(mask, mask)
            satMask = cv2.bitwise_and(satMask, satMask, mask=mask)

    samples = list(map(lambda x: np.int8(cv2.boxPoints(cv2.minAreaRect(x))), samples))
    samples = list(filter(filterByRatio, samples))

    cv2.drawContours(transformed_image, samples, -1, color=(0, 255, 0), thickness=2)


    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    return largestContour, image, llpython

def filterByRatio(sample) -> bool:
    width = np.linalg.norm(sample[0] - sample[1])
    length = np.linalg.norm(sample[1] - sample[2])

    ratio = length / width
    ratioInverted = width / length

    area = length * width

    MIN_AREA = 100
    MAX_AREA = 4000
    
    if (area < MIN_AREA) or (area > MAX_AREA):
        return False

    MIN_RATIO = 1.3
    MAX_RATIO = 1000

    if ((ratio > MIN_RATIO) and (ratio < MAX_RATIO)) or ((ratioInverted > MIN_RATIO) and (ratioInverted < MAX_RATIO)):
        return True
    return False