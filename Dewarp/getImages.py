import cv2

import os

capture = cv2.VideoCapture(1)

index = 0

capture.set(cv2.CAP_PROP_EXPOSURE, -9) # Set a manual exposure value

while capture.isOpened():
  success, img = capture.read()

  k = cv2.waitKey(5)

  if k == 27:
    break
  elif k == ord('s'):
    cv2.imwrite('imgs/img' + str(index) + '.png', img)

    print("Photo snapped!")

    index += 1

  cv2.imshow("stream", img)

capture.release()