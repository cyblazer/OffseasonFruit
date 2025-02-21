import cv2

cameraIndex = 0
camera = cv2.VideoCapture(cameraIndex)

if not camera.isOpened():
  raise IOError("Cannot find camera")

ret, image = camera.read()
if not(ret):
  print("cry")
else:
  cv2.imshow("frame", image)
  cv2.waitKey(0)