import cv2
import time

cap = cv2.VideoCapture(1)
window = cv2.namedWindow("window")

while True:
  ret, frame = cap.read()

  fps = cap.get(cv2.CAP_PROP_FPS)
  cv2.putText(frame, str(fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

  print(cap.get(cv2.CAP_PROP_EXPOSURE))
  print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  cv2.imshow("window", frame)

  if cv2.waitKey(1) == ord('q'):
    quit()

  time.sleep(0.05) 