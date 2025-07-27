import cv2

_windowName = "stream"
frame = None

cap = cv2.VideoCapture(1) #built in webcam is '0'

if not cap.isOpened():
  print("Error no output")
  exit()

#['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 
# 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 
# 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 
# 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# (x, y) => Bottom Left coordinate of textbox
def drawFunc(frame, text, cv_x, cv_y):
  #sanitize xy
  if (cv_x < 0): 
    cv_x = 0
    print("x is negative :(")
  if (cv_y < 0):
    cv_y = 0
    print("y is negative :(")

  print("debug: {}".format(text))

  cv2.putText(frame, text, (cv_x, cv_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

def hsvCallback(event, x, y, flags, param):
  global frame

  if event == cv2.EVENT_MOUSEMOVE and frame is not None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[y, x]
    # print("{} {} {}".format(h, s, v))
    # print("x:{} y:{}".format(x, y))
    # print("string expected ->" + "HSV H({}) S({}) V({})".format(h, s, v))
    # print("mysterious cap property "+ str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-50))
    # exit()
    drawFunc(frame, "HSV H({}) S({}) V({})".format(h, s, v), 150, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-150)

def readVid():
  global frame
  ret, frame = cap.read()

cv2.namedWindow(_windowName)
cv2.setMouseCallback(_windowName, hsvCallback)

ret, frame = cap.read()
print("frame size: {} {}".format(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
  cv2.imshow(_windowName, frame)

  readVid() 

  if cv2.waitKey(1) == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()    