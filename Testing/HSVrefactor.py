import cv2
import numpy as np

class HSVdropper:
  def __init__(self, camera_index=1):
    self.window_name = "stream"
    self.cap = cv2.VideoCapture(camera_index)

    self.frame = None

    if not self.cap.isOpened():
      raise RuntimeError("Error: Cannot open camera")
    
    #Runtime Values
    self.H = 0
    self.S = 0
    self.V = 0

    self.hsv = []

    cv2.namedWindow(self.window_name) #Has to be initialized already for callback to reference
    cv2.setMouseCallback(self.window_name, self.combined_callback)
        
  def tuple_mean(self):
    print(f"{[sum(x) / len(x) for x in zip(*self.hsv)]}")

  def tuple_max(self):
    print(f"{[max(x) for x in zip(*self.hsv)]}")
     
  def tuple_min(self):
    print(f"{[min(x) for x in zip(*self.hsv)]}")

  def draw_text(self, text, x, y):
    x = max(0, x)
    y = max(0, y)
    cv2.putText(self.frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  def combined_callback(self, event, x, y, flags, param):
     self.MOUSECLICK_callback(event, x, y, flags, param)
     self.MOUSEMOVE_callback(event, x, y, flags, param)

  def MOUSEMOVE_callback(self, event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and self.frame is not None:
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        self.H, self.S, self.V = h,s,v

  def MOUSECLICK_callback(self, event, x, y, flags, param):
     if event == cv2.EVENT_LBUTTONDBLCLK and self.frame is not None:
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        array = [h, s, v]
        self.hsv.append(array)
        self.tuple_max()
        self.tuple_min()

  def run(self):
      print(f"Frame size: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
      while True:
          ret, self.frame = self.cap.read()
          if not ret:
              print("Error: Failed to grab frame")
              break

          self.draw_text(f"HSV H({self.H}) S({self.S}) V({self.V})", 50, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 50)
          # Always copy frame before displaying (so text can be drawn on it)
          # self.display_frame = self.frame.copy()

          # Show display frame
          cv2.imshow(self.window_name, self.frame)

          if cv2.waitKey(1) == ord('q'):
              break

      self.cap.release()
      cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = HSVdropper(camera_index=1)
    viewer.run()