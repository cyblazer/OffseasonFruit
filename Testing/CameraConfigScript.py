import cv2

class configScript:
  def __init__(self):
    print(" ---- STARTING PROGRAM ---- ")
    print("Build information: " + cv2.getBuildInformation())
    cv2.setLogLevel(6)
    print("Current Log Level: " + str(cv2.getLogLevel()))
    self.connection = cv2.VideoCapture(1, cv2.CAP_ANY)

    settings = []
    #First things first find api backends
    # Get a list of all available video I/O backends
    available_backends = cv2.videoio_registry.getBackends()

    # print("Available OpenCV Video I/O Backends:")
    # for backend in available_backends:
    #     print(f"- {cv2.videoio_registry.getBackendName(backend)}")
    # Setting Log level to VERBOSE (6) also dumps this information

    #find default camera backend
    print("Resulting CV API Backend Selected:")
    print(self.connection.getBackendName())

    #Then just check general camera level firmware available settings
    exceptions = ["AUDIO", "1394", "GIGA", "GPHOTO2", "INTELPERC", "IOS", "ISO", "INTRINSIC", "OPENNI", "PVAPI", "XI_"]
    props = [i for i in dir(cv2) if i.startswith(('CAP_PROP_', 'CAP_MODE_')) and not any(e in i for e in exceptions)]    # print(props)
  
    for prop in props:
      getvalue = self.connection.get(getattr(cv2, prop))
      try:
        setvalue = self.connection.set(getattr(cv2, prop), 20)
      except Exception as e:
        print(e)
      
      if (getvalue != -1.0):
        print("GET:" +prop)
        print(getvalue)
      if (setvalue is not -1.0 or setvalue is not None):
        if (type(setvalue) == bool):
          print("SET:" +prop)
          print("THIS VALUE CAN BE SET BUT RETURNED: "+ str(setvalue))
        else:
          print("SET:" +prop)
          print(setvalue)


  #Things to get:
  #All possible resolutions
  #Whether default capture mode is bgr or rgb or yuy2 NOTE: YUY2 IS MORE EFFICIENT IN COMPRESSION, but MJPEG IS EVEN BETTER SUCKER
  #Whether there is camera level settings of:
  #Brightness, Contrast, Saturation, Hue, Gain, Exposure, White Balance, Sharpness, Zoom, Focus
  #Has -> IEEE1394 Settings
  #-> Auto Exposure
  #ISO_SPEED
  #Properties: Pan, Tilt, Roll
  #Auto Focus, Auto White Balance
  #Prop_Bitrate
  #CAP_PROP_READ_TIMEOUT_MSEC 

  def run(self):
    self.connection.read()

    self.connection.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  config = configScript()
  config.run()