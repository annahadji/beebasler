# Script to set basler camera using pypylon with particular parameters
import pypylon.pylon as pylon

tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
print(f"Setting up device {devices[0].GetModelName()}...")
cam = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
cam.Open()  # Enable access to camera features
cam.UserSetSelector = "Default"  # Always good to start from power-on state
cam.UserSetLoad.Execute()

cam.MaxNumBuffer = 8000
cam.Height = 1098
cam.Width = 1524
cam.OffsetY = 2720
cam.OffsetX = 1172
cam.Gain = 30
cam.BslColorSpace = "Off"
cam.PixelFormat = "Mono8"
cam.BalanceWhiteAuto = "Off"
cam.ExposureTime = 15000

cam.StopGrabbing()
cam.Close()
