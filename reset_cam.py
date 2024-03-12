# Script to reset basler camera using pypylon
import pypylon.pylon as pylon

tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
print(f"Resetting device {devices[0].GetModelName()}...")
cam = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
cam.Open()  # Enable access to camera features
cam.UserSetSelector = "Default"  # Always good to start from power-on state
cam.UserSetLoad.Execute()
cam.StopGrabbing()
cam.Close()
