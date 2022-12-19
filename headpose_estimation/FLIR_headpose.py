import cv2
from headpose.detect import PoseEstimator
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

est = PoseEstimator()
system = PySpin.System.GetInstance()
cams = system.GetCameras()

def initialize(cams=cams):
    # # initiate cameras
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cam.ExposureTime.SetValue(10000.0)
        try:
            cam.BeginAcquisition()
        except:
            print('camera already streaming')

def deinitialize(cams=cams):
    for cam in cams:
        if cam.IsInitialized():
            cam.EndAcquisition()
            cam.DeInit()
        del cam
    cams.Clear()
    system.ReleaseInstance()

def get_image(cam):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    #image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image.setflags(write=1)
    image_result.Release()
    return image

def change_res(image, resolution):
    data = Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), Image.ANTIALIAS)
    return np.asarray(image)

def pose_from_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    est.detect_landmarks(image, plot=True)  # plot the result of landmark detection
    roll, pitch, yaw = est.pose_from_image(image)  # estimate the head pose
    return roll, pitch, yaw
