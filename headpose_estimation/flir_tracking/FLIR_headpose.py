import cv2
from headpose.detect import PoseEstimator
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import freefield

# TODO: calibrate headpose

est = PoseEstimator()
system = PySpin.System.GetInstance()
cams = system.GetCameras()

def init(cams=cams):
    # # initiate cameras
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cam.ExposureTime.SetValue(10000.0)
        try:
            cam.BeginAcquisition()
        except:
            print('camera already streaming')

def halt(cams=cams):
    for cam in cams:
        if cam.IsInitialized():
            cam.EndAcquisition()
            cam.DeInit()
        del cam
    cams.Clear()
    system.ReleaseInstance()

def get_image(cam, resolution=1.0):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    #image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    if resolution < 1.0:
        image = change_res(image, resolution)
    image.setflags(write=1)
    image_result.Release()
    return image

def change_res(image, resolution):
    data = Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), Image.ANTIALIAS)
    return np.asarray(image)

def headpose_from_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    est.detect_landmarks(image, plot=True)  # plot the result of landmark detection
    roll, pitch, yaw = est.pose_from_image(image)  # estimate the head pose
    return roll, pitch, yaw

def calibrate(self, world_coordinates, camera_coordinates, plot=True):
    [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED


if __name__ == "__main__":
    init(cams)
    images = dict()
    pitches = list()
    offset =
    for i, cam in enumerate(cams):
        image = get_image(cam, resolution=1.0)  # try lower resolution?
        images[str(i)] = cam
        _, pitch, _ = headpose_from_image(image)
        pitches.append(pitch)