import cv2
from headpose.detect import PoseEstimator
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import freefield
import PySpin
from Speakers.speaker_config import SpeakerArray
from experiment.RX8 import RX8Device

# TODO: calibrate headpose

est = PoseEstimator()
system = PySpin.System.GetInstance()
cams = system.GetCameras()
led_spk = SpeakerArray().pick_speakers(23)[0]
RX81 = RX8Device(index=1).initialize()


def init(cams):
    # # initiate cameras
    if cams.__len__():  # check if there is more than one cam
        for cam in cams:  # initialize cameras
            cam.Init()
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
            cam.ExposureTime.SetValue(100000.0)
            try:
                cam.BeginAcquisition()
            except:
                print('camera already streaming')
    else:
        cams.Init()
        cams.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cams.ExposureTime.SetValue(100000.0)
        try:
            cams.BeginAcquisition()
        except:
            print('camera already streaming')

def halt(cams):
    if cams.__len__():
        for cam in cams:
            if cam.IsInitialized():
                cam.EndAcquisition()
                cam.DeInit()
            del cam
        cams.Clear()
    else:
        if cams.IsInitialized():
            cams.EndAcquisition()
            cams.DeInit()
        del cams
        cams.Clear()

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

def headpose_from_image(image, plot=True):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    est.detect_landmarks(gray_img, plot=plot)  # plot the result of landmark detection
    roll, pitch, yaw = est.pose_from_image(gray_img)  # estimate the head pose
    return roll, pitch, yaw

def calibrate():
    RX8.write(tag='bitmask', value=led_spk.digital_channel,
                    processors=led_spk.digital_proc)  # illuminate LED
    roll, pitch, yaw = get_pose()
    offset = pitch
    return offset

def get_pose(resolution=1.0):
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    init(cams)
    image = get_image(cams, resolution=resolution)  # try lower resolution?
    roll, pitch, yaw = headpose_from_image(image)
    halt(cams)
    return roll, pitch, yaw

def test():
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    init(cams)
    image = get_image(cams[0], resolution=1.0)  # try lower resolution?
    roll, pitch, yaw = headpose_from_image(image)
    plt.imshow(image)
    plt.title(f"roll:{roll}, pitch:{pitch}, yaw:{yaw}")
    plt.show()
    halt(cams)
    print(roll, pitch, yaw)



if __name__ == "__main__":
    test()