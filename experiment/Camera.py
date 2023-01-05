from labplatform.config import get_config
from labplatform.core.Setting import DeviceSetting
from labplatform.core.Device import Device
from traits.api import Instance, Float, Any, Str, List, Tuple, Bool

from headpose.detect import PoseEstimator
from matplotlib import pyplot as plt
import numpy as np
from Speakers.speaker_config import SpeakerArray
import os
from simple_pyspin import Camera
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import ImageEnhance, Image
from labplatform.core import TDTblackbox as tdt

import logging
log = logging.getLogger(__name__)

# TODO: Try camera calibration

class FlirCamSetting(DeviceSetting):
    """
    Class for defining the camera settings. primary group parameters are supposed to be changed and sometimes
    reinitialized, whereas status group parameters are not.
    """
    name = Str("FireFly", group="status", dsec="")
    type = Str("image", group="primary", dsec="nature of the signal", reinit=False)
    dtype = Instance(np.uint8, group="status", dsec="data type")
    shape = Tuple(1080, 1440, group="primary", dsec="default shape of the image", reinit=False)
    sampling_freq = Float()
    root = Str(get_config(setting="BASE_DIRECTORY"), group="status", dsec="labplatform root directory")
    setup = Str("dome", group="status", dsec="experiment setup")
    file = Str(f"{setup.default_value}_speakers.txt")
    led = Any(group="primary", dsec="central speaker with attached led for initial camera calibration", reinit=False)
    calibrated = Bool(group="primary", dsec="tells whether camera is calibrated or not", reinit=False)


class FlirCam(Device):
    """
    Class for controlling the device. We set the setting class as this classes attribute (.setting). Also we have
    placeholders for the offset of the camera, the pose for each trial, the PoseEstimator (CNN), the cam itself and the
    tdt handle, mainly for calibration.
    """
    setting = FlirCamSetting()
    pose = List()
    offset = Float()
    est = PoseEstimator()
    cam = Any()
    handle = Any()

    def _initialize(self, **kwargs):
        """
        Initializes the device and sets the state to "created". Necessary before running the device.
        """
        print("Initializing cameras ... ")
        spks = SpeakerArray(file=os.path.join(self.setting.root, self.setting.file))  # initialize speakers.
        spks.load_speaker_table()  # load speakertable
        self.setting.led = spks.pick_speakers(23)[0]  # pick central speaker
        self.cam = Camera()  # initialize camera from simple pyspin
        self.cam.init()  # initialize camera

    def _configure(self, **kwargs):
        """
        Manipulates the setting parameters of the class and sets the state to "ready. Needs to be called in each trial.
        """
        print("Configuring camera settings ...")
        pass

    def _start(self, **kwargs):
        """
        Runs the device and sets the state to "running". Here lie all the important steps the camera has to do in each
        trial, such as acquiring and saving the head pose.
        """
        self.cam.start()  # start recording images into the camera buffer
        print("Acquiring image ... ")
        try:
            if self.setting.calibrated:
                img = self.cam.get_array()  # Get image as numpy array
                roll, pitch, yaw = self.est.pose_from_image(img)  # estimate the head pose from the np array
                if self.offset:
                    self.pose.append([roll-self.offset[0], pitch-self.offset[1], yaw-self.offset[2]])  # subtract offset
                else:
                    self.pose.append([roll, pitch, yaw])
                print("Acquired pose!")
        except ValueError:
            print("Could not recognize face, make sure that camera can see the face!")

    def _pause(self, **kwargs):
        """
        Pauses the camera and sets the state to "paused".
        """
        print("Pausing cameras ... ")
        self.cam.stop()

    def _stop(self):
        """
        Closes the camera and cleans up and sets the state to "stopped".
        """
        print("Halting cameras ... ")
        self.cam.close()  # close camera

    def calibrate(self):
        """
        Calibrates the camera. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele, azi 0°,
        then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        print("Calibrating camera ...")
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor,
                                           self.setting.processor,
                                           os.path.join(expdir, self.setting.file)]],
                                           connection=self.setting.connection)
        print(f"Initialized {self.setting.processor}{self.setting.index}.")
        self.handle.write(tag='bitmask',
                          value=self.setting.led_spk.digital_channel,
                          processors=self.setting.led_spk.digital_proc)  # illuminate LED
        try:
            roll, pitch, yaw = self.get_pose()  # get head pose
            self.offset = [roll, pitch, yaw]  # use head pose as offset
            self.handle.write(tag='bitmask', value=0, processors=self.setting.led_spk.digital_proc)  # turn off LED
            self.setting.calibrated = True
            print("Successfully calibrated camera!")
        except ValueError:
            print("Could not see the face, make sure that the camera is seeing the face!")

    def snapshot(self, cmap="gray"):
        """
        Args:
            cmap: matplotlib colormap

        Returns:

        """
        print("Acquiring snapshot ...")
        image = self.cam.get_array()  # get image as np array
        plt.imshow(image, cmap=cmap)  # show image
        plt.show()

    @staticmethod
    def brighten(image, factor):
        """
        Brightens the image by a factor.
        Args:
            image: image to be brightened, must be in the form of an array.
            factor: factor by which the image gets brightened.

        Returns:
            brightened image array.
        """
        img = Image.fromarray(image)  # transform array to PIL.Image object
        filter = ImageEnhance.Brightness(img)  # create brightness filter
        brightimg = filter.enhance(factor)  # enhance image
        array = np.asarray(brightimg)  # transform back to array
        return array

    @staticmethod
    def sharpen(image, factor):
        """

        Args:
            image: image to be sharpened, must be in the form of an array.
            factor: factor by which the image gets sharpened.

        Returns:
            sharpened image array.
        """
        img = Image.fromarray(image)  # transform array to PIL.Image object
        filter = ImageEnhance.Sharpness(img)  # create sharpness filter
        sharpimg = filter.enhance(factor)  # enhance image
        array = np.asarray(sharpimg)  # transform back to array
        return array


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # see if the logic works
    cam = FlirCam()
    cam.initialize()
    cam.configure()
    cam.start()
    cam.snapshot()
    cam.pause()

    # headpose estimation
    import time
    time.sleep(10)
    est = PoseEstimator()
    cam = Camera()
    cam.init()
    cam.start()
    img = cam.get_array()
    cam.stop()
    img = Image.fromarray(img)
    brightness = ImageEnhance.Brightness(img)
    bright = brightness.enhance(6.0)
    array = np.asarray(bright)
    roll, pitch, yaw = est.pose_from_image(array)  # estimate the head pose



