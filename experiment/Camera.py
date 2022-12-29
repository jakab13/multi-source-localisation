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

import logging
log = logging.getLogger(__name__)

# TODO: fix calibration
# TODO: configure() method calls initialize() again but does not change state.

class FlirCamSetting(DeviceSetting):
    name = Str("FireFly", group="status", dsec="")
    type = Str("image", group="primary", dsec="nature of the signal")
    dtype = Instance(np.uint8)
    shape = Tuple(1080, 1440, group="primary", dsec="default shape of the image")
    sampling_freq = Float()
    root = Str(get_config(setting="BASE_DIRECTORY"), group="status", dsec="labplatform root directory")
    setup = Str("dome", group="status", dsec="experiment setup")
    file = Str(f"{setup.default_value}_speakers.txt")
    led = Any()
    _is_ready = Bool(False, group="primary", dsec="True if running, False if not running")


class FlirCam(Device):
    setting = FlirCamSetting()
    pose = List()
    offset = List()
    est = PoseEstimator()
    cam = Any()

    def _initialize(self, **kwargs):
        spks = SpeakerArray(file=os.path.join(self.setting.root, self.setting.file))
        spks.load_speaker_table()
        self.setting.led = spks.pick_speakers(23)[0]
        print("Initializing cameras ... ")
        self.cam = Camera()
        self.cam.init()

    def _configure(self, **kwargs):
        print("Configuring camera settings ...")
        pass

    def _start(self, **kwargs):
        self.cam.start()
        print("Acquiring image ... ")
        try:
            img = self.cam.get_array()  # Get 10 frames
            roll, pitch, yaw = self.est.pose_from_image(img)  # estimate the head pose
            self.pose.append([roll, pitch, yaw])
            print("Acquired pose")
        except ValueError:
            print("Could not recognize face, make sure that camera can see the face!")

    def _pause(self, **kwargs):
        print("Pausing cameras ... ")
        self.cam.stop()

    def _stop(self):
        self.cam.close()
        print("Halting cameras ... ")

    def calibrate(self, **kwargs):
        kwargs.get("RX81").write(tag='bitmask', value=self.setting.led_spk.digital_channel,
                  processors=self.setting.led_spk.digital_proc)  # illuminate LED
        roll, pitch, yaw = self.get_pose()
        offset = pitch
        kwargs.get("RX81").write(tag='bitmask', value=0, processors=self.setting.led_spk.digital_proc)  # turn off LED

    def snapshot(self, cmap="gray"):
        image = self.cam.get_array()  # Get 10 frames
        plt.imshow(image, cmap=cmap)
        plt.show()


if __name__ == "__main__":

    cam = FlirCam()
    cam.configure(_is_ready=True)
    cam.start()
