from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import Float, Int, Str, Any
import os
import logging
import numpy as np
import threading

log = logging.getLogger(__name__)

# TODO: RP2 initializes automatically upon calling it


class RP2Setting(DeviceSetting):  # this class contains important settings for the device and sits in self.setting
    sampling_freq = Float(48288.125, group='primary', dsec='Sampling frequency of the device (Hz)', reinit=False)
    buffer_size_max = Int(50000, group='primary', dsec='Max Buffer size', reinit=False)
    file = Str('MSL\\RCX\\button_rec.rcx', group='primary', dsec='Name of the rcx file to load', reinit=False)
    processor = Str('RM1', group='primary', dsec='Name of the processor', reinit=False)
    connection = Str('USB', group='primary', dsec='Connection of the device', reinit=False)
    index = Any(1, group='primary', dsec='index of the device to connect to', reinit=False)


class RP2Device(Device):

    setting = RP2Setting()
    handle = Any()  # handle for TDT method execution like handle.write, handle.read, ...
    # thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):  # this method is called upon self.initialize() execution
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor, connection=self.setting.connection,
                                               index=self.setting.index, path=os.path.join(expdir, self.setting.file))
        print(f"Initialized {self.setting.processor}")

    def _configure(self, **kwargs):  # device needs to be configured before each trial. Sets state to "ready".
        print(f"Configuring {self.setting.processor} ... ")
        pass

    def _start(self):  # starts device
        print(f"Running {self.setting.processor} ... ")

    def _pause(self):  # needs to be called after each trial
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):  # stops the device. Initialization necessary when this method is called
        print(f"Halting {self.setting.processor} ...")
        self.handle.Halt()

    def wait_for_button(self):  # stops the circuit as long as no button is being pressed
        print("Waiting for button press ...")
        while not self.handle.GetTagVal("response"):
            time.sleep(0.1)  # sleeps while the response tag in the rcx circuit does not yield 1

    def get_response(self):  # collects response, preferably called right after wait_for_button
        print("Acquiring button response ... ")
        response = self.handle.GetTagVal("response")
        return int(np.log2(response))  # because the response is stored in bit value, we need the base 2 log

    def run_normal_mode(self):
        pass

    def _deinitialize(self):
        pass


if __name__ == "__main__":
    RP2 = RP2Device()
    RP2.initialize()
    RP2.configure()
    RP2.start()
    RP2.wait_for_button()
    response = RP2.get_response()
    RP2.pause()
    RP2.stop()
