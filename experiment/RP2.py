from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Instance
import os
import logging
import numpy as np
import threading

log = logging.getLogger(__name__)

# TODO: RP2 initializes automatically upon calling it

class RP2Setting(DeviceSetting):  # this class contains important settings for the device and sits in self.setting
    sampling_freq = CFloat(48288.125, group='status', dsec='Sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='Max Buffer size')
    file = Str('MSL\\RCX\\button_rec.rcx', group='status', dsec='Name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='Name of the processor')
    connection = Str('GB', group='status', dsec='Connection of the device')
    index = Any(1, group='status', dsec='index of the device to connect to')


class RP2Device(Device):
    setting = RP2Setting()  # define setting for the device
    handle = Any()  # handle for TDT method execution like handle.write, handle.read, ...
    thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):  # this method is called upon self.initialize() execution
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()  # set processor to handle
        self.handle.initialize(proc_list=[self.setting.processor, self.setting.processor, self.setting.index, os.path.join(expdir, self.setting.file)],
                               connection=self.setting.connection)  # initialize processor.
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
        print(f"Halting {self.handle.procs.keys()} ...")
        self.handle.halt()

    def wait_for_button(self):  # stops the circuit as long as no button is being pressed
        print("Waiting for button press ...")
        while not self.handle.read(tag="response", proc=self.setting.processor):
            time.sleep(0.1)  # sleeps while the response tag in the rcx circuit does not yield 1

    def get_response(self):  # collects response, preferably called right after wait_for_button
        print("Acquiring response ... ")
        response = self.handle.read("response", self.setting.processor)
        return int(np.log2(response))  # because the response is stored in bit value, we need the base 2 log


if __name__ == "__main__":
    RP2 = RP2Device()
    RP2.configure()
    RP2.start()
    RP2.wait_for_button()
    response = RP2.get_response()
    RP2.pause()
    RP2.stop()
