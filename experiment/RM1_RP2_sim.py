from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Instance
import os
import numpy as np

import logging

log = logging.getLogger(__name__)

class RP2_Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('RCX\\button_rec.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RM1', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='name of the connection between processor and computer')
    index = CInt(1, group='primary', dsec='index of the device to connect to')

class RP2_Device(Device):
    setting = RP2_Setting()
    handle = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor, os.path.join(expdir, self.setting.file)]],
                                                zbus=False, connection="USB")
        self._output_specs = {}

    def _configure(self, **kargs):
        # if self.setting.stimulus.__len__():
            # self.handle.WriteTagV('datain', 0, self.setting.stimulus)
            # self.handle.SetTagVal('playbuflen', len(self.setting.stimulus))
        print("Configuring ...")

    def _start(self):
        # self.handle.SoftTrg(1)
        pass

    def _pause(self):
        pass

    def _stop(self):
        self.handle.halt()
        print(f"Halting {self.setting.processor} ...")

    def wait_for_button(self):
        while not self.handle.read(tag="response", proc=self.setting.processor):
            time.sleep(0.1)

if __name__ == "__main__":
    # simulate RP2 behavior
    RP2 = RP2_Device()
    RP2.initialize()
    RP2.wait_for_button()
    digin = RP2.handle.read(tag="response", proc=RP2.setting.processor)  # digital button press input
    response = int(np.log2(digin))  # base-2 logarithm of digital input
    RP2.stop()