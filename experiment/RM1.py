from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Instance
import os

import logging

log = logging.getLogger(__name__)

class RM1_Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    rx8_file = Str('RCX\\play_buf_msl.rcx', group='primary', dsec='name of the rcx file to load')
    rp2_file = Str('RCX\\button_rec.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RM1', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='name of the connection between processor and computer')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    stimulus = Any(group='primary', dsec='stimulus to play', reinit=False)

class RM1_Device(Device):
    setting = RM1_Setting()
    handle = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor, os.path.join(expdir, self.setting.rp2_file)]],
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

    def wait_to_finish_playing(self):
        pass

    def wait_for_button(self):
        while not self.handle.read(tag="response", proc=self.setting.processor):
            time.sleep(0.1)

if __name__ == "__main__":
    # simulate RP2 behavior
    RM1 = RM1_Device()
    RM1.initialize()
    RM1.stop()
    RM1.wait_for_button()