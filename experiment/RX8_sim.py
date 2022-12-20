from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Instance
import os

import logging

log = logging.getLogger(__name__)

class RX8_Setting(DeviceSetting):
    sampling_freq = CFloat(24144, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('RCX\\play_buf_msl.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RM1', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='name of the connection between processor and computer')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    stimulus = Any(group='primary', dsec='stimulus to play', reinit=False)

class RX8_Device(Device):
    setting = RX8_Setting()
    handle = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor, os.path.join(expdir, self.setting.file)]],
                                                zbus=False, connection="USB")

        self._output_specs = {}

    def _configure(self, **kargs):

        print("Configuring ...")

    def _start(self):
        # self.handle.SoftTrg(1)
        pass

    def _pause(self):
        pass

    def _stop(self):
        self.handle.halt()
        print(f"Halting {self.setting.processor} ...")

    def wait_to_finish_playing(self, tag="playback"):
        proc = self.setting.processor
        logging.info(f'Waiting for {tag} on {proc}.')
        while any(self.handle.read(tag, proc=proc)):
            time.sleep(0.01)
        logging.info('Done waiting.')


if __name__ == "__main__":
    # simulate RP2 behavior
    RX81 = RX8_Device()
    RX81.initialize()
    RX81.wait_to_finish_playing()
    RX81.stop()
