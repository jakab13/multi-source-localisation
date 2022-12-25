from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Property
import os
import logging

log = logging.getLogger(__name__)


class RP2Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('MSL\\RCX\\button_rec.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')


class RP2Device(Device):
    setting = RP2Setting()
    handle = Any()

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor,os.path.join(expdir, self.setting.file)]],
                               connection=self.setting.connection)



        self._output_specs = {}

    def _configure(self, **kwargs):
        pass

    def _start(self):
        print(f"Running {self.setting.processor} ... ")

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.setting.processor} ...")
        self.handle.halt()

    def wait_for_button(self):
        while not self.handle.read(tag="response", proc=self.settings.processor):
            time.sleep(0.1)




if __name__ == "__main__":
    RP2 = RP2Device()
    RP2.configure()
    RP2.start()
    RP2.wait_for_button()
    RP2.pause()
    RP2.stop()
