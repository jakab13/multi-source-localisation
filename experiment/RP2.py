from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from labplatform.core import TDTblackbox as tdt
import time
from traits.api import CFloat, CInt, Str, Any, Instance
import os
import logging

log = logging.getLogger(__name__)

# TODO: What about the device connection? GB OR USB?
class RP2_Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('RCX\\button_rec.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RP2', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    # stimulus = Any(group='primary', dsec='stimulus to play', reinit=False)

class RP2_Device(Device):
    setting = RP2_Setting()
    handle = Any

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor, connection=self.setting.connection,
                                               index=self.setting.index, path=os.path.join(expdir, self.setting.file))

        self._output_specs = {}

    def _configure(self, **kwargs):
        if self.setting.stimulus.__len__():
            # self.handle.WriteTagV('datain', 0, self.setting.stimulus)
            self.handle.write('playbuflen', len(self.setting.stimulus))

    def _start(self):
        print("Running RP2 ... ")

    def _pause(self):
        pass

    def _stop(self):
        self.handle.Halt()

    def wait_for_button(self):
        while not self.handle.read(tag="response", proc=self.settings.processor):
            time.sleep(0.1)




if __name__ == "__main__":
    RP2 = RP2()
    RP2.initialize()
    RP2.configure()
    RP2.start()
    RP2.wait_for_button()
    RP2.pause()
    RP2.stop()
    RP2.pause()