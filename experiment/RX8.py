from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from traits.api import CFloat, CInt, Str, Any, Instance
import threading
from labplatform.core import TDTblackbox as tdt
import logging
import os
import time

log = logging.getLogger(__name__)

class RX8Setting(DeviceSetting):
    sampling_freq = CFloat(48288.125, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    rx8_file = Str('MSL\\RCX\\play_buf_msl.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = CInt(1, group='primary', dsec='index of the device to connect to')
    stimulus = Any(group='primary', dsec='stimulus to play', context=False)


class RX8Device(Device):
    setting = RX8Setting()
    handle = Any
    thread = Instance(threading.Thread)

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor="RX8", connection="GB", index=1, path=os.path.join(expdir, self.setting.rx8_file))
        self.handle.Run()

        # create thread to monitoring hardware
        if not self.thread or not self.thread.isAlive():
            log.debug('creating thread...')
            self.thread = threading.Thread(target=self.thread_func, daemon=True)
            self.thread.start()

    def _configure(self, **kargs):
        if self.stimulus.__len__():
            # self.handle.WriteTagV('datain', 0, self.stimulus)
            self.handle.SetTagVal('playbuflen', len(self.stimulus))

        self.handle.SetTagVal('channelnr', self.channel_nr)
        log.debug('output channel change to {}'.format(self.channel_nr))

    def _start(self):
        self.handle.
        print(f"Running {self.setting.processor} ... ")

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.setting.processor} ...")
        self.handle.Halt()

    def thread_func(self):
        while self.handle.GetTagVal('playback'):
            pass
        self.stop()
        self.experiment._trial_stop = True

    def wait_to_finish_playing(self, tag="playback"):
        proc = self.setting.processor
        logging.info(f'Waiting for {tag} on {proc}.')
        while any(self.handle.read(tag, proc=proc)):
            time.sleep(0.01)
        logging.info('Done waiting.')

if __name__ == "__main__":
    RX81 = RX8Device()
