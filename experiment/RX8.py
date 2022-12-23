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
    sampling_freq = CFloat(24144.0625, group='primary', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='primary', dsec='name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='')
    index = Any(group='primary', dsec='index of the device to connect to')
    stimulus = Any(group='primary', dsec='stimulus to play', context=False)


class RX8Device(Device):
    setting = RX8Setting()
    handle = Any
    thread = Instance(threading.Thread)

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor,os.path.join(expdir, self.setting.file)]],
                               connection=self.setting.connection)

        # create thread to monitoring hardware
        if not self.thread or not self.thread.isAlive():
            log.debug('creating thread...')
            self.thread = threading.Thread(target=self.thread_func, daemon=True)
            self.thread.start()

    def _configure(self, **kwargs):
        self.set_signal_and_speaker(**kwargs)
        if self.stimulus.__len__():
            self.handle.write('playbuflen', len(self.stimulus))

        log.debug('output channel changed to {}'.format(self.channel_nr))

    def _start(self):
        self.handle.trigger(1, proc=self.setting.processor)
        print(f"Running {self.setting.processor} ... ")

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.setting.processor}{self.setting.index} ...")
        self.handle.halt()

    #def thread_func(self):
        #while self.handle.read('playback', proc=f"{self.setting.processor}{self.setting.index}"):
            #pass
        #self.stop()
        #self.experiment._stop_trial = True

    def wait_to_finish_playing(self, tag="playback"):
        proc = self.setting.processor
        logging.info(f'Waiting for {tag} on {proc}.')
        while any(self.handle.read(tag, proc=proc)):
            time.sleep(0.01)
        logging.info('Done waiting.')

    def set_signal_and_speaker(self, data, speaker):
        self.setting.stimulus = data
        self.handle.write(tag=f"data{speaker}", value=self.setting.stimulus, procs=f"{self.setting.processor}{self.setting.index}")
        self.handle.write(tag=f"chan{speaker}", value=speaker, procs=f"{self.setting.processor}{self.setting.index}")
        print(f"Set signal to speaker {speaker}")



if __name__ == "__main__":
    import slab
    # initialize RX81 by setting index to 1 and RX82 by setting index to 2
    RX81 = RX8Device()
    RX81.setting.index = 1
    RX81.initialize()
    stimulus = slab.Sound.tone()
    chan = 23
    RX81.configure(data=stimulus, speaker=chan)
    RX81.start()
    RX81.wait_to_finish_playing()




