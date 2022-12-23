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
    processor = Str('RM1', group='status', dsec='name of the processor')
    connection = Str('USB', group='status', dsec='')
    index = Any(group='primary', dsec='index of the device to connect to')
    data = Any(group='primary', dsec='stimulus to play', context=False)
    speaker = Any(group="primary", dsex="speaker to pick")

class RX8Device(Device):
    setting = RX8Setting()
    handle = Any
    thread = Instance(threading.Thread)

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor, os.path.join(expdir, self.setting.file)]],
                               connection=self.setting.connection)

        # create thread to monitoring hardware
        #if not self.thread or not self.thread.isAlive():
            #log.debug('creating thread...')
            #self.thread = threading.Thread(target=self.thread_func, daemon=True)
            #self.thread.start()

    def _configure(self, **kwargs):
        self.set_signal_and_speaker(kwargs)
        self.handle.write('playbuflen', len(self.setting.data))

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

    def wait_to_finish_playing(self, proc="all", tag="playback"):
        if proc == "all":
            proc = list(self.handle.procs.keys())
        elif isinstance(proc, str):
            proc = [proc]
        logging.info(f'Waiting for {tag} on {proc}.')
        while any(self.handle.read(tag, proc=p) for p in proc):
            time.sleep(0.01)
        logging.info('Done waiting.')

    def set_signal_and_speaker(self, data, speaker):
        self.setting.data = data
        for idx, spk in enumerate(speaker):
            self.handle.write(tag=f"data{idx}", value=self.setting.data, procs=self.setting.processor)
            self.handle.write(tag=f"chan{idx}", value=spk.channel_analog, procs=self.setting.processor)
            print(f"Set signal to chan tag {idx}")



if __name__ == "__main__":
    import slab
    from Speakers.speaker_config import SpeakerArray

    basedir = get_config(setting="BASE_DIRECTORY")
    filename = "dome_speakers.txt"
    file = os.path.join(basedir, filename)
    spk_array = SpeakerArray(file=file)
    spk_array.load_speaker_table()

    # initialize RX81 by setting index to 1 and RX82 by setting index to 2
    RX81 = RX8Device()
    RX81.initialize()
    data = slab.Sound.tone().data
    chan = spk_array.pick_speakers(list(x for x in range(19, 24)))
    RX81.handle.write(tag="data0", value=chan[0].channel_analog, procs="RM1")
    RX81.set_signal_and_speaker(data=data, speaker=chan)
    RX81.configure(data=data, speaker=chan)
    RX81.start()
    RX81.wait_to_finish_playing()
    RX81.stop()




