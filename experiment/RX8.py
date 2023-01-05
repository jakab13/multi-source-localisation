from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from traits.api import CFloat, CInt, Str, Any, Instance, Property
import threading
from labplatform.core import TDTblackbox as tdt
import logging
import os
import time

log = logging.getLogger(__name__)


class RX8Setting(DeviceSetting):
    sampling_freq = CFloat(24144.0625, group='status', dsec='sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='buffer size cannot be larger than this')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='status', dsec='name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='name of the processor')
    connection = Str('GB', group='status', dsec='connection type of the processor')
    index = Any(group='primary', dsec='index of the device to connect to', reinit=False)
    signals = Any(group='primary', dsec='stimulus to play', reinit=False)
    speakers = Any(group="primary", dsex="speaker to pick", reinit=False)

class RX8Device(Device):
    setting = RX8Setting()
    handle = Any()
    thread = Instance(threading.Thread)

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[self.setting.processor, self.setting.processor, os.path.join(expdir, self.setting.file)]],
                               connection=self.setting.connection)
        print(f"Initialized {self.setting.processor}{self.setting.index}.")

        # create thread to monitoring hardware
        #if not self.thread or not self.thread.is_alive():
            #log.debug('creating thread...')
            #self.thread = threading.Thread(target=self.thread_func, daemon=True)
            #self.thread.start()

    def _configure(self, **kwargs):
        for idx, spk in enumerate(self.setting.speakers):
            self.handle.write(tag=f"data{idx}", value=self.setting.signals[idx].data.flatten(), procs=self.handle.procs)
            self.handle.write(tag=f"chan{idx}", value=spk.channel_analog, procs=self.handle.procs)
            print(f"Set signal to chan {idx}")
        self.handle.write("playbuflen", self.setting.sampling_freq, procs=self.handle.procs)
        print(f"Configured {self.setting.processor}{self.setting.index}.")

    def _start(self):
        self.handle.trigger(trig=1, proc=self.handle.procs[self.setting.processor])
        print(f"Running {self.setting.processor}{self.setting.index} ... ")
        self.wait_to_finish_playing()

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
        print('Done waiting.')


if __name__ == "__main__":
    import slab
    from Speakers.speaker_config import SpeakerArray
    import random
    import pathlib

    basedir = get_config(setting="BASE_DIRECTORY")
    filename = "dome_speakers.txt"
    file = os.path.join(basedir, filename)
    spk_array = SpeakerArray(file=file)
    spk_array.load_speaker_table()
    sound_root = get_config(setting="SOUND_ROOT")
    sound_fp = pathlib.Path(sound_root + "\\tts-numbers_resamp\\")
    sound_list = slab.Precomputed(slab.Sound.read(sound_fp / file) for file in os.listdir(sound_fp))

    speakers = spk_array.pick_speakers([x for x in range(19, 24)])


    # initialize RX81 by setting index to 1 and RX82 by setting index to 2
    RX81 = RX8Device()
    RX81.setting.index = 1
    RX81.initialize()

    signals = random.sample(sound_list, 5)

    RX81.setting.signals = signals
    RX81.setting.speakers = speakers

    RX81.configure()
    RX81.start()
    RX81.pause()


