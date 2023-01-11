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


class RX81Setting(DeviceSetting):  # this class contains settings for the device and sits in RX8.setting
    sampling_freq = CFloat(24144.0625, group='status', dsec='Sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='Max buffer size')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='status', dsec='Name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='Name of the processor')
    connection = Str('GB', group='status', dsec='Connection type of the processor')
    index = CInt(1, group='status', dsec='Index of the device to connect to')
    signals = Any(group='primary', dsec='Stimulus to play', reinit=False)
    speakers = Any(group="primary", dsex="Speaker to pick", reinit=False)


class RX81Device(Device):
    setting = RX81Setting()  # device setting
    handle = Any()  # device handle
    thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[f"{self.setting.processor}{self.setting.index}", self.setting.processor, self.setting.index, os.path.join(expdir, self.setting.file)],
                               connection=self.setting.connection)
        self.handle.write("playbuflen", self.setting.sampling_freq, procs=self.handle.procs)
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
        print(f"Configured {self.setting.processor}{self.setting.index}.")

    def _start(self):
        self.handle.trigger(trig=1, proc=self.handle.procs[self.setting.processor])
        print(f"Running {self.setting.processor}{self.setting.index} ... ")
        self.wait_to_finish_playing()

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.handle.procs.keys()} ...")
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

class RX82Setting(DeviceSetting):  # this class contains settings for the device and sits in RX8.setting
    sampling_freq = CFloat(24144.0625, group='status', dsec='Sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='Max buffer size')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='status', dsec='Name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='Name of the processor')
    connection = Str('GB', group='status', dsec='Connection type of the processor')
    index = CInt(2, group='status', dsec='Index of the device to connect to', reinit=False)
    signals = Any(group='primary', dsec='Stimulus to play', reinit=False)
    speakers = Any(group="primary", dsex="Speaker to pick", reinit=False)


class RX82Device(Device):
    setting = RX82Setting()  # device setting
    handle = Any()  # device handle
    thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[f"{self.setting.processor}{self.setting.index}", self.setting.processor, self.setting.index, os.path.join(expdir, self.setting.file)],
                               connection=self.setting.connection)
        self.handle.write("playbuflen", self.setting.sampling_freq, procs=self.handle.procs)
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
        print(f"Configured {self.setting.processor}{self.setting.index}.")

    def _start(self):
        self.handle.trigger(trig=1, proc=self.handle.procs[self.setting.processor])
        print(f"Running {self.setting.processor}{self.setting.index} ... ")
        self.wait_to_finish_playing()

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.handle.procs.keys()} ...")
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
    sound_fp = pathlib.Path(sound_root + "\\tts-numbers_resamp_24414\\")
    sound_list = slab.Precomputed(slab.Sound.read(sound_fp / file) for file in os.listdir(sound_fp))

    speakers = spk_array.pick_speakers([x for x in range(19, 24)])


    # initialize RX81 by setting index to 1 and RX82 by setting index to 2
    RX81 = RX81Device()
    RX81.initialize()
    RX81.stop()

    RX82 = RX82Device()
    RX82.initialize()
    RX82.stop()

    signals = random.sample(sound_list, 5)

    RX81.setting.signals = signals
    RX81.setting.speakers = speakers

    RX81.configure()
    RX81.start()
    RX81.pause()


