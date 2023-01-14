from labplatform.config import get_config
from labplatform.core.Device import Device
from labplatform.core.Setting import DeviceSetting
from traits.api import CFloat, CInt, Str, Any, List
import threading
from labplatform.core import TDTblackbox as tdt
import logging
import os
import time

log = logging.getLogger(__name__)


class RX8Setting(DeviceSetting):  # this class contains settings for the device and sits in RX8.setting
    sampling_freq = CFloat(24144.0625, group='status', dsec='Sampling frequency of the device (Hz)')
    buffer_size_max = CInt(50000, group='status', dsec='Max buffer size')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='status', dsec='Name of the rcx file to load')
    processor = Str('RX8', group='status', dsec='Name of the processor')
    connection = Str('GB', group='status', dsec='Connection type of the processor')
    index = List([1, 2], group='status', dsec='Index of the device to connect to')
    signals = Any(group='primary', dsec='Stimulus to play', reinit=False, )
    speakers = Any(group="primary", dsex="Speaker to pick", reinit=False)
    device_name = Str("RX8", group="status", dsec="Name of the device")



class RX8Device(Device):

    setting = RX8Setting()  # device setting
    handle = Any()  # device handle

    # thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.Processors()
        self.handle.initialize(proc_list=[[f"{self.setting.processor}{self.setting.index[0]}",
                                           self.setting.processor,
                                          os.path.join(expdir, self.setting.file)],
                                          [f"{self.setting.processor}{self.setting.index[1]}",
                                           self.setting.processor,
                                           os.path.join(expdir, self.setting.file)]],
                               connection=self.setting.connection,
                               zbus=True)
        self.handle.write("playbuflen", self.setting.sampling_freq, procs=self.handle.procs)
        print(f"Initialized {self.setting.processor}.")

        # create thread to monitoring hardware
        #if not self.thread or not self.thread.is_alive():
            #log.debug('creating thread...')
            #self.thread = threading.Thread(target=self.thread_func, daemon=True)
            #self.thread.start()

    def _configure(self, **kwargs):
        for idx, spk in enumerate(self.setting.speakers):
            self.handle.write(tag=f"data{idx}", value=self.setting.signals[idx].data.flatten(), procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")
            self.handle.write(tag=f"chan{idx}", value=spk.channel_analog, procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")
            print(f"Set signal to chan {idx}")
        print(f"Configured {self.setting.processor}.")

    def _start(self):
        self.handle.trigger("zBusA", proc=self.handle)
        print(f"Running {self.setting.processor} ... ")
        self.wait_to_finish_playing()

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.setting.processor} ...")
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

    def run_normal_mode(self):
        pass

    def _deinitialize(self):
        pass


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
    sound_fp = pathlib.Path(sound_root + "\\tts-countries_resamp_24414\\")
    sound_list = slab.Precomputed(slab.Sound.read(sound_fp / file) for file in os.listdir(sound_fp))

    speakers = spk_array.pick_speakers([x for x in range(20, 27)])


    # initialize RX81 by setting index to 1 and RX82 by setting index to 2
    RX8 = RX8Device()
    RX8.initialize()

    signals = random.sample(sound_list, random.randint(2, 6))
    speakers = random.sample(speakers, len(signals))

    RX8.setting.signals = signals
    RX8.setting.speakers = speakers

    RX8.configure()
    RX8.start()
    RX8.pause()


