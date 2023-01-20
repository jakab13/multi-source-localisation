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
    sampling_freq = CFloat(24144.0625, group='primary', dsec='Sampling frequency of the device (Hz)', reinit=False)
    buffer_size_max = CInt(50000, group='status', dsec='Max buffer size')
    file = Str('MSL\\RCX\\play_buf_msl.rcx', group='primary', dsec='Name of the rcx file to load', reinit=False)
    processor = Str('RM1', group='status', dsec='Name of the processor')
    connection = Str('USB', group='status', dsec='Connection type of the processor')
    index = CInt(1, group='primary', dsec='Index of the device to connect to', reinit=False)
    signals = Any(group='primary', dsec='Stimulus to play', reinit=False)
    speakers = Any(group="primary", dsex="Speaker to pick", reinit=False)


class RX81Device(Device):
    setting = RX81Setting()  # device setting
    handle = Any()  # device handle
    # thread = Instance(threading.Thread)  # important for threading

    def _initialize(self, **kwargs):
        expdir = get_config('DEVICE_ROOT')
        self.handle = tdt.initialize_processor(processor=self.setting.processor, connection=self.setting.connection,
                                               index=self.setting.index, path=os.path.join(expdir, self.setting.file))
        self.handle.SetTagVal("playbuflen", self.setting.sampling_freq)
        print(f"Initialized {self.setting.processor}.")

        # create thread to monitoring hardware
        #if not self.thread or not self.thread.is_alive():
            #log.debug('creating thread...')
            #self.thread = threading.Thread(target=self.thread_func, daemon=True)
            #self.thread.start()

    def _configure(self, **kwargs):
        for idx, spk in enumerate(self.setting.speakers):
            tdt.set_variable(var_name=f"data{idx}", value=self.setting.signals[idx].data.flatten(), proc=self.handle)
            self.handle.SetTagVal(f"chan{idx}", spk.channel_analog)
            print(f"Set signal to chan {idx}")
        print(f"Configured {self.setting.processor}")

    def _start(self):
        self.handle.SoftTrg(1)
        print(f"Running {self.setting.processor}... ")
        self.wait_to_finish_playing()

    def _pause(self):
        print(f"Pausing {self.setting.processor} ... ")
        pass

    def _stop(self):
        print(f"Halting {self.setting.processor} ...")
        self.handle.Halt()

    #def thread_func(self):
        #while self.handle.read('playback', proc=f"{self.setting.processor}{self.setting.index}"):
            #pass
        #self.stop()
        #self.experiment._stop_trial = True

    def wait_to_finish_playing(self, tag="playback"):
        logging.info(f'Waiting for {tag} on {self.setting.processor}.')
        while self.handle.GetTagVal(tag):
            time.sleep(0.01)
        print('Done waiting.')


if __name__ == "__main__":
    import slab
    from Speakers.speaker_config import SpeakerArray
    import random
    import pathlib

    # load whole speaker array
    basedir = get_config(setting="BASE_DIRECTORY")
    filename = "dome_speakers.txt"
    file = os.path.join(basedir, filename)
    spk_array = SpeakerArray(file=file)
    spk_array.load_speaker_table()

    # pick relevant speakers
    speakers = spk_array.pick_speakers([x for x in range(19, 24)])

    # get sound files
    sound_root = get_config(setting="SOUND_ROOT")
    sound_fp = pathlib.Path(sound_root + "\\tts-countries_resamp_24414\\")
    sound_list = random.sample(slab.Precomputed(slab.Sound.read(sound_fp / file) for file in os.listdir(sound_fp)), k=50)

    # define trial sequence
    seq = slab.Trialsequence(conditions=[2, 3, 4, 5], n_reps=5)

    # instantiate device
    RX81 = RX81Device()

    # initialize device
    RX81.initialize()

    # experiment logic:
    seq.__next__()
    signals_sample = random.sample(sound_list, seq.this_trial)
    speakers_sample = random.sample(speakers, len(signals_sample))

    # set device signals and speakers to the sample within each trial
    RX81.setting.signals = signals_sample
    RX81.setting.speakers = speakers_sample

    # configure device
    RX81.configure()
    # start device
    RX81.start()
    # pause device
    RX81.pause()

    # stop device
    RX81.stop()


