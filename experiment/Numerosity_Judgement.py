from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RP2 import RP2Device
from experiment.RX8 import RX8Device
from experiment.Camera import ArUcoCam
from Speakers.speaker_config import SpeakerArray
import os
from traits.api import List, Str, Int, Dict, Float, Any, Bool
import random
import slab
import pathlib
import time
import numpy as np
import logging
import datetime

log = logging.getLogger(__name__)
config = slab.load_config(os.path.join(get_config("BASE_DIRECTORY"), "config", "numjudge_config.txt"))

#TODO: I think the experiment skips the last trial


class NumerosityJudgementSetting(ExperimentSetting):

    experiment_name = Str('NumJudge', group='status', dsec='name of the experiment', noshow=True)
    conditions = List(config.conditions, group="status", dsec="Number of simultaneous talkers in the experiment")
    trial_number = Int(config.trial_number, group='status', dsec='Number of trials in each condition')
    stim_duration = Float(config.trial_duration, group='status', dsec='Duration of each trial, (s)')
    setup = Str("FREEFIELD", group="status", dsec="Name of the experiment setup")

    def _get_total_trial(self):
        return self.trial_number * len(self.conditions)


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=setting.trial_number)
    results = Any()
    devices = Dict()
    speakers_sample = List()
    signals_sample = Dict()
    time_0 = Float()
    speakers = List()
    signals = Dict()
    off_center = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\400_tone.wav"))
    paradigm_start = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_start.wav"))
    paradigm_end = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_end.wav"))
    plane = Str("v")
    response = Any()
    solution = Any()
    rt = Any()
    is_correct = Bool()
    reversed_speech = Bool(False)

    def _devices_default(self):
        rp2 = RP2Device()
        rx8 = RX8Device()
        cam = ArUcoCam()
        return {"RP2": rp2,
                "RX8": rx8,
                "ArUcoCam": cam}

    def _initialize(self, **kwargs):
        self.devices["RX8"].handle.write("playbuflen",
                                         self.setting.stim_duration * self.devices["RX8"].setting.sampling_freq,
                                         procs=self.devices["RX8"].handle.procs)

    def _start(self, **kwargs):
        pass

    def _pause(self, **kwargs):
        # self.devices["RX8"].pause()
        pass

    def _stop(self, **kwargs):
        pass

    def setup_experiment(self, info=None):
        self.results.write(self.reversed_speech, "reversed_speech")
        self.results.write(self.plane, "plane")
        self.results.write(self.sequence, "sequence")
        self.results.write(np.ndarray.tolist(np.array(self.devices["ArUcoCam"].offset)), "offset")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.load_speakers()
        self.load_signals()
        self.devices["RX8"].handle.write("data0", self.paradigm_start.data.flatten(), procs="RX81")
        self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
        self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
        self.devices["RX8"].wait_to_finish_playing()
        # self.devices["RX8"].handle.write("playbuflen",
                                         # self.devices["RX8"].setting.sampling_freq*self.setting.stim_duration,
                                         # procs=self.devices["RX8"].handle.procs)
        time.sleep(1)

    def _prepare_trial(self):
        self.check_headpose()
        self.devices["RX8"].clear_buffers(n_buffers=1, proc="RX81")
        self.devices["RX8"].clear_channels(n_channels=1, proc="RX81")
        self.sequence.__next__()
        self.sequence.print_trial_info()
        self.solution = self.sequence.this_trial
        self.pick_speakers_this_trial(n_speakers=self.sequence.this_trial)
        self.pick_signals_this_trial(n_signals=self.sequence.this_trial)
        for idx, spk in enumerate(self.speakers_sample):
            sound = spk.apply_equalization(list(self.signals_sample.values())[idx], level_only=False)
            self.devices["RX8"].handle.write(tag=f"data{idx}",
                                             value=sound.data.flatten(),
                                             procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")
            self.devices["RX8"].handle.write(tag=f"chan{idx}",
                                             value=spk.channel_analog,
                                             procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")

    def _start_trial(self):
        self.time_0 = time.time()  # starting time of the trial
        log.info(f'trial {self.setting.current_trial}/{self.setting.total_trial-1} start: {time.time() - self.time_0}')
        self.devices["RX8"].start()
        self.devices["RP2"].start()
        self.devices["ArUcoCam"].start()
        self.devices["RP2"].wait_for_button()
        self.response = self.devices["RP2"].get_response()
        self.rt = int(round(time.time() - self.time_0, 3) * 1000)
        self.is_correct = True if self.response == self.solution else False

    def _stop_trial(self):
        log.info(f"trial {self.setting.current_trial}/{self.setting.total_trial-1} end: {time.time() - self.time_0}")
        for device in self.devices.keys():
            self.devices[device].pause()
        self.devices["RX8"].clear_buffers(n_buffers=self.sequence.this_trial,
                                          buffer_length=self.devices["RX8"].setting.sampling_freq, proc=["RX81", "RX82"])
        if self.setting.current_trial + 1 == self.setting.total_trial:
            self.devices["RX8"].handle.write(tag='bitmask',
                                             value=0,
                                             procs="RX81")  # turn off LED
            self.devices["RX8"].clear_channels(n_channels=5, proc=["RX81", "RX82"])
            self.devices["RX8"].handle.write("data0", self.paradigm_end.data.flatten(), procs="RX81")
            self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
            self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
            self.devices["RX8"].wait_to_finish_playing()
        self.results.write(self.response, "response")
        self.results.write(self.solution, "solution")
        self.results.write(self.rt, "rt")
        self.results.write(self.is_correct, "is_correct")
        self.results.write(np.ndarray.tolist(np.array(self.devices["ArUcoCam"].pose)), "headpose")
        self.results.write([x.id for x in self.speakers_sample], "speakers_sample")
        self.results.write([x for x in self.signals_sample.keys()], "signals_sample")

    def load_signals(self, sound_type="tts-countries_n13_resamp_48828"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        all_talkers = dict()
        talker_id_range = range(225, 377)
        for talker_id in talker_id_range:
            talker_sorted = list()
            for i, sound in enumerate(os.listdir(sound_fp)):
                if str(talker_id) in sound:
                    talker_sorted.append(sound_list[i])
            if talker_sorted.__len__():
                all_talkers[str(talker_id)] = talker_sorted
        self.signals = all_talkers

    def load_speakers(self, filename=f"{setting.setup}_speakers.txt", calibration=True):
        basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        if calibration:
            spk_array.load_calibration(file=os.path.join(get_config("CAL_ROOT"), f"{self.setting.setup}_calibration.pkl"))
        if self.plane == "v":
            speakers = spk_array.pick_speakers([x for x in range(20, 28)])
        elif self.plane == "h":
            speakers = spk_array.pick_speakers([2, 8, 15, 23, 31, 38, 44])
        else:
            log.info("Wrong plane, must be v or h. Unable to load speakers!")
            speakers = [None]
        self.speakers = speakers

    def pick_speakers_this_trial(self, n_speakers):
        # speakers_no_rep = list(x for x in self.speakers if x not in self.speakers_sample)
        self.speakers_sample = random.sample(self.speakers, n_speakers)

    def pick_signals_this_trial(self, n_signals):
        randsamp = random.sample(list(self.signals.keys()), n_signals)
        sample = dict()
        for samp in randsamp:
            sample[samp] = self.signals[samp][random.choice(range(n_signals))]
            self.signals_sample = sample

    def calibrate_camera(self, report=True):
        """
        Calibrates the cameras. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        log.info("Calibrating camera")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        log.info('Point towards led and press button to start calibration')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        self.devices["ArUcoCam"].retrieve()
        offset = self.devices["ArUcoCam"].pose
        self.devices["ArUcoCam"].offset = offset
        # self.devices["ArUcoCam"].pause()
        for i, v in enumerate(self.devices["ArUcoCam"].offset):  # check for NoneType in offset
            if v is None:
                self.devices["ArUcoCam"].offset[i] = 0
                log.info("Calibration unsuccessful, make sure markers can be detected by cameras!")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs=f"RX81")  # turn off LED
        self.devices["ArUcoCam"].calibrated = True
        if report:
            log.info(f"Camera offset: {offset}")
        log.info('Calibration complete!')

    def check_headpose(self):
        while True:
            #self.devices["ArUcoCam"].configure()
            self.devices["ArUcoCam"].retrieve()
            # self.devices["ArUcoCam"].pause()
            try:
                if np.sqrt(np.mean(np.array(self.devices["ArUcoCam"].pose) ** 2)) > 12.5:
                    log.info("Subject is not looking straight ahead")
                    self.devices["RX8"].clear_channels(n_channels=1, proc=["RX81", "RX82"])
                    self.devices["RX8"].handle.write("data0", self.off_center.data.flatten(), procs="RX81")
                    self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
                    #self.devices["RX8"].start()
                    #self.devices["RX8"].pause()
                    self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
                    self.devices["RX8"].wait_to_finish_playing()
                else:
                    break
            except TypeError:
                log.info("Cannot detect markers, make sure cameras are set up correctly and arucomarkers can be detected.")
                continue


if __name__ == "__main__":

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    # Create subject
    try:
        subject = Subject(name="Foo",
                          group="Pilot",
                          birth=datetime.date(1996, 11, 18),
                          species="Human",
                          sex="M")
        subject.data_path = os.path.join(get_config("DATA_ROOT"), "Foo_test.h5")
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
        #test_subject.file_path
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
        sl.read_from_h5file()
        subject = sl.subjects[0]
        subject.data_path = os.path.join(get_config("DATA_ROOT"), "Foo_test.h5")
    # subject.file_path
    experimenter = "Max"
    nj = NumerosityJudgementExperiment(subject=subject, experimenter=experimenter)
    nj.results = slab.ResultsFile(subject=subject.name)
    # nj.devices["RP2"].experiment = nj
    nj.calibrate_camera()
    nj.start()
    # nj.configure_traits()
