from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
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

log = logging.getLogger(__name__)
slab.set_default_samplerate(48828)


class SpatialUnmaskingSetting(ExperimentSetting):

    experiment_name = Str('SpatMask', group='status', dsec='name of the experiment', noshow=True)
    n_conditions = Int(4, group="status", dsec="Number of masker speaker positions in the experiment")
    trial_number = Int(1000, group='status', dsec='Number of trials in each condition')
    stim_duration = Float(1.0, group='status', dsec='Duration of one trial, (s)')
    setup = Str("FREEFIELD", group="status", dsec="Name of the experiment setup")


class SpatialUnmaskingExperiment_exmp(ExperimentLogic):

    setting = SpatialUnmaskingSetting()
    data = ExperimentData()
    results = Any()
    sequence = slab.Trialsequence(setting.n_conditions, n_reps=1, kind="random_permutation")
    devices = Dict()
    time_0 = Float()
    speakers = List()
    signals = Dict()
    off_center = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\400_tone.wav"))
    paradigm_start = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_start.wav"))
    staircase_end = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\staircase_end.wav"))
    paradigm_end = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_end.wav"))
    stairs = Any(slab.Staircase(start_val=70,
                                n_reversals=2,
                                step_sizes=[3, 1],
                                step_up_factor=1,
                                step_type="lin"))
    target_speaker = Any()
    selected_target_sounds = List()
    masker_speaker = Any()
    maskers = Dict()
    plane = Str("v")
    masker_sound = Any()  # slab.Sound.pinknoise(duration=setting.trial_duration, samplerate=24414)
    masker_sound_id = Any()
    talker = Any()
    potential_maskers = Any()
    threshold = Any()
    solution = Int()
    response = Int()
    is_correct = Bool()
    rt = Any()

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
        pass

    def _stop(self, **kwargs):
        self.stairs.close_plot()

    def setup_experiment(self, info=None):
        self.load_speakers()
        self.load_signals()
        self.load_maskers()
        self.talker = random.choice(["229", "318", "256", "307", "248", "245", "284", "268"])
        self.pick_masker_according_to_talker()
        self.selected_target_sounds = self.signals[self.talker]  # select numbers 1-9 for one talker
        self.results.write(self.sequence, "sequence")
        self.results.write(self.talker, "talker")
        self.results.write(np.ndarray.tolist(np.array(self.devices["ArUcoCam"].pose)), "offset")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.results.write(self.stairs, "stairs")
        # self._tosave_para["reaction_time"] = Any
        self.sequence.__next__()
        self.sequence.print_trial_info()
        self.devices["RX8"].handle.write("data0", self.paradigm_start.data.flatten(), procs="RX81")
        self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
        self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
        self.devices["RX8"].wait_to_finish_playing()
        # self.devices["RX8"].handle.write("playbuflen",
                                         # self.devices["RX8"].setting.sampling_freq*self.setting.stim_duration,
                                         # procs=self.devices["RX8"].handle.procs)
        time.sleep(1)

    def _prepare_trial(self):
        if self.stairs.finished:
            self.threshold = self.stairs.threshold()
            self.results.write(self.threshold, "threshold")
            self.stairs.close_plot()
            self.devices["RX8"].clear_channels(n_channels=5, proc=["RX81", "RX82"])
            self.stairs = slab.Staircase(start_val=70,
                                         n_reversals=2,
                                         step_sizes=[3, 1],
                                         step_up_factor=1,
                                         step_type="lin")
            # self._tosave_para["stairs"] = self.stairs
            self.devices["RX8"].handle.write("data0", self.staircase_end.data.flatten(), procs="RX81")
            self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
            self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
            self.devices["RX8"].wait_to_finish_playing()
            self.devices["RX8"].clear_buffers(n_buffers=1, proc="RX81")
            self.sequence.__next__()
            time.sleep(1.0)
        self.masker_speaker = self.speakers[self.sequence.this_trial - 1]
        self.masker_sound_id = random.sample(self.potential_maskers.keys(), 1)[0]
        self.masker_sound = self.potential_maskers[self.masker_sound_id]
        self.stairs.print_trial_info()
        log.info(f"Staircase number {self.sequence.this_n} out of {self.sequence.n_conditions}")

    def _start_trial(self):
        self.time_0 = time.time()  # starting time of the trial
        level = self.stairs.__next__()
        log.info(f"trial {self.setting.current_trial} dB level: {level}")
        self.check_headpose()
        # self.devices["RX8"].clear_buffer()
        target_sound_i = random.choice(range(len(self.selected_target_sounds)))
        target_sound = self.selected_target_sounds[target_sound_i]  # choose random number from sound_list
        target_sound.level = level
        target_sound = self.target_speaker.apply_equalization(target_sound, level_only=False)
        self.masker_sound = self.masker_speaker.apply_equalization(self.masker_sound, level_only=False)
        self.devices["RX8"].handle.write("chan0",
                                         self.target_speaker.channel_analog,
                                         f"{self.target_speaker.TDT_analog}{self.target_speaker.TDT_idx_analog}")
        self.devices["RX8"].handle.write("data0",
                                         target_sound.data.flatten(),
                                         f"{self.target_speaker.TDT_analog}{self.target_speaker.TDT_idx_analog}")
        self.devices["RX8"].handle.write("chan1",
                                         self.masker_speaker.channel_analog,
                                         f"{self.masker_speaker.TDT_analog}{self.masker_speaker.TDT_idx_analog}")
        self.devices["RX8"].handle.write("data1",
                                         self.masker_sound.data[:, 0].flatten(),
                                         f"{self.masker_speaker.TDT_analog}{self.masker_speaker.TDT_idx_analog}")
        log.info(f'trial {self.stairs.this_trial_n} start: {time.time() - self.time_0}')
        # simulate response
        # response = self.stairs.simulate_response(threshold=60)
        for device in self.devices.keys():
            self.devices[device].start()
        # self.devices["RX8"].pause()
        # self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
        # self.devices["RX8"].wait_to_finish_playing()
        self.devices["RP2"].wait_for_button()
        self.response = self.devices["RP2"].get_response()
        self.rt = int(round(time.time() - self.time_0, 3) * 1000)
        # self._tosave_para["reaction_time"] = reaction_time
        log.info(f"response: {self.response}")
        # self.stairs.add_response(response)
        solution_converter = {"0": 8,
                              "1": 5,
                              "2": 4,
                              "3": 9,
                              "4": 1,
                              "5": 6,
                              "6": 3,
                              "7": 2
                              }
        self.solution = solution_converter[str(target_sound_i)]
        log.info(f"solution: {self.solution}")
        # self._tosave_para["solution"] = solution
        self.is_correct = True if self.solution == self.response else False
        self.stairs.add_response(1) if self.response == self.solution else self.stairs.add_response(0)
        self.stairs.plot()
        # print(response)
        # print(solution)

    def _stop_trial(self):
        log.info(f"trial {self.setting.current_trial} end: {time.time() - self.time_0}")
        for device in self.devices.keys():
            self.devices[device].pause()
        if self.sequence.n_remaining == 0 and self.stairs.finished:
            self.setting.current_trial = self.setting.total_trial
        if self.setting.current_trial == self.setting.total_trial:
            self.devices["RX8"].handle.write(tag='bitmask',
                                             value=0,
                                             procs="RX81")  # turn off LED
            self.devices["RX8"].clear_channels(n_channels=5, proc=["RX81", "RX82"])
            self.devices["RX8"].handle.write("data0", self.paradigm_end.data.flatten(), procs="RX81")
            self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
            self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
            self.devices["RX8"].wait_to_finish_playing()
        self.results.write(np.ndarray.tolist(np.array(self.devices["ArUcoCam"].pose)), "headpose")
        self.results.write(self.response, "response")
        self.results.write(self.solution, "solution")
        self.results.write(self.rt, "rt")
        self.results.write(self.is_correct, "is_correct")

    def load_signals(self, target_sounds_type="tts-numbers_n13_resamp_48828"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, target_sounds_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        all_talkers = dict()
        talker_id_range = range(225, 377)
        for talker_id in talker_id_range:
            talker_sorted = list()
            for i, sound in enumerate(os.listdir(sound_fp)):
                if str(talker_id) in sound:
                    talker_sorted.append(sound_list[i])
            all_talkers[str(talker_id)] = talker_sorted
        self.signals = all_talkers

    def load_maskers(self, sound_type="babble-numbers-reversed-n13-shifted_resamp_48828"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        all_talkers = dict()
        talker_id_range = os.listdir(sound_fp)
        for talker_id in talker_id_range:
            talker_sorted = list()
            for i, sound in enumerate(os.listdir(sound_fp)):
                if str(talker_id) in sound:
                    talker_sorted.append(sound_list[i])
            if talker_sorted.__len__():
                all_talkers[str(talker_id)] = talker_sorted
        self.maskers = all_talkers

    def load_speakers(self, filename=f"{setting.setup}_speakers.txt", calibration=True):
        basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        if calibration:
            spk_array.load_calibration(file=os.path.join(get_config("CAL_ROOT"), f"{self.setting.setup}_calibration.pkl"))
        if self.plane == "v":
            speakers = spk_array.pick_speakers([x for x in range(20, 27) if x != 23])
        elif self.plane == "h":
            speakers = spk_array.pick_speakers([2, 8, 15, 31, 38, 44])
        else:
            log.info("Wrong plane, must be v or h. Unable to load speakers!")
            speakers = [None]
        self.speakers = speakers
        self.target_speaker = spk_array.pick_speakers(23)[0]

    def pick_masker_according_to_talker(self):
        potential_maskers = dict()
        for key, masker in self.maskers.items():
            if self.talker not in masker:
                potential_maskers[key] = masker
        self.potential_maskers = potential_maskers

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
                    self.devices["RX8"].clear_channels(n_channels=2, proc=["RX81", "RX82"])
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

class NumerosityJudgementSetting(ExperimentSetting):

    experiment_name = Str('NumJudge', group='status', dsec='name of the experiment', noshow=True)
    conditions = List([2, 3, 4], group="status", dsec="Number of simultaneous talkers in the experiment")
    trial_number = Int(3, group='status', dsec='Number of trials in each condition')
    stim_duration = Float(1.0, group='status', dsec='Duration of each trial, (s)')
    setup = Str("FREEFIELD", group="status", dsec="Name of the experiment setup")

    def _get_total_trial(self):
        return self.trial_number * len(self.conditions)


class NumerosityJudgementExperiment_exmp(ExperimentLogic):

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
            speakers = spk_array.pick_speakers([x for x in range(20, 27)])
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


class LocalizationAccuracySetting(ExperimentSetting):

    experiment_name = Str('LocaAccu_Test', group='status', dsec='name of the experiment', noshow=True)
    conditions = Int(6, group="status", dsec="Number of total speakers")
    trial_number = Int(3, group='status', dsec='Number of trials in each condition')
    stim_duration = Float(1.0, group='status', dsec='Duration of each trial, (s)')
    setup = Str("FREEFIELD", group="status", dsec="Name of the experiment setup")

    def _get_total_trial(self):
        return self.trial_number * self.conditions


class LocalizationAccuracySetting(ExperimentSetting):

    experiment_name = Str('LocaAccu', group='status', dsec='name of the experiment', noshow=True)
    conditions = Int(5, group="status", dsec="Number of total speakers")
    trial_number = Int(3, group='status', dsec='Number of trials in each condition')
    stim_duration = Float(1.0, group='status', dsec='Duration of each trial, (s)')
    setup = Str("FREEFIELD", group="status", dsec="Name of the experiment setup")

    def _get_total_trial(self):
        return self.trial_number * self.conditions


class LocalizationAccuracyExperiment_exmp(ExperimentLogic):

    setting = LocalizationAccuracySetting()
    data = ExperimentData()
    results = Any()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=setting.trial_number)
    devices = Dict()
    time_0 = Float()
    all_speakers = List()
    target = Any()
    signals = Any()
    off_center = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\400_tone.wav"))
    # off_center.level = 70
    paradigm_start = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_start.wav"))
    paradigm_end = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc_48828\\paradigm_end.wav"))
    # pose = Any()
    error = List()
    plane = Str("v")
    mode = Str()
    actual = Any()
    perceived = Any()
    accuracy = Any()
    rt = Any()
    solution = Any()

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
        pass

    def _stop(self, **kwargs):
        log.info(f"Final mean error - azimuth: {np.mean(np.array(self.error)[:, 0])}, elevation: {np.mean(np.array(self.error)[:, 1])}")

    def setup_experiment(self, info=None):
        self.results.write(self.sequence, "sequence")
        self.results.write(np.ndarray.tolist(np.array(self.devices["ArUcoCam"].pose)), "offset")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.load_speakers()
        if self.mode == "babble":
            self.load_babble()
        elif self.mode == "noise":
            self.load_pinknoise()
        else:
            log.error("Unable to load stimuli! Abort experiment ... ")
        self.devices["RX8"].handle.write("data0", self.paradigm_start.data.flatten(), procs="RX81")
        self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
        self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
        self.devices["RX8"].wait_to_finish_playing()
        self.devices["RX8"].handle.write("data0", np.zeros(len(self.paradigm_start.data.flatten())), procs="RX81")
        # self.devices["RX8"].handle.write("playbuflen",
                                         # self.devices["RX8"].setting.sampling_freq*self.setting.stim_duration,
                                         # procs=self.devices["RX8"].handle.procs)
        time.sleep(1)

    def _prepare_trial(self):
        self.devices["RX8"].clear_channels(n_channels=1, proc=["RX81", "RX82"])
        self.check_headpose()
        self.devices["RX8"].clear_channels(n_channels=1, proc="RX81")
        self.devices["RX8"].clear_buffers(n_buffers=1, proc=["RX81", "RX82"])
        self.sequence.__next__()
        self.sequence.print_trial_info()
        self.solution = self.sequence.this_trial - 1
        self.pick_speaker_this_trial(speaker_id=self.solution)
        signal = random.choice(self.signals)
        sound = self.target.apply_equalization(signal, level_only=False)
        self.devices["RX8"].handle.write(tag=f"data0",
                                         value=sound.data[:, 0].flatten(),
                                         procs=f"{self.target.TDT_analog}{self.target.TDT_idx_analog}")
        self.devices["RX8"].handle.write(tag=f"chan0",
                                         value=self.target.channel_analog,
                                         procs=f"{self.target.TDT_analog}{self.target.TDT_idx_analog}")

    def _start_trial(self):
        self.time_0 = time.time()  # starting time of the trial
        log.info(f'trial {self.setting.current_trial}/{self.setting.total_trial-1} start: {time.time() - self.time_0}')
        for device in self.devices.keys():
            self.devices[device].start()
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs="RX81")  # illuminate central speaker LED
        self.devices["RP2"].wait_for_button()
        self.rt = int(round(time.time() - self.time_0, 3) * 1000)
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.devices["ArUcoCam"].retrieve()
        # reaction_time = int(round(time.time() - self.time_0, 3) * 1000)
        self.actual = np.ndarray.tolist(np.array([self.target.azimuth, self.target.elevation]))
        self.perceived = np.ndarray.tolist(np.array(self.devices["ArUcoCam"].pose))
        self.accuracy = np.ndarray.tolist(np.abs(np.subtract(self.actual, self.perceived)))
        self.error.append(self.accuracy)
        time.sleep(1)
        self.devices["RP2"].wait_for_button()
        log.info(f"Trial {self.setting.current_trial} error - azimuth: {self.accuracy[0]}, elevation: {self.accuracy[1]}")
        # time.sleep(0.2)

    def _stop_trial(self):
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs="RX81")  # illuminate central speaker LED
        #accuracy = np.abs(np.subtract([self.target.azimuth, self.target.elevation], self.pose))
        #log.warning(f"Accuracy azi: {accuracy[0]}, ele: {accuracy[1]}")
        log.info(f"trial {self.setting.current_trial}/{self.setting.total_trial-1} end: {time.time() - self.time_0}")
        for device in self.devices.keys():
            self.devices[device].pause()
        if self.setting.current_trial + 1 == self.setting.total_trial:
            self.devices["RX8"].handle.write(tag='bitmask',
                                             value=0,
                                             procs="RX81")  # turn off LED
            self.devices["RX8"].clear_channels(n_channels=1, proc=["RX81", "RX82"])
            self.devices["RX8"].handle.write("data0", self.paradigm_end.data.flatten(), procs="RX81")
            self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
            self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
            self.devices["RX8"].wait_to_finish_playing()
        self.results.write(self.actual, "actual")
        self.results.write(self.perceived, "perceived")
        self.results.write(self.accuracy, "accuracy")
        self.results.write(self.rt, "rt")
        self.results.write(self.target.id, "target_spk_id")

    def load_babble(self, sound_type="babble-numbers-reversed-n13-shifted_resamp_48828"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def load_pinknoise(self):
        noise = slab.Sound.pinknoise(duration=0.025, samplerate=self.devices["RX8"].setting.sampling_freq)
        silence = slab.Sound.silence(duration=0.025, samplerate=self.devices["RX8"].setting.sampling_freq)
        end_silence = slab.Sound.silence(duration=0.775, samplerate=self.devices["RX8"].setting.sampling_freq)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise, end_silence)
        stim = stim.ramp(when='both', duration=0.01)
        self.signals = [stim]

    def load_speakers(self, filename=f"{setting.setup}_speakers.txt", calibration=True):
        basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        if calibration:
            spk_array.load_calibration(file=os.path.join(get_config("CAL_ROOT"), f"{self.setting.setup}_calibration.pkl"))
        if self.plane == "v":
            speakers = spk_array.pick_speakers([x for x in range(20, 27)])
        elif self.plane == "h":
            speakers = spk_array.pick_speakers([2, 8, 15, 23, 31, 38, 44])
        else:
            log.info("Wrong plane, must be v or h. Unable to load speakers!")
            speakers = [None]
        self.all_speakers = speakers

    def pick_speaker_this_trial(self, speaker_id):
        self.target = self.all_speakers[speaker_id]

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