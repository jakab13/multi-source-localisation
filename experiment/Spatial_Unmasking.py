from abc import ABC
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
from traits.api import List, Str, Int, Dict, Float, Any
import random
import slab
import pathlib
import time
import numpy as np
import logging
import datetime

log = logging.getLogger(__name__)
config = slab.load_config(os.path.join(get_config("BASE_DIRECTORY"), "config", "spatmask_config.txt"))

# TODO: check out threading module
# TODO: data saving still sucks! log_trial() maybe? set_h5_atrributes()? data.data_spec?
# TODO: clear camera buffer every once in a while


class SpatialUnmaskingSetting(ExperimentSetting):

    experiment_name = Str('SpatMask', group='status', dsec='name of the experiment', noshow=True)
    n_conditions = Int(config.n_conditions, group="status", dsec="Number of masker speaker positions in the experiment")
    # trial_number = Int(1, group='primary', dsec='Number of trials in each condition', reinit=False)
    trial_duration = Float(config.trial_duration, group='status', dsec='Duration of one trial, (s)', reinit=False)


class SpatialUnmaskingExperiment(ExperimentLogic, ABC):

    setting = SpatialUnmaskingSetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(setting.n_conditions, n_reps=1, kind="random_permutation")
    devices = Dict()
    time_0 = Float()
    speakers = List()
    signals = List()
    warning_tone = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "warning\\warning_tone.wav"))
    stairs = Any()
    target_speaker = Any()
    selected_target_sounds = List()
    masker_speaker = Any()
    masker_sound = slab.Sound.pinknoise(duration=setting.trial_duration)

    def _initialize(self, **kwargs):
        self.load_speakers()
        self.load_signals()
        self.devices["RP2"] = RP2Device()
        self.devices["RX8"] = RX8Device()
        self.devices["ArUcoCam"] = ArUcoCam()
        self.devices["RX8"].handle.write("playbuflen",
                                         self.devices["RX8"].setting.sampling_freq*self.setting.trial_duration,
                                         procs=self.devices["RX8"].handle.procs)

    def _start(self, **kwargs):
        pass

    def _pause(self, **kwargs):
        pass

    def _stop(self, **kwargs):
        pass

    def setup_experiment(self, info=None):
        talker = random.randint(1, 108)
        self.masker_speaker = Any()
        self.selected_target_sounds = self.signals[talker * 5:(talker + 1) * 5]  # select numbers 1-5 for one talker
        self._tosave_para["sequence"] = self.sequence
        self._tosave_para["talker"] = talker
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.stairs = slab.Staircase(start_val=70, n_reversals=2, step_sizes=[4, 1])

    def _prepare_trial(self):
        if self.stairs.finished:
            self.stairs = slab.Staircase(start_val=config.start_val,
                                         n_reversals=config.n_reversals,
                                         step_sizes=config.step_sizes)
            self.sequence.__next__()
        self.masker_speaker = self.speakers[self.sequence.this_n]

    def _start_trial(self):
        self.time_0 = time.time()  # starting time of the trial
        level = self.stairs.__next__()
        log.warning(f"trial {self.setting.current_trial} dB level: {level}")
        self.check_headpose()
        target_sound_i = random.choice(range(len(self.selected_target_sounds)))
        target_sound = self.selected_target_sounds[target_sound_i]  # choose random number from sound_list
        target_sound.level = level
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
                                         self.masker_sound.data.flatten(),
                                         f"{self.masker_speaker.TDT_analog}{self.masker_speaker.TDT_idx_analog}")
        log.warning('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        # simulate response
        response = self.stairs.simulate_response(threshold=60)
        self.stairs.add_response(response)
        self.devices["RX8"].start()
        self.devices["RX8"].pause()
        # self.devices["RP2"].wait_for_button()
        reaction_time = int(round(time.time() - self.time_0, 3) * 1000)
        # response = self.devices["RP2"].get_response()
        solution = target_sound_i + 1
        is_correct = True if solution / response == 1 else False
        # self.stairs.add_response(1) if response/solution is True else self.stairs.add_response(0)
        self.stairs.plot()
        self.process_event({'trial_stop': 0})  # stops the trial

    def _stop_trial(self):
        is_correct = True if self.sequence.this_trial / self.devices["RP2"]._output_specs["response"] == 1 else False
        #self.data.write(key="solution", data=self.sequence.this_trial)
        #self.data.write(key="reaction_time", data=self.reaction_time)
        #self.data.write(key="is_correct", data=is_correct)
        #self.data.save()
        log.warning('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))

    def load_signals(self, target_sounds_type="tts-numbers_resamp_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, target_sounds_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(20, 27) if x != 23])
        self.speakers = speakers
        self.target_speaker = spk_array.pick_speakers(23)[0]

    def calibrate_camera(self, report=True):
        """
        Calibrates the cameras. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        log.warning("Calibrating camera")
        led = self.target_speaker  # central speaker
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        log.warning('Point towards led and press button to start calibration')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        self.devices["ArUcoCam"].start()
        offset = self.devices["ArUcoCam"].get_pose()
        self.devices["ArUcoCam"].offset = offset
        self.devices["ArUcoCam"].pause()
        for i, v in enumerate(self.devices["ArUcoCam"].offset):  # check for NoneType in offset
            if v is None:
                self.devices["ArUcoCam"].offset[i] = 0
                log.warning("Calibration unsuccessful, make sure markers can be detected by cameras!")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs=f"{led.TDT_digital}{led.TDT_idx_digital}")  # turn off LED
        self.devices["ArUcoCam"].calibrated = True
        if report:
            log.warning(f"Camera offset: {offset}")
        log.warning('Calibration complete!')

    def check_headpose(self):
        while True:
            self.devices["ArUcoCam"].configure()
            self.devices["ArUcoCam"].start()
            self.devices["ArUcoCam"].pause()
            try:
                if np.sqrt(np.mean(np.array(self.devices["ArUcoCam"].setting.pose) ** 2)) > 10:
                    log.warning("Subject is not looking straight ahead")
                    for idx in range(1, 5):  # clear all speakers before loading warning tone
                        self.devices["RX8"].handle.write(f"data{idx}", 0, procs=["RX81", "RX82"])
                        self.devices["RX8"].handle.write(f"chan{idx}", 99, procs=["RX81", "RX82"])
                    self.devices["RX8"].handle.write("data0", self.warning_tone.data.flatten(), procs="RX81")
                    self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
                    self.devices["RX8"].start()
                    self.devices["RX8"].pause()
                else:
                    break
            except TypeError:
                log.warning("Cannot detect markers, make sure cameras are set up correctly and arucomarkers can be detected.")
                continue


if __name__ == "__main__":

    log = logging.getLogger()
    log.setLevel(logging.WARNING)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
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
                          sex="M",
                          cohort="SpatMask")
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
    su = SpatialUnmaskingExperiment(subject=subject, experimenter=experimenter)
    su.calibrate_camera()
    su.start()
    # su.configure_traits()
