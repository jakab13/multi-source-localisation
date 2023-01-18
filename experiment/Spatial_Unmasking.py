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

# TODO: check out threading module


class SpatialUnmaskingSetting(ExperimentSetting):

    experiment_name = Str('SpatMask', group='status', dsec='name of the experiment', noshow=True)
    conditions = List([20, 21, 22, 24, 25, 26], group="status", dsec="Number of simultaneous talkers in the experiment")
    trial_number = Int(20, group='primary', dsec='Number of trials in each condition', reinit=False)
    trial_duration = Float(1.0, group='primary', dsec='Duration of one trial, (s)', reinit=False)

    def _get_total_trial(self):
        return self.trial_number * len(self.conditions)


class SpatialUnmaskingExperiment(ExperimentLogic):

    setting = SpatialUnmaskingSetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=1, kind="random_permutation")
    devices = Dict()
    reaction_time = Int()
    time_0 = Float()
    speakers = List()
    signals = List()
    warning_tone = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "warning\\warning_tone.wav"))
    stairs = slab.Staircase(start_val=70, n_reversals=2, step_sizes=[4, 1], n_up=1, n_down=1)
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
        self.sequence.__next__()
        talker = random.randint(1, 108)
        self.masker_speaker = self.speakers[self.sequence.this_trial][0]
        self.selected_target_sounds = self.signals[talker * 5:(talker + 1) * 5]  # select numbers 1-5 for one talker
        self.devices["RX8"].handle.write("chan0", self.target_speaker.channel_analog, "RX81")
        self._tosave_para["sequence"] = self.sequence
        self._tosave_para["reaction_time"] = float
        self._tosave_para["solution"] = int
        self._tosave_para["is_correct"] = bool
        self._tosave_para["is_correct"] = bool
        self._tosave_para["elevation"] = bool
        self._tosave_para["threshold"] = bool
        self._tosave_para["intensities"] = bool
        self._tosave_para["reversal_points"] = bool
        self._tosave_para["reversal_intensities"] = bool

    def _prepare_trial(self):
        while True:
            self.devices["ArUcoCam"].configure()
            self.devices["ArUcoCam"].start()
            self.devices["ArUcoCam"].pause()
            if np.sqrt(np.mean(np.array(self.devices["ArUcoCam"].setting.pose) ** 2)) > 10:
                log.warning("Subject is not looking straight ahead")
                for idx in range(1, 5):  # clear all speakers before loading warning tone
                    self.devices["RX8"].handle.write(f"data{idx}", 0, procs=["RX81", "RX82"])
                    self.devices["RX8"].handle.write(f"chan{idx}", 99, procs=["RX81", "RX82"])
                self.devices["RX8"].handle.write("data0", self.warning_tone.data.flatten(), procs="RX81")
                self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
                    # self.devices["RX8"].handle.write(f"chan{idx}", 0, procs=["RX81", "RX82"])
                self.devices["RX8"].start()
                self.devices["RX8"].pause()
                # self.devices["RP2"].wait_for_button()
            else:
                break
        if self.stairs.finished:
            self.stairs = slab.Staircase(start_val=70, n_reversals=2, step_sizes=[4, 1], n_up=1, n_down=1)  # renew
            self.sequence.__next__()
            self.masker_speaker = self.speakers[self.sequence.this_trial][0]

    def _start_trial(self):
        for level in self.stairs:
            target_sound_i = random.choice(range(len(self.selected_target_sounds)))
            target_sound = self.selected_target_sounds[target_sound_i]  # choose random number from sound_list
            target_sound.level = level
            self.devices["RX8"].handle.write(tag=f"data0",
                                             value=target_sound.data.flatten(),
                                             procs=f"{self.target_speaker.TDT_analog}{self.target_speaker.TDT_idx_analog}")
            self.devices["RX8"].handle.write(tag=f"chan1",
                                             value=self.masker_speaker.channel_analog,
                                             procs=f"{self.masker_speaker.TDT_analog}{self.masker_speaker.TDT_idx_analog}")
            self.devices["RX8"].handle.write(tag=f"data1",
                                             value=self.masker_sound.data.flatten(),
                                             procs=f"{self.masker_speaker.TDT_analog}{self.masker_speaker.TDT_idx_analog}")
            self.time_0 = time.time()  # starting time of the trial
            log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
            #self.devices["RX8"].start()
            #self.devices["RP2"].wait_for_button()
            self.reaction_time = int(round(time.time() - self.time_0, 3) * 1000)
            #self.devices["RP2"].get_response()
            # simulate response
            response = self.stairs.simulate_response(threshold=3)
            self.stairs.add_response(response)
            self.stairs.plot()
            self.devices["RX8"].pause()
        self.process_event({'trial_stop': 0})

    def _stop_trial(self):
        is_correct = True if self.sequence.this_trial / self.devices["RP2"]._output_specs["response"] == 1 else False
        self.data.write(key="response", data=self.devices["RP2"]._output_specs["response"])
        self.data.write(key="solution", data=self.sequence.this_trial)
        self.data.write(key="reaction_time", data=self.reaction_time)
        self.data.write(key="is_correct", data=is_correct)
        self.data.save()
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))

    def load_signals(self, sound_type="tts-numbers_resamp_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = get_config(setting="BASE_DIRECTORY")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(20, 27) if x != 23])
        self.speakers = speakers
        self.target_speaker = spk_array.pick_speakers(23)[0]

    def pick_signals_this_trial(self, n_signals):
        signals_no_rep = list(x for x in self.signals if x not in self.signals_sample)
        self.signals_sample = random.sample(signals_no_rep, n_signals)

    def calibrate_camera(self, report=True):
        """
        Calibrates the cameras. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        log.info("Calibrating camera")
        led = self.speakers[3]  # central speaker
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        log.info('Point towards led and press button to start calibration')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        self.devices["ArUcoCam"].start()
        offset = self.devices["ArUcoCam"].get_pose()
        self.devices["ArUcoCam"].offset = offset
        self.devices["ArUcoCam"].pause()
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs=f"{led.TDT_digital}{led.TDT_idx_digital}")  # turn off LED
        self.devices["ArUcoCam"].calibrated = True
        if report:
            log.info(f"Camera offset: {offset}")
        log.info('Calibration complete!')


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
                          sex="M",
                          cohort="NumJudge")
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
        #test_subject.file_path
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
        sl.read_from_h5file()
        subject = sl.subjects[0]
    # subject.file_path
    experimenter = "Max"
    nj = NumerosityJudgementExperiment(subject=subject, experimenter=experimenter)
    # nj.calibrate_camera()
    # nj.start()
    # nj.configure_traits()
