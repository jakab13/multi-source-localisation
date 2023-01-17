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
from traits.api import List, Str, Int, Dict, Float
import random
import slab
import pathlib
import time
import numpy as np
import logging
import datetime

log = logging.getLogger(__name__)


# TODO: test experiment data class (write/read data from file) and subject class
# TODO: fix checking for headpose (_before_start)
# TODO: check out threading module
# TODO: why is the experiment hacky?
# TODO: why does FooExp run automatically but not NumJudge?


class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = Str('NumJudge', group='status', dsec='name of the experiment', noshow=True)
    n_blocks = Int(1, group="status", dsec="Number of total blocks per session")
    conditions = List([2, 3, 4], group="status", dsec="Number of simultaneous talkers in the experiment")
    signal_log = List([999], group="primary", dsec="Logs of the signals used in previous trials", reinit=False)
    speaker_log = List([999], group="primary", dsec="Logs of the speakers used in previous trials", reinit=False)
    trial_number = Int(20, group='primary', dsec='Number of trials in each condition', reinit=False)
    trial_duration = Float(1.0, group='primary', dsec='Duration of each trial, (s)', reinit=False)

    def _get_total_trial(self):
        return self.trial_number * len(self.conditions)


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=setting.trial_number)
    devices = Dict()
    speakers_sample = List()
    signals_sample = List()
    response = Int()
    reaction_time = Int()
    time_0 = Float()
    speakers = List()
    signals = List()
    warning_tone = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "warning\\warning_tone.wav"))

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
        pass

    def start_experiment(self, info=None):
        pass

    def _prepare_trial(self):
        self.sequence.__next__()
        self.pick_speakers_this_trial(n_speakers=self.sequence.this_trial)
        self.pick_signals_this_trial(n_signals=self.sequence.this_trial)
        for idx, spk in enumerate(self.speakers_sample):
            self.devices["RX8"].handle.write(tag=f"data{idx}",
                                             value=self.signals_sample[idx].data.flatten(),
                                             procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")
            self.devices["RX8"].handle.write(tag=f"chan{idx}",
                                             value=spk.channel_analog,
                                             procs=f"{spk.TDT_analog}{spk.TDT_idx_analog}")

    def _before_start_validate(self):
        for device in self.devices.keys():
            if self.devices[device].state != "Ready":
                self.devices[device].change_state("Ready")

    def _before_start(self):
        while True:
            self.devices["ArUcoCam"].configure()
            self.devices["ArUcoCam"].start()
            self.devices["ArUcoCam"].pause()
            if np.sqrt(np.mean(np.array(self.devices["ArUcoCam"].setting.pose) ** 2)) > 10:
                log.warning("Subject is not looking straight ahead")
                self.devices["RX8"].handle.write("data0", self.warning_tone.data.flatten(), procs="RX81")
                self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
                for idx in range(1, 5):  # clear all speakers
                    self.devices["RX8"].handle.write(f"data{idx}", 0, procs=["RX81", "RX82"])
                    # self.devices["RX8"].handle.write(f"chan{idx}", 0, procs=["RX81", "RX82"])
                self.devices["RX8"].start()
                self.devices["RX8"].pause()
                # self.devices["RP2"].wait_for_button()
            else:
                break

    def _start_trial(self):
        self.time_0 = time.time()
        self.devices["RX8"].start()
        self.devices["RP2"].wait_for_button()
        self.response = self.devices["RP2"].get_response()
        self.reaction_time = int(round(time.time() - self.time_0, 3) * 1000)
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))

    def _stop_trial(self):
        is_correct = True if self.sequence.this_trial / self.response == 1 else False
        self.data.write(key="response", data=self.response, current_trial=self.setting.current_trial)
        self.data.write(key="solution", data=self.sequence.this_trial, current_trial=self.setting.current_trial)
        self.data.write(key="reaction_time", data=self.reaction_time, current_trial=self.setting.current_trial)
        self.data.write(key="is_correct", data=is_correct, current_trial=self.setting.current_trial)
        self.data.save()
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        self.time_0 = time.time()

    def load_signals(self, sound_type="tts-countries_resamp_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = get_config(setting="BASE_DIRECTORY")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(20, 27)])
        self.speakers = speakers

    def pick_speakers_this_trial(self, n_speakers):
        speakers_no_rep = list(x for x in self.speakers if x not in self.speakers_sample)
        self.speakers_sample = random.sample(speakers_no_rep, n_speakers)

    def pick_signals_this_trial(self, n_signals):
        signals_no_rep = list(x for x in self.signals if x not in self.signals_sample)
        self.signals_sample = random.sample(signals_no_rep, n_signals)

    def calibrate_camera(self, report=True, limit=0.5):
        """
        Calibrates the cameras. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        print("Calibrating camera ...")
        led = self.speakers[3]  # central speaker
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=led.channel_digital,
                                         procs=f"{led.TDT_digital}{led.TDT_idx_digital}")  # illuminate LED
        print('Point towards led and press button to start calibration...')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        _log = np.zeros(2)
        while True:  # wait in loop for sensor to stabilize
            self.devices["ArUcoCam"].start()
            pose = self.devices["ArUcoCam"].get_pose()
            self.devices["ArUcoCam"].pause()
            log = np.vstack((_log, pose))
            if log[-1, 0] is None or log[-1, 1] is None:
                print('No marker detected')
            # check if orientation is stable for at least 30 data points
            if len(log) > 30 and all(log[-20:, 0] != None) and all(log[-20:, 1] != None):
                diff = np.mean(np.abs(np.diff(log[-20:], axis=0)), axis=0).astype('float16')
                if report:
                    print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
                if diff[0] < limit and diff[1] < limit:  # limit in degree
                    break
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs=f"{led.TDT_digital}{led.TDT_idx_digital}")  # turn off LED
        pose_offset = np.around(np.mean(log[-20:].astype('float16'), axis=0), decimals=2)
        log.info('Calibration complete!')
        self.devices["ArUcoCam"].offset = pose_offset
        self.devices["ArUcoCam"].calibrated = True

    def run_normal_mode(self):
        pass


if __name__ == "__main__":

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    # Create subject
    try:
        subject = Subject(name="Foo",
                          group="Test",
                          birth=datetime.date(1996, 11, 18),
                          species="Human",
                          sex="M",
                          cohort="MSL")
        subject.data_path = os.path.join(get_config("DATA_ROOT"), subject.name)
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Foo_test.h5"))
        #test_subject.file_path
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Foo_Test.h5"))
        sl.read_from_h5file()
        subject = sl.subjects[0]
    # subject.file_path
    experimenter = "Max"
    nj = NumerosityJudgementExperiment(subject=subject, experimenter=experimenter)
    for trial in range(10):
        nj._before_start_validate()
        nj._before_start()
        nj.start()
        nj.pause_experiment()

    # nj.configure_traits()
