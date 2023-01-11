from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RP2 import RP2Device
from experiment.RX8 import RX81Device, RX82Device
# from experiment.Camera import ArUcoCam
from Speakers.speaker_config import SpeakerArray
import os
from traits.api import List, CInt, Str, Int, Dict
import random
import slab
import pathlib
import time
import numpy as np

# TODO: stimuli names do not include gender --> sort stimuli by gender
# TODO: test experiment data class (write/read data from file)
# TODO: check signal and speaker log before trial
# TODO: Implement ArUcoCam

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = Str('Numerosity Judgement', group='primary', dsec='name of the experiment', noshow=True)
    speakers = List(group="primary", dsec="List of speakers", reinit=False)
    signals = List(group="primary", dsec="Set to choose stimuli from", reinit=False)
    n_blocks = CInt(1, group="status", dsec="Number of total blocks per session")
    n_trials = CInt(20, group="status", dsec="Number of total trials per block")
    conditions = List([2, 3, 4, 5], group="status", dsec="Number of simultaneous talkers in the experiment")
    signal_log = List([9999], group="primary", dsec="Logs of the signals used in previous trials", reinit=False)
    speaker_log = List([999], group="primary", dsec="Logs of the speakers used in previous trials", reinit=False)


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=setting.n_trials)
    devices = Dict()
    speakers_sample = List()
    signals_sample = List()
    response = Int()
    reaction_time = Int()
    time_0 = time.time()


    def _initialize(self, **kwargs):
        self.devices["RP2"] = RP2Device()
        self.devices["RX81"] = RX81Device()
        #self.devices["RX81"].initialize()
        #self.devices["RX82"] = RX82Device()
        #self.devices["RX82"].initialize()
        self.load_speakers()
        self.load_signals()
        # self.devices["ArUcoCam"] = ArUcoCam()

    def _configure(self, **kwargs):
        self.devices["RX81"].setting.signals = self.signals_sample
        self.devices["RX81"].setting.speakers = self.speakers_sample
        #for device in self.devices.keys():
            #self.devices[device].configure()

    def _start(self, **kwargs):
        pass

    def _pause(self, **kwargs):
        pass

    def _stop(self, **kwargs):
        pass

    def _prepare_trial(self, **kwargs):
        pass

    def _start_trial(self):
        pass

    def _stop_trial(self):
        current_trial = self.sequence.this_n
        is_correct = True if self.sequence.this_trialtrial / self.response == 1 else False
        self.data.write(key="response", data=self.response, current_trial=current_trial)
        self.data.write(key="solution", data=self.sequence.this_trial, current_trial=current_trial)
        self.data.write(key="reaction_time", data=self.reaction_time, current_trial=current_trial)
        self.data.write(key="is_correct", data=is_correct, current_trial=current_trial)
        self.data.save()
        print(f"Trial {current_trial} end {time.time()-self.time_0}")
        self.time_0 = time.time()

    def setup_experiment(self, info=None):
        self.sequence.__next__()
        self.pick_speakers_this_trial(n_speakers=self.sequence.this_trial)
        self.pick_signals_this_trial(n_signals=self.sequence.this_trial)
        print("Set up experiment!")

    def configure_experiment(self):
        print("Configured experiment!")

    def start_experiment(self, info=None):
        start_time = time.time()
        self.devices["RP2"].wait_for_button()
        self.response = self.devices["RP2"].get_response()
        self.reaction_time = int(round(time.time() - start_time, 3) * 1000)

    def load_signals(self, sound_type="tts-numbers_resamp_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.setting.signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = get_config(setting="BASE_DIRECTORY")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(19, 28)])
        self.setting.speakers = speakers

    @staticmethod
    def set_log(iterable, k):
        pass

    def pick_speakers_this_trial(self, n_speakers):
        speakers_no_rep = list(x for x in self.setting.speakers if x not in self.setting.speaker_log)
        self.speakers_sample = random.sample(speakers_no_rep, n_speakers)

    def pick_signals_this_trial(self, n_signals):
        signals_no_rep = list(x for x in self.setting.signals if x not in self.setting.signal_log)
        self.signals_sample = random.sample(signals_no_rep, n_signals)

    def calibrate_camera(self, report=True, limit=0.5):
        """
        Calibrates the camera. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        for key in self.devices.keys():
            if not self.devices[key].state == "Ready":
                print("Devices not ready for camera calibration. Make sure devices are initialized!")
        print("Calibrating camera ...")

        spks = SpeakerArray(file=os.path.join(self.devices["ArUcoCam"].setting.root, self.devices["ArUcoCam"].file))  # initialize speakers.
        spks.load_speaker_table()  # load speakertable
        self.devices["ArUcoCam"].led = spks.pick_speakers(23)[0]  # pick central speaker

        print(f"Initialized processors {[k for k in self.devices.keys]}.")
        self.devices["RX81"].write(tag='bitmask',
                                   value=self.led.channel_digital,
                                   processors=self.led.TDT_digital)  # illuminate LED
        print('Point towards led and press button to start calibration...')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        _log = np.zeros(2)
        while True:  # wait in loop for sensor to stabilize
            pose = self.get_pose()
            # print(pose)
            log = np.vstack((_log, pose))
            if log[-1, 0] == None or log[-1, 1] == None:
                print('No marker detected')
            # check if orientation is stable for at least 30 data points
            if len(log) > 30 and all(log[-20:, 0] != None) and all(log[-20:, 1] != None):
                diff = np.mean(np.abs(np.diff(log[-20:], axis=0)), axis=0).astype('float16')
                if report:
                    print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]), end="\r", flush=True)
                if diff[0] < limit and diff[1] < limit:  # limit in degree
                    break
        self.devices["RX81"].write(tag='bitmask', value=0, processors=self.led.TDT_digital)  # turn off LED
        pose_offset = np.around(np.mean(log[-20:].astype('float16'), axis=0), decimals=2)
        # print('calibration complete, thank you!')
        self.devices["ArUcoCam"].offset = pose_offset
        self.devices["ArUcoCam"].calibrated = True


if __name__ == "__main__":
    try:
        subject = Subject()
        subject.name = "Max"
        subject.group = "Test"
        subject.species = "Human"
        subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Max_Test.h5"))
        # subject.file_path
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Max_Test.h5"))
        sl.read_from_h5file()
        test_subject = sl.subjects[0]
    subject.data_path = os.path.join(get_config("DATA_ROOT"), subject.name)
    nje = NumerosityJudgementExperiment(subject=subject)

    nje.start()
    nje.pause()
    nje.stop()


