from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RM1_RP2_sim import RP2Device
from experiment.RX8_sim import RX8Device
from experiment.Camera import FlirCam
from Speakers.speaker_config import SpeakerArray
import os
from traits.api import Any, List, CInt, Str
import numpy as np
import random
import slab
import pathlib

#TODO: what are all the methods supposed to do? What is the basic workflow of the experiment logic?
#TODO: stimuli names do not include gender --> sort stimuli by gender
#TODO: how to write responses to ExperimentData?

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = Str('Numerosity Judgement', group='status', dsec='name of the experiment', noshow=True)
    speakers = List(group="primary", dsec="list of speakers", reinit=False)
    signals = List(group="primary", dsec="Set to choose stimuli from", reinit=False)
    n_blocks = CInt(1, group="primary", dsec="Number of total blocks per session", reinit=False)
    n_trials = CInt(20, group="primary", dsec="Number of total trials per block", reinit=False)
    n_conditions = List([2, 3, 4, 5], group="primary",
                        dsec="Number of simultaneous talkers in the experiment", reinit=False)
    signal_log = Any(group="primary", dsec="Logs of the speakers and signals used in previous trials", reinit=False)


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = Any()
    devices = Any()
    all_speakers = Any()
    all_signals = Any()
    speakers_sample = Any()
    signals_sample = Any()

    def _initialize(self, **kwargs):
        self.device["RP2"] = RP2Device()
        self.decive["RX81"] = RX8Device()
        self.device["RX81"].setting.index = 1
        self.device["RX82"] = RX8Device()
        self.device["RX82"].setting.index = 2
        self.device["FlirCam"] = FlirCam()

        for device in self.devices.keys:
            self.devices[device].initialize()

    def _configure(self, **kwargs):
        self.devices["RX81"].setting.signals = self.signals_sample
        self.devices["RX81"].setting.speakers = self.speakers_sample
        for device in self.devices.keys:
            self.devices[device].configure()

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
        pass

    def setup_experiment(self, info=None):
        self.sequence = slab.Trialsequence(conditions=self.setting.conditions, n_reps=self.setting.n_trials)
        self.initialize()
        self.load_speakers()
        self.load_signals()

    def configure_experiment(self):
        self.pick_speakers_this_trial(n_speakers=self.sequence.this_trial)
        self.pick_signals_this_trial(n_signals=self.sequence.this_trial)
        self.configure()

    def start_experiment(self, info=None):
        self.sequence.__next__()
        for device in self.devices.keys:
            self.devices[device].start()
        self.devices["RP2"].wait_for_button()
        response = self.devices["RP2"].get_response()

    def load_signals(self, sound_type="tts-numbers_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.all_signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = get_config(setting="BASE_DIRECTORY")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(19, 28)])
        self.all_speakers = speakers

    def pick_speakers_this_trial(self, n_speakers):
        self.speakers_sample = random.sample(self.all_speakers, n_speakers)

    def pick_signals_this_trial(self, n_signals):
        self.signals_sample = random.sample(self.all_signals, n_signals)





if __name__ == "__main__":
    subject = Subject()
    subject.name = "Max"
    subject.group = "pilot"
    subject.species = "Human"
    experiment = NumerosityJudgementExperiment(subject=subject)

    experiment.initialize()
    experiment.start()
    experiment.pause()
    experiment.stop()

    # sort signals
    sound_type = "tts-numbers_24414"
    sound_root = get_config(setting="SOUND_ROOT")
    sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
    sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))

    # sort signals by talker
    all_talkers = dict()
    talker_id_range = range(225, 377)
    for talker_id in talker_id_range:
        talker_sorted = list()
        for i, sound in enumerate(os.listdir(sound_fp)):
            if str(talker_id) in sound:
                talker_sorted.append(sound_list[i])
        all_talkers[str(talker_id)] = talker_sorted

    # sort signals by number
    number_range = ["one", "two", "three", "four", "five"]
    all_numbers = dict()
    for number in number_range:
        numbers_sorted = list()
        for i, sound in enumerate(os.listdir(sound_fp)):
            if number in sound:
                numbers_sorted.append(sound_list[i])
        all_numbers[number] = numbers_sorted
