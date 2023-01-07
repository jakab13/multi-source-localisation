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
from traits.api import Any, List, CInt, Str, Property, Int
import numpy as np
import random
import slab
import pathlib

#TODO: what are all the methods supposed to do? What is the basic workflow of the experiment logic?

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = Str('Numerosity Judgement', group='status', dsec='name of the experiment', noshow=True)
    speakers = List(group="primary", dsec="list of speakers")
    signals = List(group="primary", dsec="Set to choose stimuli from")
    n_blocks = CInt(1, group="primary", dsec="Number of total blocks per session")
    n_trials = CInt(20, group="primary", dsec="Number of total trials per block")
    n_conditions = CInt(4, group="primary", dsec="Number of conditions in the experiment")


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = Any()
    devices = Any()
    speakers = Any()
    signals = Any()
    sample = Any()

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
        self.pick_speakers()
        self.load_signals()

    def configure_experiment(self, condition):
        self.pick_signals(condition=condition)
        self.configure()

    def start_experiment(self, info=None):
        for device in self.devices.keys:
            self.devices[device].start()

    def load_signals(self, sound_type="tts_numbers_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def pick_signals(self, condition):
        self.sample = random.sample(self.signals, condition)

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = get_config(setting="BASE_DIRECTORY")
        filename = filename
        file = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=file)
        spk_array.load_speaker_table()
        speakers = spk_array.pick_speakers([x for x in range(19, 28)])
        self.speakers = speakers



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
