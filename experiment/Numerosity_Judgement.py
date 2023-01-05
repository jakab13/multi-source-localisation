from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RM1_RP2_sim import RP2Device
from experiment.RX8_sim import RX8Device
from Speakers.speaker_config import SpeakerArray
import os
from traits.api import Any, List, CInt, Str, Property, Int
import numpy as np
import random
import slab


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

    def _initialize(self, **kwargs):
        self.device["RP2"] = RP2Device()
        self.decive["RX81"] = RX8Device()
        self.device["RX81"].setting.index = 1
        self.device["RX82"] = RX8Device()
        self.device["RX82"].setting.index = 2

        keys = self.devices.keys()
        for device in keys:
            self.devices[device].initialize()

    def setup_experiment(self, info=None):
        self.sequence = slab.Trialsequence(conditions=self.setting.conditions, n_reps=self.setting.n_trials)
        self.initialize()

    def configure_experiment(self):
        pass

    def _configure(self, **kargs):
        pass

    def _prepare_trial(self, **kwargs):
        self.set_signals_and_speakers(self, kwargs)

    def _start_trial(self):
        self.devices['RM1'].start()

    def _stop_trial(self):
        pass

    def _stop(self):
        pass

    def _pause(self):
        pass

    def load_sounds(self):
        pass


if __name__ == "__main__":
    subject = Subject()
    experiment = NumerosityJudgementExperiment(subject=subject)

    basedir = get_config(setting="BASE_DIRECTORY")
    filename = "dome_speakers.txt"
    file = os.path.join(basedir, filename)
    spk_array = SpeakerArray(file=file)
    spk_array.load_speaker_table()
    spks = spk_array.pick_speakers(picks=[x for x in range(20, 23)])

    signals = [slab.Sound.vowel()] * len(spks)

    experiment.initialize()
    experiment.start()
    experiment.pause()
    experiment.stop()
