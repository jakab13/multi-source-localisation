from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RM1_RP2_sim import RP2Device
from experiment.RX8_sim import RX8Device

import os
from traits.api import Any, List, CInt
import numpy as np
import random
import slab

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = "Numerosity Judgement"
    speakers = List(group="primary", dsec="list of speakers")
    signals = List(group="primary", dsec="Set to choose stimuli from")
    n_blocks = CInt(1, group="primary", dsec="Number of total blocks per session")
    n_trials = CInt(20, group="primary", dsec="Number of total trials per block")
    conditions = CInt(4, group="primary", dsex="Number of conditions in the experiment")

class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = Any
    RX81 = RX8Device()
    RX82 = RX8Device()
    RP2 = RP2Device()

    def _initialize(self, **kwargs):
        self.RX81.initialize()
        self.RP2.initialize()
        #self.sequence = slab.Trialsequence()

    def setup_experiment(self, info=None):
        self.sequence = slab.Trialsequence(conditions=self.setting.conditions, n_reps=self.setting.n_trials)

    def generate_stimulus(self):
        pass

    def _prepare_trial(self):
        self.devices["RP2"].handle.SetTagVal("Bit", self.sequence[self.setting.current_trial])

    def _start_trial(self):
        self.devices['RP2'].start()

    def _stop_trial(self):
        pass

    def _stop(self):
        pass

    def _pause(self):
        pass

    def set_signals_and_speakers(self, signals, speakers):
        for idx, speaker in enumerate(speakers):
            self.setting.stimulus = signals[idx]
            self.handle.write(tag=f"data{idx}",
                              value=self.setting.stimulus,
                              procs=f"{self.setting.processor}{self.setting.index}")
            self.handle.write(tag=f"chan{idx}",
                              value=speaker,
                              procs=f"{self.setting.processor}{self.setting.index}")
            print(f"Set signal to speaker {speaker}")


if __name__ == "__main__":

    try:
        test_subject = Subject()
        test_subject.name ="Ole"
        test_subject.group ="Test"
        test_subject.add_subject_to_h5file(os.path.join(get_config("SUBJECT_ROOT"), "Ole_Test.h5"))
        #test_subject.file_path
    except ValueError:
        # read the subject information
        sl = SubjectList(file_path=os.path.join(get_config("SUBJECT_ROOT"), "Ole_Test.h5"))
        sl.read_from_h5file()
        test_subject = sl.subjects[0]
    experiment = NumerosityJudgementExperiment(subject=test_subject)
    experiment.initialize()
    experiment.devices["RP2"].configure()
    experiment.start()
