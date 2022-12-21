from labplatform.core.Setting import ExperimentSetting
from labplatform.core.ExperimentLogic import ExperimentLogic
from labplatform.core.Data import ExperimentData
from labplatform.core.Subject import Subject, SubjectList
from labplatform.config import get_config
from experiment.RP2 import RP2Device
from experiment.RX8 import RX81Device

import os
from traits.api import Any, List, CInt
import numpy as np
import random
import slab

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = "Numerosity Judgement"
    speakers = List(group="primary", dsec="list of speakers")
    stimuli = List(group="primary", dsec="Set to choose stimuli from")
    n_blocks = CInt(1, group="primary", dsec="Number of total blocks per session")
    n_trials = CInt(20, group="primary", dsec="Number of total trials per block")
    conditions = CInt(4, group="primary", dsex="Number of conditions in the experiment")

class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = List()
    RX81 = RX81Device()
    RP2 = RP2Device()

    def _initialize(self, **kargs):
        self.RX81.initialize()
        self.RP2.initialize()
        self.sequence = slab.Trialsequence

    def setup_experiment(self, info=None):
        self.sequence = np.repeat(range(4), self.setting.trial_number)
        random.shuffle(self.sequence)

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
    experiment.devices["RP2"].setting.stimulus = np.ones(50000)
    experiment.initialize()
    experiment.devices["RP2"].configure()
    experiment.configure(trial_number=10)
    experiment.start()
