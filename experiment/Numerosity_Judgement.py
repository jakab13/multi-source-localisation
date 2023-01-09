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
from traits.api import Any, List, CInt, Str, Int, Float
import random
import slab
import pathlib
import time

#TODO: what are all the methods supposed to do? What is the basic workflow of the experiment logic?
#TODO: stimuli names do not include gender --> sort stimuli by gender
#TODO: test experiment data class (write/read data from file)
#TODO: check signal and speaker log before trial

class NumerosityJudgementSetting(ExperimentSetting):
    experiment_name = Str('Numerosity Judgement', group='status', dsec='name of the experiment', noshow=True)
    speakers = List(group="primary", dsec="List of speakers", reinit=False)
    signals = List(group="primary", dsec="Set to choose stimuli from", reinit=False)
    n_blocks = CInt(1, group="status", dsec="Number of total blocks per session")
    n_trials = CInt(20, group="status", dsec="Number of total trials per block")
    conditions = List([2, 3, 4, 5], group="status",
                        dsec="Number of simultaneous talkers in the experiment")
    signal_log = Any(group="primary", dsec="Logs of the signals used in previous trials", reinit=False)
    speaker_log = Any(group="primary", dsec="Logs of the speakers used in previous trials", reinit=False)


class NumerosityJudgementExperiment(ExperimentLogic):

    setting = NumerosityJudgementSetting()
    data = ExperimentData()
    sequence = Any()
    devices = Any()
    speakers_sample = List()
    signals_sample = List()
    response = Int()
    reaction_time = Int()
    time_0 = time.time()

    def _initialize(self, **kwargs):
        self.devices["RP2"] = RP2Device()
        self.decives["RX81"] = RX8Device()
        self.devices["RX81"].setting.index = 1
        self.devices["RX82"] = RX8Device()
        self.devices["RX82"].setting.index = 2
        self.devices["FlirCam"] = FlirCam()

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
        start_time = time.time()
        self.devices["RP2"].wait_for_button()
        self.response = self.devices["RP2"].get_response()
        self.reaction_time = int(round(time.time() - start_time, 3) * 1000)

    def load_signals(self, sound_type="tts-numbers_24414"):
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

    def get_idx_val(self, iterable, k):
        indices = list()
        values = list()
        index_value = random.sample(enumerate(iterable), k)
        for idx, val in index_value:
            indices.append(idx)
            values.append(val)
        return indices, values

    def pick_speakers_this_trial(self, n_speakers):
        self.setting.speaker_log, self.speakers_sample = self.get_idx_val(self.setting.speakers, k=n_speakers)

    def pick_signals_this_trial(self, n_signals):
        self.setting.signal_log, self.signals_sample = self.get_idx_val(self.setting.signals, k=n_signals)






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
