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
import slab
import time
import numpy as np
import logging
import datetime
import pathlib
import random

log = logging.getLogger(__name__)
config = slab.load_config(os.path.join(get_config("BASE_DIRECTORY"), "config", "locaaccu_config.txt"))


class LocalizationAccuracySetting(ExperimentSetting):

    experiment_name = Str('LocaAccu', group='status', dsec='name of the experiment', noshow=True)
    conditions = Int(config.conditions, group="status", dsec="Number of total speakers")
    trial_number = Int(config.trial_number, group='status', dsec='Number of trials in each condition')
    trial_duration = Float(config.trial_duration, group='status', dsec='Duration of each trial, (s)')

    def _get_total_trial(self):
        return self.trial_number * self.conditions


class LocalizationAccuracyExperiment(ExperimentLogic):

    setting = LocalizationAccuracySetting()
    data = ExperimentData()
    sequence = slab.Trialsequence(conditions=setting.conditions, n_reps=setting.trial_number)
    devices = Dict()
    time_0 = Float()
    all_speakers = List()
    target = Any()
    signals = Any()
    # warning_tone = warning_tone.trim(0.0, 0.225)
    pose = Any()
    #error = np.array([])
    plane = Str("v")
    off_center = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc\\off_center.wav"))
    paradigm_start = slab.Sound.read(os.path.join(get_config("SOUND_ROOT"), "misc\\paradigm_start.wav"))

    def _devices_default(self):
        rp2 = RP2Device()
        rx8 = RX8Device()
        cam = ArUcoCam()
        return {"RP2": rp2,
                "RX8": rx8,
                "ArUcoCam": cam}

    def _initialize(self, **kwargs):
        self.devices["RX8"].handle.write("playbuflen",
                                         self.paradigm_start.duration * self.devices["RX8"].setting.sampling_freq,
                                         procs=self.devices["RX8"].handle.procs)

    def _start(self, **kwargs):
        pass

    def _pause(self, **kwargs):
        pass

    def _stop(self, **kwargs):
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs="RX81")  # turn off LED

    def setup_experiment(self, info=None):
        self._tosave_para["sequence"] = self.sequence
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        self.load_speakers()
        self.load_signal()
        self.devices["RX8"].handle.write("data0", self.paradigm_start.data.flatten(), procs="RX81")
        self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
        self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
        self.devices["RX8"].wait_to_finish_playing()
        self.devices["RX8"].handle.write("playbuflen",
                                         self.devices["RX8"].setting.sampling_freq*self.setting.stim_duration,
                                         procs=self.devices["RX8"].handle.procs)

    def _prepare_trial(self):
        self.devices["RX8"].clear_buffers()
        self.check_headpose()
        self.sequence.__next__()
        solution = self.sequence.this_trial - 1
        self._tosave_para["solution"] = solution
        self.pick_speaker_this_trial(speaker_id=solution)
        signal = random.choice(self.signals)
        self.devices["RX8"].handle.write(tag=f"data0",
                                         value=signal[0].data[:, 0].flatten(),
                                         procs=f"{self.target.TDT_analog}{self.target.TDT_idx_analog}")
        self.devices["RX8"].handle.write(tag=f"chan0",
                                         value=self.target.channel_analog,
                                         procs=f"{self.target.TDT_analog}{self.target.TDT_idx_analog}")

    def _start_trial(self):
        self.time_0 = time.time()  # starting time of the trial
        log.info('trial {} start: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        for device in self.devices.keys():
            self.devices[device].start()
        self.devices["RP2"].wait_for_button()
        reaction_time = int(round(time.time() - self.time_0, 3) * 1000)
        #self.pose = np.array(self.devices["ArUcoCam"]._output_specs["pose"])
        #actual = np.array([self.target.azimuth, self.target.elevation])
        #self.error = np.append(self.error, np.abs(actual-self.pose))
        self._tosave_para["reaction_time"] = reaction_time
        self.devices["RP2"].wait_for_button()
        # time.sleep(0.2)

    def _stop_trial(self):
        #accuracy = np.abs(np.subtract([self.target.azimuth, self.target.elevation], self.pose))
        #log.warning(f"Accuracy azi: {accuracy[0]}, ele: {accuracy[1]}")
        log.info('trial {} end: {}'.format(self.setting.current_trial, time.time() - self.time_0))
        for device in self.devices.keys():
            self.devices[device].pause()
        self.data.save()

    def load_signals(self, sound_type="babble-numbers-reversed-shifted_resamp_24414"):
        sound_root = get_config(setting="SOUND_ROOT")
        sound_fp = pathlib.Path(os.path.join(sound_root, sound_type))
        sound_list = slab.Precomputed(slab.Sound.read(pathlib.Path(sound_fp / file)) for file in os.listdir(sound_fp))
        self.signals = sound_list

    def load_speakers(self, filename="dome_speakers.txt"):
        basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
        filepath = os.path.join(basedir, filename)
        spk_array = SpeakerArray(file=filepath)
        spk_array.load_speaker_table()
        if self.plane == "v":
            speakers = spk_array.pick_speakers([x for x in range(20, 27)])
        elif self.plane == "h":
            speakers = spk_array.pick_speakers([2, 8, 15, 23, 31, 38, 44])
        else:
            log.info("Wrong plane, must be v or h. Unable to load speakers!")
            speakers = [None]
        self.all_speakers = speakers

    def pick_speaker_this_trial(self, speaker_id):
        self.target = self.all_speakers[speaker_id]
        self._tosave_para["target"] = self.target

    def calibrate_camera(self, report=True):
        """
        Calibrates the cameras. Initializes the RX81 to access the central loudspeaker. Illuminates the led on ele,
        azi 0°, then acquires the headpose and uses it as the offset. Turns the led off afterwards.
        """
        log.info("Calibrating camera")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=1,
                                         procs="RX81")  # illuminate central speaker LED
        log.info('Point towards led and press button to start calibration')
        self.devices["RP2"].wait_for_button()  # start calibration after button press
        self.devices["ArUcoCam"].retrieve()
        offset = self.devices["ArUcoCam"]._output_specs["pose"]
        self.devices["ArUcoCam"].offset = offset
        # self.devices["ArUcoCam"].pause()
        for i, v in enumerate(self.devices["ArUcoCam"].offset):  # check for NoneType in offset
            if v is None:
                self.devices["ArUcoCam"].offset[i] = 0
                log.info("Calibration unsuccessful, make sure markers can be detected by cameras!")
        self.devices["RX8"].handle.write(tag='bitmask',
                                         value=0,
                                         procs=f"RX81")  # turn off LED
        self.devices["ArUcoCam"].calibrated = True
        if report:
            log.info(f"Camera offset: {offset}")
        log.info('Calibration complete!')

    def check_headpose(self):
        while True:
            #self.devices["ArUcoCam"].configure()
            self.devices["ArUcoCam"].retrieve()
            # self.devices["ArUcoCam"].pause()
            try:
                if np.sqrt(np.mean(np.array(self.devices["ArUcoCam"]._output_specs["pose"]) ** 2)) > 12.5:
                    log.info("Subject is not looking straight ahead")
                    self.devices["RX8"].clear_buffers()
                    self.devices["RX8"].handle.write("data0", self.off_center.data.flatten(), procs="RX81")
                    self.devices["RX8"].handle.write("chan0", 1, procs="RX81")
                    #self.devices["RX8"].start()
                    #self.devices["RX8"].pause()
                    self.devices["RX8"].handle.trigger("zBusA", proc=self.devices["RX8"].handle)
                    self.devices["RX8"].wait_to_finish_playing()
                else:
                    break
            except TypeError:
                log.info("Cannot detect markers, make sure cameras are set up correctly and arucomarkers can be detected.")
                continue


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
                          group="Pilot",
                          birth=datetime.date(1996, 11, 18),
                          species="Human",
                          sex="M")
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
    la = LocalizationAccuracyExperiment(subject=subject, experimenter=experimenter)
    # la.calibrate_camera()
    la.start()
    # nj.configure_traits()
