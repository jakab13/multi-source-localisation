from labplatform.config import get_config
from Speakers.speaker_config import SpeakerArray, load_speaker_config
from pathlib import Path
from matplotlib import pyplot as plt
import slab
from copy import deepcopy
import os
import numpy as np
import datetime
import logging
import pickle
from Speakers.TDT_RP2RX8_speaker_calibration import RP2RX8SpeakerCal
logger = logging.getLogger(__name__)


class FFCalibrator:

    def __init__(self, config):
        self.config = config
        # load configuration parameters
        param = load_speaker_config(self.config)
        self.config_param = param
        # load speaker table
        cal_dir = get_config("CAL_ROOT")
        speaker_dir = os.path.join(get_config("BASE_DIRECTORY"), "speakers")
        self.speakerArray = SpeakerArray(file=os.path.join(speaker_dir, param['speaker_file']))
        self.speakerArray.load_speaker_table()
        # sound wave data
        self.stimulus = None
        # calibration results
        self.results = dict()
        # file path
        self.cal_dir = cal_dir
        self._datestr = datetime.datetime.now().strftime(get_config("DATE_FMT"))
        # initialize the device used to perform speaker calibration
        # the device should be an instance of Device class
        self.device = RP2RX8SpeakerCal()
        self.calib_param = {"ref_spk_id": 23,
                            "samplerate": 24414,
                            "n_repeats": 20,
                            "calib_db": 75,
                            'filter_bank': {'length': 512,
                                            'bandwidth': 0.125,
                                            'low_cutoff': 20,
                                            'high_cutoff': 12000,
                                            'alpha': 1.0,
                                            "threshold": 0.3},
                            'ramp_dur': 0.005,
                            "stim_dur": 0.1,
                            "stim_type": "chirp",
                            "speaker_distance": 1.4,
                            }

    def calibrate(self, speakers="all", save=True):
        """
        Equalize the loudspeaker array in two steps. First: equalize over all
        level differences by a constant for each speaker. Second: remove spectral
        difference by inverse filtering. For more details on how the
        inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank

        Args:
            speakers (list, string): Select speakers for equalization. Can be a list of speaker indices or 'all'
            bandwidth (float): Width of the filters, used to divide the signal into subbands, in octaves. A small
                bandwidth results in a fine tuned transfer function which is useful for equalizing small notches.
            threshold (float): Threshold for level equalization. Correct level only for speakers that deviate more
                than <threshold> dB from reference speaker
            low_cutoff (int | float): The lower limit of frequency equalization range in Hz.
            high_cutoff (int | float): The upper limit of frequency equalization range in Hz.
            alpha (float): Filter regularization parameter. Values below 1.0 reduce the filter's effect, values above
                amplify it. WARNING: large filter gains may result in temporal distortions of the sound
            file_name (string): Name of the file to store equalization parameters.
            save (Bool): saves the calibration file as .pkl when set True.

        """
        self.load_stimulus()
        bandwidth = self.calib_param["filter_bank"]["bandwidth"]
        threshold = self.calib_param["filter_bank"]["threshold"]
        low_cutoff = self.calib_param["filter_bank"]["low_cutoff"]
        high_cutoff = self.calib_param["filter_bank"]["high_cutoff"]
        alpha = self.calib_param["filter_bank"]["alpha"]
        stimlevel = self.calib_param["calib_db"]
        whitenoise = slab.Sound.whitenoise(duration=10.0,
                                           level=stimlevel,
                                           samplerate=self.device.setting.device_freq)
        self.set_signal_and_speaker(signal=whitenoise,
                                    speaker=self.speakerArray.pick_speakers(23)[0], equalize=False)
        self.device.RX8.trigger("zBusA", proc=self.device.RX8)
        self.device.wait_to_finish_playing()
        intensity = float(input("Enter measured sound intensity: "))
        self.results["SPL_const"] = intensity - whitenoise.level
        sound = self.stimulus
        if speakers == "all":  # use the whole speaker table
            speakers = self.speakerArray.pick_speakers(speakers)
        else:
            speakers = self.speakerArray.pick_speakers(picks=speakers)
        reference_speaker = self.speakerArray.pick_speakers(self.calib_param["ref_spk_id"])[0]
        self.results["SPL_ref"] = self.calib_param["ref_spk_id"]
        target, equalization_levels_before = self._level_equalization(speakers, sound, reference_speaker, threshold)
        print(equalization_levels_before)
        filter_bank, rec = self._frequency_equalization(speakers, sound, target, equalization_levels_before,
                                                        bandwidth, low_cutoff, high_cutoff, alpha)
        target, equalization_levels_after = self._level_equalization(speakers, sound, reference_speaker, threshold)
        print(equalization_levels_after)
        self.results['filters'] = filter_bank
        self.results['filters_spks'] = [spk.id for spk in speakers]
        self.results["SPL_eq_spks"] = [spk.id for spk in speakers]
        self.results["SPL_eq"] = equalization_levels_after
        print(equalization_levels_before - equalization_levels_after)
        if save:
            self._save_result()

    def _save_result(self, file_name=None):
        if file_name is None:  # use the default filename and rename the existing file
            file_name = Path(os.path.join(get_config("CAL_ROOT"), f'{self.config}_calibration.pkl'))
        else:
            file_name = Path(file_name)
        if os.path.isfile(file_name):  # move the old calibration to the log folder
            date = self._datestr
            file_name.rename(file_name.parent / (file_name.stem + "_deprecated_" + date + file_name.suffix))
        with open(file_name, 'wb') as f:  # save the newly recorded calibration
            pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)

    def load_stimulus(self):
        if self.calib_param["stim_type"] == "chirp":
            self.stimulus = slab.Sound.chirp(duration=self.calib_param["stim_dur"],
                                             samplerate=self.calib_param["samplerate"],
                                             level=self.calib_param["calib_db"])
            self.stimulus = self.stimulus.ramp(duration=self.calib_param["ramp_dur"])
        else:
            print("Stimulus type must be chirp!")

    def set_signal_and_speaker(self, signal, speaker, equalize=True):
        """
        Load a signal into the processor buffer and set the output channel to match the speaker.
        The processor is chosen automatically depending on the speaker.

            Args:
                signal (array-like): signal to load to the buffer, must be one-dimensional
                speaker (Speaker, int) : speaker to play the signal from, can be index number or [azimuth, elevation]
                equalize (bool): if True (=default) apply loudspeaker equalization
        """
        if not isinstance(signal, slab.Sound):
            signal = slab.Sound(signal)
        speaker = self.speakerArray.pick_speakers(picks=speaker)[0]
        if equalize:
            logging.info('Applying calibration.')  # apply level and frequency calibration
            to_play = speaker.apply_equalization(signal, speaker)
        else:
            to_play = signal
        self.device.RX8.write(tag='chan', value=speaker.channel_analog, procs=f"{speaker.TDT_analog}{speaker.TDT_idx_analog}")
        self.device.RX8.write(tag='data', value=to_play.data.flatten(), procs=f"{speaker.TDT_analog}{speaker.TDT_idx_analog}")
        other_procs = ["RX81", "RX82"]
        other_procs.remove(f"{speaker.TDT_analog}{speaker.TDT_idx_analog}")  # set the analog output of other processors to non existent number 99
        self.device.RX8.write(tag='chan', value=99, procs=other_procs)

    def play_and_record(self, speaker, sound, compensate_delay=True, compensate_attenuation=False, equalize=True):
        """
        Play the signal from a speaker and return the recording. Delay compensation
        means making the buffer of the recording processor n samples longer and then
        throwing the first n samples away when returning the recording so sig and
        rec still have the same length. For this to work, the circuits rec_buf.rcx
        and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
        be plugged in.
        Parameters:
            speaker: integer between 1 and 48, index number of the speaker
            sound: instance of slab.Sound, signal that is played from the speaker
            compensate_delay: bool, compensate the delay between play and record
            compensate_attenuation:
            equalize:
            recording_samplerate: samplerate of the recording
        Returns:
            rec: 1-D array, recorded signal
        """
        recording_samplerate = self.device.setting.device_freq
        # self.device.RX8.write(tag="playbuflen", value=sound.n_samples, procs=["RX81", "RX82"])
        if compensate_delay:
            n_delay = self.get_recording_delay(play_from="RX8", rec_from="RP2")
            n_delay += 50  # make the delay a bit larger to avoid missing the sound's onset
        else:
            n_delay = 0
        rec_n_samples = int(sound.duration * recording_samplerate)
        # print('buffer length', rec_n_samples + n_delay)
        self.device.RX8.write(tag="playbuflen", value=sound.n_samples, procs=["RX81", "RX82"])
        self.device.RP2.SetTagVal("playbuflen", rec_n_samples + n_delay)
        self.set_signal_and_speaker(sound, speaker, equalize)
        self.device.RX8.trigger("zBusA", proc=self.device.RX8)
        self.device.wait_to_finish_playing()
        rec = self.device.RP2.ReadTagV('data', 0, rec_n_samples)[n_delay:]
        rec = slab.Sound(np.array(rec), samplerate=recording_samplerate)
        if sound.samplerate != recording_samplerate:
            rec = rec.resample(recording_samplerate)
        if compensate_attenuation:
            if isinstance(rec, slab.Binaural):
                iid = rec.left.level - rec.right.level
                rec.level = sound.level
                rec.left.level += iid
            else:
                rec.level = sound.level
        return rec

    def get_recording_delay(self, play_from=None, rec_from=None):
        """
            Calculate the delay it takes for played sound to be recorded. Depends
            on the distance of the microphone from the speaker and on the device
            digital-to-analog and analog-to-digital conversion delays.

            Args:
                distance (float): distance between listener and speaker array in meters
                sample_rate (int): sample rate under which the system is running
                play_from (str): processor used for digital to analog conversion
                rec_from (str): processor used for analog to digital conversion

        """
        distance = self.calib_param["speaker_distance"]
        sample_rate = self.device.setting.device_freq
        n_sound_traveling = int(distance / 343 * sample_rate)
        if play_from:
            if play_from == "RX8":
                n_da = 24
            elif play_from == "RP2":
                n_da = 30
            else:
                logging.warning(f"dont know D/A-delay for processor type {play_from}...")
                n_da = 0
        else:
            n_da = 0
        if rec_from:
            if rec_from == "RX8":
                n_ad = 47
            elif rec_from == "RP2":
                n_ad = 65
            else:
                logging.warning(f"dont know A/D-delay for processor type {rec_from}...")
                n_ad = 0
        else:
            n_ad = 0
        return n_sound_traveling + n_da + n_ad

    def apply_equalization(self, signal, speaker, level=True, frequency=True):
        """
        Apply level correction and frequency equalization to a signal

        Args:
            signal: signal to calibrate
            speaker: index number, coordinates or row from the speaker table. Determines which calibration is used
            level:
            frequency:
        Returns:
            slab.Sound: calibrated copy of signal
        """
        signal = slab.Sound(signal)
        speaker = self.speakerArray.pick_speakers(speaker)[0]
        equalized_signal = deepcopy(signal)
        if level:
            if speaker.level is None:
                raise ValueError("speaker not level-equalized! Load an existing equalization of calibrate the setup!")
            equalized_signal.level += speaker.level
        if frequency:
            if speaker.filter is None:
                raise ValueError(
                    "speaker not frequency-equalized! Load an existing equalization of calibrate the setup!")
            equalized_signal = speaker.filter.apply(equalized_signal)
        return equalized_signal

    def _level_equalization(self, speakers, sound, reference_speaker, threshold):
        """
        Record the signal from each speaker in the list and return the level of each
        speaker relative to the target speaker(target speaker must be in the list)
        """
        target_recording = self.play_and_record(reference_speaker, sound, equalize=False)
        recordings = []
        for speaker in speakers:
            recordings.append(self.play_and_record(speaker, sound, equalize=False))
        recordings = slab.Sound(recordings)
        recordings.data[:, np.logical_and(recordings.level > target_recording.level - threshold,
                                          recordings.level < target_recording.level + threshold)] = target_recording.data
        equalization_levels = target_recording.level - recordings.level
        # recordings.data[:, recordings.level < threshold] = target_recording.data  # thresholding
        # recordings.data[:, equalization_levels.level < threshold] = target_recording.data
        # return target_recording.level / recordings.level
        return target_recording, equalization_levels

    def _frequency_equalization(self, speakers, sound, reference_sound, calibration_levels, bandwidth,
                                low_cutoff, high_cutoff, alpha):
        """
        play the level-equalized signal, record and compute and a bank of inverse filter
        to equalize each speaker relative to the target one. Return filterbank and recordings
        """
        recordings = []
        for speaker, level in zip(speakers, calibration_levels):
            attenuated = deepcopy(sound)
            attenuated.level += level
            temp_recs = []
            for i in range(10):
                rec = self.play_and_record(speaker, attenuated, equalize=False)
                # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
                temp_recs.append(rec.data)
            recordings.append(slab.Sound(data=np.mean(temp_recs, axis=0), samplerate=self.device.setting.device_freq))
        recordings = slab.Sound(recordings, samplerate=self.device.setting.device_freq)
        length = self.calib_param["filter_bank"]["length"]
        filter_bank = slab.Filter.equalizing_filterbank(reference_sound, recordings, length=length, low_cutoff=low_cutoff,
                                                        high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
        # check for notches in the filter:
        transfer_function = filter_bank.tf(show=False)[1][0:900, :]
        if (transfer_function < -30).sum() > 0:
            print("Some of the equalization filters contain deep notches - try adjusting the parameters.")
        return filter_bank, recordings

    def test_equalization(self, speakers="all"):
        """
        Test the effectiveness of the speaker equalization
        """
        if not self.speakerArray.calib_result:
            self.speakerArray.calib_result = self.results
            self.speakerArray._apply_calib_result()
        stim = self.stimulus
        stim.level = self.calib_param['calib_db']

        # the recordings from the un-equalized, the level equalized and the fully equalized sounds
        rec_raw, rec_level, rec_full = [], [], []
        speakers = self.speakerArray.pick_speakers(speakers)
        for speaker in speakers:
            level_equalized = self.apply_equalization(stim, speaker=speaker, level=True, frequency=False)
            full_equalized = self.apply_equalization(stim, speaker=speaker, level=True, frequency=True)
            rec_raw.append(self.play_and_record(speaker, stim, equalize=False))
            rec_level.append(self.play_and_record(speaker, level_equalized, equalize=False))
            rec_full.append(self.play_and_record(speaker, full_equalized, equalize=False))
        return slab.Sound(rec_raw), slab.Sound(rec_level), slab.Sound(rec_full)

    def spectral_range(self, signal, thresh=3, plot=True, log=False):
        """
        Compute the range of differences in power spectrum for all channels in
        the signal. The signal is devided into bands of equivalent rectangular
        bandwidth (ERB - see More& Glasberg 1982) and the level is computed for
        each frequency band and each channel in the recording. To show the range
        of spectral difference across channels the minimum and maximum levels
        across channels are computed. Can be used for example to check the
        effect of loud speaker equalization.
        """
        bandwidth = self.calib_param["filter_bank"]["bandwidth"]
        low_cutoff = self.calib_param["filter_bank"]["low_cutoff"]
        high_cutoff = self.calib_param["filter_bank"]["high_cutoff"]
        # generate ERB-spaced filterbank:
        length = self.calib_param["filter_bank"]["length"]
        filter_bank = slab.Filter.cos_filterbank(length=length, bandwidth=bandwidth,
                                                 low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                                 samplerate=signal.samplerate)
        center_freqs, _, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth)
        center_freqs = slab.Filter._erb2freq(center_freqs)
        # create arrays to write data into:
        levels = np.zeros((signal.n_channels, filter_bank.n_channels))
        max_level, min_level = np.zeros(filter_bank.n_channels), np.zeros(filter_bank.n_channels)
        for i in range(signal.n_channels):  # compute ERB levels for each channel
            levels[i] = filter_bank.apply(signal.channel(i)).level
        for i in range(filter_bank.n_channels):  # find max and min for each frequency
            max_level[i] = max(levels[:, i])
            min_level[i] = min(levels[:, i])
        difference = max_level - min_level
        if plot is True or isinstance(plot, plt.Axes):
            if isinstance(plot, plt.Axes):
                ax = plot
            else:
                fig, ax = plt.subplots(1)
            # frequencies where the difference exceeds the threshold
            bads = np.where(difference > thresh)[0]
            for y in [max_level, min_level]:
                if log is True:
                    ax.semilogx(center_freqs, y, color="black", linestyle="--")
                else:
                    ax.plot(center_freqs, y, color="black", linestyle="--")
            for bad in bads:
                ax.fill_between(center_freqs[bad - 1:bad + 1], max_level[bad - 1:bad + 1],
                                min_level[bad - 1:bad + 1], color="red", alpha=.6)
        plt.show()
        return difference

if __name__ == '__main__':
    cal = FFCalibrator('FREEFIELD')
   # cal.calibrate()
