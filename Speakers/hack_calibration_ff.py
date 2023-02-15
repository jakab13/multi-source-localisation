import freefield
import pickle
import slab
import os
from Speakers.speaker_config import SpeakerArray
from labplatform.config import get_config
import numpy as np

# important saved calibration results:
#    SPL_ref: int, id of reference speaker
#    SPL_const: float, constant used to set slab calibration constant
#    SPL_eq_spks, list of int, id of speakers in SPL equalization result
#    SPL_eq: list of float, dB differences w.r.t. reference
#    filters_spks: list of int, id of speakers in spectrum equalization result
#    filters: 2d array, list of equalizing filters; 2nd dimension is the different filters

fp = os.path.join("C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Anaconda3_64\\envs\\freefield\\Lib\\site-packages\\freefield\\data", "calibration_dome.pkl")
with open(fp, "rb") as f:
    calib_ff = pickle.load(f)

calib_labplatform = dict()

spk_array = SpeakerArray()
spk_array.load_speaker_table(file=os.path.join(get_config("BASE_DIRECTORY"), "speakers", "dome_speakers.txt"))

calib_labplatform["SPL_ref"] = 23
calib_labplatform["SPL_const"] = 1.5  # need sound level meter for this
calib_labplatform["SPL_eq_spks"] = [spk.id for spk in spk_array.speakers]
calib_labplatform["SPL_eq"] = [calib_ff[str(i)]["level"] for i in range(len(spk_array.speakers))]
calib_labplatform["filters_spks"] = [spk.id for spk in spk_array.speakers]
calib_labplatform["filters"] = [calib_ff[str(i)]["filter"].resample(24414) for i in range(len(spk_array.speakers))]

calib_filename = os.path.join(get_config("CAL_ROOT"), "calibration_labplatform_test.pkl")
with open(calib_filename, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(calib_labplatform, f, pickle.HIGHEST_PROTOCOL)


# test
if __name__ == "__main__":
    spks = SpeakerArray()
    spks.load_speaker_table(file=os.path.join(get_config("BASE_DIRECTORY"), "speakers", "dome_speakers.txt"))
    spks.load_calibration(file="calibration_labplatform_test.pkl")