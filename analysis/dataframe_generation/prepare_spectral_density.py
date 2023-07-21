from analysis.utils.math import spectemp_coverage
from analysis.utils.misc import *
import os
import pickle as pkl
from labplatform.config import get_config
import pandas as pd
import pickle


"""
This script is mainly for dataframe generation in order to further analyze with (spectral density). 
Hence, we only prepare the data for the numerosity judgement paradigm.
"""


# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
revspeech_h = revspeech_h.sort_index()
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False
clearspeech_h = clearspeech_h.sort_index()

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
revspeech_v = revspeech_v.sort_index()
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False
clearspeech_v = clearspeech_v.sort_index()

# get talker files
talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds_clear = pkl.load(files)

talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_reversed.pkl")
with open(talker_files_path, "rb") as files:
    sounds_reversed = pkl.load(files)

# get info from trials horizontal
signals_sample_clear_h = clearspeech_h.signals_sample  # talker IDs
country_idxs_clear_h = clearspeech_h.country_idxs  # indices of the country names from a talker

signals_sample_reversed_h = revspeech_h.signals_sample  # talker IDs
country_idxs_reversed_h = revspeech_h.country_idxs  # indices of the country names from a talker

# get info from trials vertical
signals_sample_clear_v = clearspeech_v.signals_sample  # talker IDs
country_idxs_clear_v = clearspeech_v.country_idxs  # indices of the country names from a talker

signals_sample_reversed_v = revspeech_v.signals_sample  # talker IDs
country_idxs_reversed_v = revspeech_v.country_idxs  # indices of the country names from a talker

dyn_range = 65  # threshold dB yielding highest coverage variance
resize = 0.6  # resize duration
upper_freq = 11000

# extract horizontal data
clearspeech_data_h = dict(sound=[], coverage=[])
for trial_n in range(clearspeech_h.__len__()):
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    signals = signals_sample_clear_h[trial_n]
    country_idx = country_idxs_clear_h[trial_n]
    trial_composition = [sounds_clear[x][y].resize(resize) for x, y in zip(signals, country_idx)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range=dyn_range, upper_freq=upper_freq)
    clearspeech_data_h["sound"].append(sum(trial_composition))
    clearspeech_data_h["coverage"].append(percentage_filled)

revspeech_data_h = dict(sound=[], coverage=[])
for trial_n in range(revspeech_h.__len__()):
    # reversed
    sound = slab.Sound(data=np.zeros(int(48828)), samplerate=48828)
    signals = signals_sample_reversed_h[trial_n]
    country_idx = country_idxs_reversed_h[trial_n]
    trial_composition = [sounds_reversed[x][y].resize(resize) for x, y in zip(signals, country_idx)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range, upper_freq=upper_freq)
    revspeech_data_h["sound"].append(sum(trial_composition))
    revspeech_data_h["coverage"].append(percentage_filled)

clearspeech_data_v = dict(dict(sound=[], coverage=[]))
for trial_n in range(clearspeech_v.__len__()):
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    signals = signals_sample_clear_v[trial_n]
    country_idx = country_idxs_clear_v[trial_n]
    trial_composition = [sounds_clear[x][y].resize(resize) for x, y in zip(signals, country_idx)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range, upper_freq=upper_freq)
    clearspeech_data_v["sound"].append(sum(trial_composition))
    clearspeech_data_v["coverage"].append(percentage_filled)

revspeech_data_v = dict(dict(sound=[], coverage=[]))
for trial_n in range(revspeech_v.__len__()):
    sound = slab.Sound(data=np.zeros(int(48828)), samplerate=48828)
    signals = signals_sample_reversed_v[trial_n]
    country_idx = country_idxs_reversed_v[trial_n]
    trial_composition = [sounds_reversed[x][y].resize(resize) for x, y in zip(signals, country_idx)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range, upper_freq=upper_freq)
    revspeech_data_v["sound"].append(sum(trial_composition))
    revspeech_data_v["coverage"].append(percentage_filled)

# make list of dicts
dicts = list()
dicts.append(clearspeech_data_h)
dicts.append(revspeech_data_h)
dicts.append(clearspeech_data_v)
dicts.append(revspeech_data_v)

df = pd.DataFrame(dicts, index=["clearspeech_h", "revspeech_h", "clearspeech_v", "revspeech_v"])
filename = "Results/coverage_dataframe.pkl"
pickle.dump(df, open(filename, "wb"))


