from analysis.utils.plotting import *
import os
import seaborn as sns
from analysis.utils.math import spectemp_coverage
import matplotlib.pyplot as plt
import pickle as pkl
from labplatform.config import get_config
import librosa
from sklearn.metrics import auc
sns.set_theme()

# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False

signals_sample_reversed_h = revspeech_h.signals_sample  # talker IDs
country_idxs_reversed_h = revspeech_h.country_idxs  # indices of the country names from a talker

# get sub ids
sub_ids = extract_subject_ids_from_dataframe(dfh)

# get talker files
talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds_clear = pkl.load(files)

talkers = list(sounds_clear.keys())

sounds_reversed = dict()
for talker in talkers:
    sounds_reversed[talker] = list()
    for sound in sounds_clear[talker]:
        revsound = slab.Sound(data=sound[::-1], samplerate=sound.samplerate)
        sounds_reversed[talker].append(revsound)

dyn_range = 65  # threshold dB yielding highest coverage variance
upper_freq = 11000

revspeech_data_h = dict(sound=[], coverage=[])
for trial_n in range(revspeech_h.__len__()):
    # reversed
    signals = signals_sample_reversed_h[trial_n]
    country_idx = country_idxs_reversed_h[trial_n]
    trial_composition = [sounds_reversed[x][y].resize(0.6) for x, y in zip(signals, country_idx)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range, upper_freq=upper_freq)
    revspeech_data_h["sound"].append(sum(trial_composition))
    revspeech_data_h["coverage"].append(percentage_filled)

coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# plot results
sns.lineplot(x=revspeech_h.solution, y=revspeech_data_h["coverage"], errorbar="se", label="reversed speech regenerated")
sns.lineplot(x=clearspeech_h.solution, y=coverage.loc["clearspeech_h"]["coverage"], errorbar="se", label="clear speech")
sns.lineplot(x=revspeech_h.solution, y=coverage.loc["revspeech_h"]["coverage"], errorbar="se", label="clear speech")
plt.title("Spectral Density Horizontal")
plt.xlabel("Sound source perceptional discrepancy")
plt.ylabel("coverage")

