from analysis.utils.plotting import *
import os
import seaborn as sns
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

# get sub ids
sub_ids = extract_subject_ids_from_dataframe(dfh)

# get talker files
talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds_clear = pkl.load(files)

talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_reversed.pkl")
with open(talker_files_path, "rb") as files:
    sounds_reversed = pkl.load(files)


# get info from trials horizontal
speakers_sample_clear_h = clearspeech_h.speakers_sample  # indices of speakers from speakertable
signals_sample_clear_h = clearspeech_h.signals_sample  # talker IDs
country_idxs_clear_h = clearspeech_h.country_idxs  # indices of the country names from a talker

speakers_sample_reversed_h = revspeech_h.speakers_sample  # indices of speakers from speakertable
signals_sample_reversed_h = revspeech_h.signals_sample  # talker IDs
country_idxs_reversed_h = revspeech_h.country_idxs  # indices of the country names from a talker

# get info from trials vertical
speakers_sample_clear_v = clearspeech_v.speakers_sample  # indices of speakers from speakertable
signals_sample_clear_v = clearspeech_v.signals_sample  # talker IDs
country_idxs_clear_v = clearspeech_v.country_idxs  # indices of the country names from a talker

speakers_sample_reversed_v = revspeech_v.speakers_sample  # indices of speakers from speakertable
signals_sample_reversed_v = revspeech_v.signals_sample  # talker IDs
country_idxs_reversed_v = revspeech_v.country_idxs  # indices of the country names from a talker

# load dataframe containing coverage
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# plot results
sns.lineplot(x=clearspeech_h.solution, y=coverage.loc["clearspeech_h"]["coverage"], errorbar="se", label="clear speech")
sns.lineplot(x=revspeech_h.solution, y=coverage.loc["revspeech_h"]["coverage"], errorbar="se", label="reversed speech")
plt.title("Spectral Density Horizontal")
plt.xlabel("Sound source perceptional discrepancy")
plt.ylabel("coverage")

# vertical
sns.lineplot(x=clearspeech_v.solution, y=coverage.loc["clearspeech_v"]["coverage"], errorbar="se", label="clear speech")
sns.lineplot(x=revspeech_v.solution, y=coverage.loc["revspeech_v"]["coverage"], errorbar="se", label="reversed speech")
plt.title("Spectral Density Vertical")
plt.xlabel("Solution")
plt.ylabel("coverage")

