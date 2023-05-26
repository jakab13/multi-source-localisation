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
filled_v = dfh.reversed_speech.ffill()
revspeech_v = dfh[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
clearspeech_v = dfh[np.where(filled_v==False, True, False)]  # True where reversed_speech is False

# get sub ids
sub_ids = extract_subject_ids_from_dataframe(dfh)

# get talker files
talker_files_path = os.path.join(get_config("DATA_ROOT"), "MSL", "used_sounds_pkl", "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds_clear = pkl.load(files)

talker_files_path = os.path.join(get_config("DATA_ROOT"), "MSL", "used_sounds_pkl", "numjudge_talker_files_reversed.pkl")
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

# extract horizontal data
data_h = dict(clearspeech=dict(sound=[], fwhm=[]), revspeech=dict(sound=[], fwhm=[]))
for trial_n in range(clearspeech_h.__len__()):
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    signals = signals_sample_clear_h[trial_n]
    country_idx = country_idxs_clear_h[trial_n]
    trial_composition = [sounds_clear[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
    for signal in trial_composition:
        sound += signal
    fwhm = sound.spectral_feature("fwhm")[0]
    data_h["clearspeech"]["sound"].append(sound)
    # data["rms"].append(rms)
    # data["auc"].append(rms_auc)
    data_h["clearspeech"]["fwhm"].append(fwhm)

    # reversed
    sound = slab.Sound(data=np.zeros(int(48828)), samplerate=48828)
    signals = signals_sample_reversed_h[trial_n]
    country_idx = country_idxs_reversed_h[trial_n]
    trial_composition = [sounds_reversed[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
    for signal in trial_composition:
        sound += signal
    fwhm = sound.spectral_feature("fwhm")[0]
    data_h["revspeech"]["sound"].append(sound)
    # data["rms"].append(rms)
    # data["auc"].append(rms_auc)
    data_h["revspeech"]["fwhm"].append(fwhm)
    # y = sound.data
    # S = librosa.magphase(librosa.stft(y.flatten()))[0]
    # rms = librosa.feature.rms(S=S)[0]
    # times = np.linspace(0, sound.duration, rms.flatten().__len__())
    # rms_auc = auc(times, rms)

data_v = dict(clearspeech=dict(sound=[], fwhm=[]), revspeech=dict(sound=[], fwhm=[]))
for trial_n in range(clearspeech_v.__len__()):
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    signals = signals_sample_clear_v[trial_n]
    country_idx = country_idxs_clear_v[trial_n]
    trial_composition = [sounds_clear[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
    for signal in trial_composition:
        sound += signal
    fwhm = sound.spectral_feature("fwhm")[0]
    data_v["clearspeech"]["sound"].append(sound)
    # data["rms"].append(rms)
    # data["auc"].append(rms_auc)
    data_v["clearspeech"]["fwhm"].append(fwhm)
    # reversed
    sound = slab.Sound(data=np.zeros(int(48828)), samplerate=48828)
    signals = signals_sample_reversed_v[trial_n]
    country_idx = country_idxs_reversed_v[trial_n]
    trial_composition = [sounds_reversed[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
    for signal in trial_composition:
        sound += signal
    fwhm = sound.spectral_feature("fwhm")[0]
    data_v["revspeech"]["sound"].append(sound)
    # data["rms"].append(rms)
    # data["auc"].append(rms_auc)
    data_v["revspeech"]["fwhm"].append(fwhm)
    # y = sound.data
    # S = librosa.magphase(librosa.stft(y.flatten()))[0]
    # rms = librosa.feature.rms(S=S)[0]
    # times = np.linspace(0, sound.duration, rms.flatten().__len__())
    # rms_auc = auc(times, rms)

# plot results
sns.lineplot(x=clearspeech_h.solution, y=data_h["clearspeech"]["fwhm"], errorbar="se", label="clear speech")
sns.lineplot(x=revspeech_h.solution, y=data_h["revspeech"]["fwhm"], errorbar="se", label="reversed speech")
plt.title("Spectral Density Horizontal")
plt.xlabel("Solution")
plt.ylabel("FWHM")

# vertical
sns.lineplot(x=clearspeech_v.solution, y=data_v["clearspeech"]["fwhm"], errorbar="se", label="clear speech")
sns.lineplot(x=revspeech_v.solution, y=data_v["revspeech"]["fwhm"], errorbar="se", label="reversed speech")
plt.title("Spectral Density Vertical")
plt.xlabel("Solution")
plt.ylabel("FWHM")

# extract rms for all sounds
fig, ax = plt.subplots(2, 4, sharex=True)
ax = ax.flatten()
for axis, key in enumerate(sounds_clear):
    ax[axis].set_title(key)
    ax[axis].set_xlabel("Time [s]")
    ax[axis].set_ylabel("RMS Energy")
    for sound in sounds_clear[key]:
        y = sound.data
        S = librosa.magphase(librosa.stft(y.flatten()))[0]
        rms = librosa.feature.rms(S=S)[0]
        times = np.linspace(0, sound.duration, rms.flatten().__len__())
        ax[axis].semilogy(times, rms)


# RMS
y1 = trial_composition[0].data
S1 = librosa.magphase(librosa.stft(y1.flatten()))[0]
y2 = trial_composition[1].data
S2 = librosa.magphase(librosa.stft(y2.flatten()))[0]

rms1 = librosa.feature.rms(S=S1)[0]
rms2 = librosa.feature.rms(S=S2)[0]
times = np.linspace(0, trial_composition[0].duration, rms1.flatten().__len__())

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].semilogy(times, rms1.flatten()+rms2.flatten(), label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()

