from analysis.utils.plotting import *
from analysis.utils.math import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from labplatform.config import get_config

sns.set_theme()

# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfv.reversed_speech.ffill()
clearspeech_h = dfv[np.where(filled_h == False, True, False)]  # True where reversed_speech is False

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
signals_sample_clear_h = clearspeech_h.signals_sample  # talker IDs
country_idxs_clear_h = clearspeech_h.country_idxs  # indices of the country names from a talker

# variance of dynamic range effect
variances = dict()
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

# extract horizontal data
for dyn_range in range(105, 112):
    observations = list()
    variances[dyn_range] = list()
    for trial_n in range(len(clearspeech_h)):
        signals = signals_sample_clear_h[trial_n]
        country_idx = country_idxs_clear_h[trial_n]
        trial_composition = [sounds_clear[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
        sound = sum(trial_composition)
        _, _, power = sound.spectrogram(show=False)
        power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
        dB_max = power.max()
        dB_min = dB_max - dyn_range
        interval = power[np.where((power > dB_min) & (power < dB_max))]
        percentage_filled = interval.shape[0] / power.flatten().shape[0]
        observations.append(percentage_filled)
    variances[dyn_range].append(variance(observations))

clearspeech_data_h = dict(sound=[], coverage=[])
dyn_range = 109  # highest variance
for trial_n in range(len(clearspeech_h)):
    signals = signals_sample_clear_h[trial_n]
    country_idx = country_idxs_clear_h[trial_n]
    trial_composition = [sounds_clear[x][y].resize(1.0) for x, y in zip(signals, country_idx)]
    sound = sum(trial_composition)
    _, _, power = sound.spectrogram(show=False)
    power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
    dB_max = power.max()
    dB_min = dB_max - dyn_range
    interval = power[np.where((power > dB_min) & (power < dB_max))]
    percentage_filled = interval.shape[0] / power.flatten().shape[0]
    clearspeech_data_h["sound"].append(sound)
    clearspeech_data_h["coverage"].append(percentage_filled)

drop_idxs = [175, 1958]
response = clearspeech_h.response.drop(
    [clearspeech_h.response.index[drop_idxs[0]], clearspeech_h.response.index[drop_idxs[1]]])
solution = clearspeech_h.solution.drop(
    [clearspeech_h.solution.index[drop_idxs[0]], clearspeech_h.solution.index[drop_idxs[1]]])

xticks = clearspeech_h.solution - clearspeech_h.response
sns.lineplot(x=xticks.abs(), y=clearspeech_data_h["coverage"])
plt.title("Spectral Coverage Clearspeech Horizontal")
plt.ylabel("coverage")
