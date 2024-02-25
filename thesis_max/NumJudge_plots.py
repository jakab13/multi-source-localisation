from analysis.utils.plotting import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
from labplatform.config import get_config
import pickle as pkl
import string
import scienceplots
plt.style.use("science")
plt.ion()


# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

sub_ids = extract_subject_ids_from_dataframe(dfh)

# divide reversed speech blocks from clear speech
filledv = dfv.reversed_speech.ffill()
revspeechv = dfv[np.where(filledv==True, True, False)]  # True where reversed_speech is True
clearspeechv = dfv[np.where(filledv==False, True, False)]  # True where reversed_speech is False

filledh = dfh.reversed_speech.ffill()
revspeechh = dfh[np.where(filledh==True, True, False)]  # True where reversed_speech is True
clearspeechh = dfh[np.where(filledh==False, True, False)]  # True where reversed_speech is False

# sort by index
revspeechh = revspeechh.sort_index()
clearspeechh = clearspeechh.sort_index()
revspeechv = revspeechv.sort_index()
clearspeechv = clearspeechv.sort_index()

# SPECTRO TEMPORAL COVERAGE
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

layout = """
ab
"""
_, ax = plt.subplot_mosaic(layout, sharex=True, sharey=True, figsize=(6, 3))
sns.lineplot(x=clearspeechh.solution.fillna(0), y=clearspeechh.response.fillna(0), ax=ax["a"],
             err_style="bars", label="Forward speech", palette="magma")
sns.lineplot(x=revspeechh.solution.fillna(0), y=revspeechh.response.fillna(0), ax=ax["a"],
             err_style="bars", label="Reversed speech", palette="magma")

sns.lineplot(x=clearspeechv.solution.fillna(0), y=clearspeechv.response.fillna(0), ax=ax["b"],
             err_style="bars", label="Forward speech", palette="viridis")
sns.lineplot(x=revspeechv.solution.fillna(0), y=revspeechv.response.fillna(0), ax=ax["b"],
             err_style="bars",  label="Reversed speech", palette="viridis")

ax["a"].text(-0.1, 1.0, string.ascii_uppercase[0], transform=ax["a"].transAxes,
               size=20, weight='bold')
ax["b"].text(-0.1, 1.0, string.ascii_uppercase[1], transform=ax["b"].transAxes,
               size=20, weight='bold')
plt.xticks(range(2, 7))
plt.yticks(range(2, 7))


for key in list(ax.keys()):
    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()
    lims = [1.9, 6.1]
    sns.lineplot(x=lims, y=lims, color='grey', linestyle="dashed", ax=ax[key])
    # ax[key].set_title("Azimuth") if key == "a" else ax[key].set_title("Elevation")
    ax[key].set_xlabel("Actual Number Of Sounds")
    ax[key].set_ylabel("Reported Number Of Sounds")

plt.savefig("/home/max/labplatform/plots/MA_thesis/results/numerosity_judgment_performance.png",
            dpi=400, bbox_inches="tight")

layout = """
cd
"""
_, ax = plt.subplot_mosaic(layout, sharex=True, sharey=True, figsize=(6, 3))

# plot results
sns.lineplot(x=clearspeechh.solution, y=coverage.loc["clearspeech_h"]["coverage"], label="Forward speech",
             ax=ax["c"],
             err_style="bars")
sns.lineplot(x=revspeechh.solution, y=coverage.loc["revspeech_h"]["coverage"], label="Reversed speech",
             ax=ax["c"],
             err_style="bars")

# vertical
sns.lineplot(x=clearspeechv.solution, y=coverage.loc["clearspeech_v"]["coverage"], label="Forward speech",
             err_style="bars",
             ax=ax["d"])
sns.lineplot(x=revspeechv.solution, y=coverage.loc["revspeech_v"]["coverage"], label="Reversed speech",
             err_style="bars",
             ax=ax["d"])

for key in list(ax.keys()):
    lims = [[1.9, 6.1], [0.5, 1.0]]
    sns.lineplot(x=lims[0], y=lims[1], color='grey', linestyle="dashed", ax=ax[key])
    # ax[key].set_title("Azimuth") if key == "c" else ax[key].set_title("Elevation")
    ax[key].set_xlabel("Actual Number Of Sounds")
    ax[key].set_ylabel("Spectro-Temporal Coverage")
ax["c"].text(-0.1, 1.0, string.ascii_uppercase[0], transform=ax["c"].transAxes,
               size=20, weight='bold')
ax["d"].text(-0.1, 1.0, string.ascii_uppercase[1], transform=ax["d"].transAxes,
               size=20, weight='bold')

plt.savefig("/home/max/labplatform/plots/MA_thesis/results/spectro_temporal_coverage.png",
            dpi=400, bbox_inches="tight")

# Dynamic Range Distribution
"""
CAREFUL: THE BELOW CODE IS VERY COMPUTATION INTENSE AND MIGHT TAKE AGES FOR RUNNING. SO BE CAREFUL!!!
"""
from analysis.utils.math import spectemp_coverage, variance

signals_sample_clear_h = clearspeechh.signals_sample  # talker IDs
country_idxs_clear_h = clearspeechh.country_idxs  # indices of the country names from a talker

# get talker files
talker_files_path = os.path.join(get_config("SOUND_ROOT"), "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds_clear = pkl.load(files)

upper_freq_lim = 11000
variances = dict()
for dyn_range in range(1, 101):
    observations = list()
    variances[dyn_range] = list()
    for trial_n in range(len(clearspeechh)):
        signals = signals_sample_clear_h[trial_n]
        country_idx = country_idxs_clear_h[trial_n]
        trial_composition = [sounds_clear[x][y].resize(0.6) for x, y in zip(signals, country_idx)]
        coverage = spectemp_coverage(trial_composition, dyn_range, upper_freq_lim)
        observations.append(coverage)
    variances[dyn_range].append(variance(observations))

df = pd.DataFrame()
var_vals = [x[0] for x in list(variances.values())]
df["variance"] = var_vals
df["dyn_range"] = variances.keys()

sns.lineplot(x=df.dyn_range, y=df.variance, marker="o")  # plot results
plt.axvline(x=65,
            ymin=0,
            ymax=1,
            linestyle="--",
            color="black")
plt.xlabel("Dynamic Range Minimum Value [dB]")
plt.ylabel("Spectro-Temporal Coverage Variance")

plt.savefig("/home/max/labplatform/plots/MA_thesis/results/variance_distribution_coverage.png",
            dpi=400, bbox_inches="tight")
