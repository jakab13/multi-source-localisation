from analysis.utils.plotting import *
from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
from Speakers.speaker_config import SpeakerArray
sns.set_theme()


# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "SpatMask"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")
sub_ids_h = extract_subject_ids_from_dataframe(dfh)
sub_ids_v = extract_subject_ids_from_dataframe(dfv)

# get speaker coordinates
speaker_ids = [2, 8, 15, 31, 38, 44]
basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
filepath = os.path.join(basedir, "dome_speakers.txt")
spks = SpeakerArray(file=filepath)
spks.load_speaker_table()
spks = spks.pick_speakers(picks=speaker_ids)
coords = [[x.azimuth, x.elevation] for x in spks]
azi = [x[0] for x in coords]
ele = [x[1] for x in coords]


# make mosaic plot
layout = [["A", "B"],
          ["C", "C"]]

fig, ax = mosaic_plot(layout)

threshs_h = []
for i, sub in enumerate(sub_ids_h):
    spks_order = dfh.loc[sub].sequence.dropna()[0]["trials"].copy()
    spks_order.remove(spks_order[-1])
    reversal_ints = [x["reversal_intensities"] for x in dfh.loc[sub].stairs.dropna()]
    reversal_ints.pop(0)
    threshs = pd.DataFrame()
    for i, spk in enumerate(spks_order):
        threshs[azi[spk - 1]] = reversal_ints[i]
    threshs_h.append(threshs)
threshs_all_subjects_h = pd.concat(threshs_h)

sns.boxplot(data=threshs_h[0], ax=ax["A"])
ax["A"].set_title("Subject 01")
sns.boxplot(data=threshs_h[1], ax=ax["B"])
ax["B"].set_title("Subject 01")
sns.boxplot(data=threshs_all_subjects_h, ax=ax["C"])
ax["C"].set_title("Mean")
fig.suptitle("Vertical")
fig.tight_layout()
plt.show()

# Vertical
speaker_ids = [x for x in range(20, 27) if x != 23]
basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
filepath = os.path.join(basedir, "dome_speakers.txt")
spks = SpeakerArray(file=filepath)
spks.load_speaker_table()
spks = spks.pick_speakers(picks=speaker_ids)
coords = [[x.azimuth, x.elevation] for x in spks]
azi = [x[0] for x in coords]
ele = [x[1] for x in coords]

# make mosaic plot
layout = [["A", "B"],
          ["C", "C"]]

fig, ax = mosaic_plot(layout)

threshs_v = []
for i, sub in enumerate(sub_ids_v[1:]):
    spks_order = dfv.loc[sub].sequence.dropna()[0]["trials"].copy()
    spks_order.remove(spks_order[-1])
    reversal_ints = [x["reversal_intensities"] for x in dfv.loc[sub].stairs.dropna()]
    reversal_ints.pop(0)
    threshs = pd.DataFrame()
    for i, spk in enumerate(spks_order):
        threshs[ele[spk - 1]] = reversal_ints[i]
    threshs_v.append(threshs)
threshs_all_subjects_v = pd.concat(threshs_v)

sns.boxplot(data=threshs_v[0], ax=ax["A"])
ax["A"].set_title("Subject 03")
sns.boxplot(data=threshs_v[1], ax=ax["B"])
ax["B"].set_title("Subject 05")
sns.boxplot(data=threshs_all_subjects_v, ax=ax["C"])
ax["C"].set_title("Mean")
fig.suptitle("Vertical")
fig.tight_layout()
plt.show()

# Make boxplots depending on distance between masker and speaker
# Merge columns
abs_distance_v = pd.DataFrame()
abs_distance_v[12.5] = pd.concat([threshs_all_subjects_v[-12.5],
                                  threshs_all_subjects_v[12.5]])
abs_distance_v[25] = pd.concat([threshs_all_subjects_v[-25.0],
                                threshs_all_subjects_v[25.0]])
abs_distance_v[37.5] = pd.concat([threshs_all_subjects_v[-37.5],
                                  threshs_all_subjects_v[37.5]])

# horizontal
abs_distance_h = pd.DataFrame()
abs_distance_h[17.5] = pd.concat([threshs_all_subjects_h[-17.5],
                                  threshs_all_subjects_h[17.5]])
abs_distance_h[35.0] = pd.concat([threshs_all_subjects_h[-35.0],
                                  threshs_all_subjects_h[35.0]])
abs_distance_h[52.5] = pd.concat([threshs_all_subjects_h[-52.5],
                                  threshs_all_subjects_h[52.5]])


fig, ax = mosaic_plot(layout=[["A"], ["B"]])
ax["A"].sharey(ax["B"])
sns.boxplot(abs_distance_h, ax=ax["A"]).set(title="Horizontal absolute distance all subjects")
sns.boxplot(abs_distance_v, ax=ax["B"]).set(title="Vertical absolute distance all subjects")
fig.tight_layout()
plt.show()
