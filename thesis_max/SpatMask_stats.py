from analysis.utils.misc import *
import os
from labplatform.config import get_config
from Speakers.speaker_config import SpeakerArray
from scipy import stats
from statsmodels.stats.multitest import multipletests


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
filepath = os.path.join(basedir, "FREEFIELD_speakers.txt")
spks = SpeakerArray(file=filepath)
spks.load_speaker_table()
spks = spks.pick_speakers(picks=speaker_ids)
coords = [[x.azimuth, x.elevation] for x in spks]
azi = [x[0] for x in coords]
ele = [x[1] for x in coords]

# get horizontal thresholds
threshs_h = []
for i, sub in enumerate(sub_ids_h):
    spks_order = dfh.loc[sub].sequence.dropna()[0]["trials"].copy()
    # spks_order.remove(spks_order[-1])
    reversal_ints = [x["reversal_intensities"] for x in dfh.loc[sub].stairs.dropna()]
    # reversal_ints.pop(0)
    threshs = pd.DataFrame()
    for i, spk in enumerate(spks_order):
        threshs[azi[spk - 1]] = reversal_ints[i]
        # threshs[azi[spk]] = reversal_ints[i]
    threshs_h.append(threshs.mean())
threshs_all_subjects_h = pd.concat(threshs_h)

# Vertical
speaker_ids = [x for x in range(20, 27) if x != 23]
basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
filepath = os.path.join(basedir, "FREEFIELD_speakers.txt")
spks = SpeakerArray(file=filepath)
spks.load_speaker_table()
spks = spks.pick_speakers(picks=speaker_ids)
coords = [[x.azimuth, x.elevation] for x in spks]
azi = [x[0] for x in coords]
ele = [x[1] for x in coords]


# get vertical thresholds
threshs_v = []
for i, sub in enumerate(sub_ids_v):
    spks_order = dfv.loc[sub].sequence.dropna()[0]["trials"].copy()
    # spks_order.remove(spks_order[-1])
    reversal_ints = [x["reversal_intensities"] for x in dfv.loc[sub].stairs.dropna()]
    # reversal_ints.pop(0)
    threshs = pd.DataFrame()
    for i, spk in enumerate(spks_order):
        threshs[ele[spk - 1]] = reversal_ints[i]
    threshs_v.append(threshs.mean())
threshs_all_subjects_v = pd.concat(threshs_v)


# Make boxplots depending on distance between masker and speaker
# Merge columns
abs_distance_v = pd.DataFrame()
abs_distance_v[12.5] = pd.concat([threshs_all_subjects_v[-12.5],
                                  threshs_all_subjects_v[12.5]], ignore_index=True)
abs_distance_v[25.0] = pd.concat([threshs_all_subjects_v[-25.0],
                                  threshs_all_subjects_v[25.0]], ignore_index=True)
abs_distance_v[37.5] = pd.concat([threshs_all_subjects_v[-37.5],
                                  threshs_all_subjects_v[37.5]], ignore_index=True)

# horizontal
abs_distance_h = pd.DataFrame()
abs_distance_h[17.5] = pd.concat([threshs_all_subjects_h[-17.5],
                                  threshs_all_subjects_h[17.5]], ignore_index=True)
abs_distance_h[35.0] = pd.concat([threshs_all_subjects_h[-35.0],
                                  threshs_all_subjects_h[35.0]], ignore_index=True)
abs_distance_h[52.5] = pd.concat([threshs_all_subjects_h[-52.5],
                                  threshs_all_subjects_h[52.5]], ignore_index=True)

pvalsh = [stats.wilcoxon(abs_distance_h[17.5], abs_distance_h[35.0])[1],
          stats.wilcoxon(abs_distance_h[35.0], abs_distance_h[52.5])[1],
          stats.wilcoxon(abs_distance_h[17.5], abs_distance_h[52.5])[1]]

pvalsv = [stats.ttest_rel(abs_distance_v[12.5], abs_distance_v[25.0])[1],
          stats.ttest_rel(abs_distance_v[12.5], abs_distance_v[37.5])[1],
          stats.ttest_rel(abs_distance_v[25.0], abs_distance_v[37.5])[1]]

method = "bonferroni"
# wilcoxon, bonferroni corrected
print(f"HORIZONTAL: {multipletests(pvalsh, method=method)} \n"
      f"VERTICAL: {multipletests(pvalsv, method=method)}")

# means and standard errors
print(f"### MEANS AND STD ERRORS ### \n"
      f"AZIMUTH: \n"
      f"17.5: {np.mean(abs_distance_h[17.5])} ± {stats.sem(abs_distance_h[17.5])} \n"
      f"35.0: {np.mean(abs_distance_h[35.0])} ± {stats.sem(abs_distance_h[35.0])} \n"
      f"52.5: {np.mean(abs_distance_h[52.5])} ± {stats.sem(abs_distance_h[52.5])} \n"
      f"ELEVATION: \n"
      f"12.5: {np.mean(abs_distance_v[12.5])} ± {stats.sem(abs_distance_v[12.5])} \n"
      f"25.0: {np.mean(abs_distance_v[25.0])} ± {stats.sem(abs_distance_v[25.0])} \n"
      f"37.5: {np.mean(abs_distance_v[37.5])} ± {stats.sem(abs_distance_v[37.5])} \n")

# Page L test for monotony (better stats than wilcoxon)
method = "exact"
print(f"### PAGE TREND TEST ### \n"
      f"AZIMUTH: {stats.page_trend_test(abs_distance_h.abs(), method=method)} \n"
      f"ELEVATION: {stats.page_trend_test(abs_distance_v.abs(), method=method)}")
