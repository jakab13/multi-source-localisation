from analysis.dataframe_generation.compile_task_results import df_la, df_su, df_nj
import datetime
import pandas as pd
import scipy
import numpy as np
from analysis.dataframe_generation.utils import pick_speakers
from stimuli.tts_models import models
import pathlib
import os

tts_model = models["tts_models"][16]
DIR = pathlib.Path(os.getcwd())

speaker_rewiring_datetime = pd.to_datetime(datetime.datetime(2024, 7, 9, 12, 10, 6, 570176))

excluded_subs = ["sub_01", "sub_02", "sub_04", "sub_05", "sub_07", "sub_09", "sub_12", "sub_15", "sub_16", "sub_17",
                 "sub_19",
                 "sub_103_collocated",
                 # "sub_116",
                 # "sub_119",
                 # "sub_120",
                 # "sub_106"
                 ]
for e in excluded_subs:
    df_nj = df_nj[~(df_nj.subject_id == e)]
    df_la = df_la[~(df_la.subject_id == e)]
    df_su = df_su[~(df_su.subject_id == e)]


columns_performance = ["subject_id", "plane", "stim_type", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
df_performance_nj = pd.DataFrame(columns=columns_performance)

for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        for stim_type in df_nj.stim_type.unique():
            df_curr = df_nj[(df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_type == stim_type)]
            df_curr = df_curr.dropna(subset=["stim_number", "resp_number"])
            if len(df_curr) > 0:
                x = df_curr.stim_number.values.astype(float)
                y = df_curr.resp_number.values.astype(float)
                reg = scipy.stats.linregress(x, y)
                df_performance_nj.loc[len(df_performance_nj)] = [subject_id, plane, stim_type, reg.slope, reg.intercept, reg.rvalue, reg.pvalue, reg.stderr, reg.intercept_stderr]


columns_performance = ["subject_id", "plane", "stim_type", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
df_performance_la = pd.DataFrame(columns=columns_performance)
for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        for stim_type in df_la.stim_type.unique():
            df_curr = df_la[(df_la.subject_id == subject_id) & (df_la.plane == plane) & (df_la.stim_type == stim_type)]
            if len(df_curr) > 0:
                x = df_curr.stim_loc.values.astype(float)
                y = df_curr.resp_loc.values.astype(float)
                reg = scipy.stats.linregress(x, y)
                if reg.slope < 0 or reg.pvalue > 0.05:
                    continue
                else:
                    slope = 1 - np.abs(1 - reg.slope)
                    df_performance_la.loc[len(df_performance_la)] = [subject_id, plane, stim_type, slope, reg.intercept, reg.rvalue, reg.pvalue, reg.stderr, reg.intercept_stderr]

columns_slopes = ["subject_id", "plane", "stim_type_la", "stim_type_nj", "la_slope", "nj_slope"]
df_slopes = pd.DataFrame(columns=columns_slopes)
for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        for stim_type_la in df_la.stim_type.unique():
            df_curr_la = df_performance_la[(df_performance_la.subject_id == subject_id) & (df_performance_la.plane == plane) & (df_performance_la.stim_type == stim_type_la)]
            for stim_type_nj in df_nj.stim_type.unique():
                df_curr_nj = df_performance_nj[(df_performance_nj.subject_id == subject_id) & (df_performance_nj.plane == plane) & (df_performance_nj.stim_type == stim_type_nj)]
                if len(df_curr_la) > 0 and len(df_curr_nj) > 0:
                    la_slope = df_curr_la.slope.iloc[0]
                    nj_slope = df_curr_nj.slope.iloc[0]
                    df_slopes.loc[len(df_slopes)] = [subject_id, plane, stim_type_la, stim_type_nj, la_slope, nj_slope]

df_nj["la_slope"] = 0.0
df_nj["nj_slope"] = 0.0
df_la["la_slope"] = 0.0
df_la["la_intercept"] = 0.0

for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        for stim_type_nj in df_nj.stim_type.unique():
            nj_slope = df_performance_nj[(df_performance_nj.subject_id == subject_id) &
                                         (df_performance_nj.plane == plane) &
                                         (df_performance_nj.stim_type == stim_type_nj)]["slope"].mean()
            q_curr_nj = (df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_type == stim_type_nj)
            df_nj.loc[q_curr_nj, "nj_slope"] = nj_slope

for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        nj_slope = df_performance_nj[(df_performance_nj.subject_id == subject_id) &
                                     (df_performance_nj.plane == plane)]["slope"].mean()
        for stim_type_la in df_la.stim_type.unique():
            la_slope = df_performance_la[(df_performance_la.subject_id == subject_id) &
                                         (df_performance_la.plane == plane) &
                                         (df_performance_la.stim_type == stim_type_la)]["slope"].mean()
            la_intercept = df_performance_la[(df_performance_la.subject_id == subject_id) &
                                         (df_performance_la.plane == plane) &
                                         (df_performance_la.stim_type == stim_type_la)]["intercept"].mean()
            q_curr_la = (df_la.subject_id == subject_id) & (df_la.plane == plane) & (df_la.stim_type == stim_type_la)
            rmse = ((df_la[q_curr_la]["stim_loc"] - df_la[q_curr_la]["resp_loc"]) ** 2).mean() ** .5
            df_la.loc[q_curr_la, "rmse"] = rmse
            df_la.loc[q_curr_la, "la_slope"] = la_slope
            df_la.loc[q_curr_la, "la_intercept"] = la_intercept
            df_la.loc[q_curr_la, "nj_slope"] = nj_slope



df_la = df_la[~((df_la.plane == "vertical") & (df_la.stim_loc > 50))]
df_la = df_la[~((df_la.plane == "vertical") & (df_la.stim_loc < -50))]
df_la = df_la[~((df_la.plane == "horizontal") & (df_la.resp_loc > 60))]
df_la = df_la[~((df_la.plane == "horizontal") & (df_la.resp_loc < -60))]

df_nj = df_nj.dropna(subset=["stim_number", "resp_number"])

df_nj = df_nj[df_nj.resp_number < 7]

dfs = [df_la, df_su, df_nj]


def get_ff_wiring(row):
    datetime_c = row["datetime_c"]
    delta = pd.Timedelta(datetime_c - speaker_rewiring_datetime).days
    if delta < 0 and row["round"] == 2:
        wiring = "modified"
    else:
        wiring = "original"
    return wiring


def get_speaker_loc(row):
    plane = row["plane"]
    loc = None
    if plane == "distance":
        loc = row["masker_speaker_loc"]
    else:
        speaker_id = row["masker_speaker_id"]
        if speaker_id == 23:
            loc = 0
        else:
            speaker = pick_speakers(speaker_id)[0]
            if plane == "horizontal":
                loc = speaker.azimuth
            elif plane == "vertical":
                loc = speaker.elevation
    return loc

for df in dfs:
    df["ff_wiring"] = df.apply(lambda x: get_ff_wiring(x), axis=1)

df_su["masker_speaker_loc"] = df_su.apply(lambda x: get_speaker_loc(x), axis=1)


df_la.loc[(df_la.plane == "horizontal") & (df_la.stim_loc == 17.5) & (df_la.ff_wiring == "modified"), "threshold"] = None
df_su.loc[(df_su.plane == "vertical") & (df_su.masker_speaker_loc == 0.0) & (df_su.ff_wiring == "modified"), "threshold"] += 3
# df_su.loc[(df_su.plane == "horizontal") & (df_su.masker_speaker_loc == 17.5) & (df_su.ff_wiring == "modified"), "masker_speaker_loc"] = 8.75
df_su.loc[(df_su.plane == "horizontal") & (df_su.masker_speaker_loc == 17.5) & (df_su.ff_wiring == "modified"), "masker_speaker_loc"] = None


def equalise_su_collocated(row):
    output = row.threshold
    if row.plane == "vertical":
        q = (df_su.subject_id == row.subject_id) & (df_su.plane == "horizontal") & (df_su.masker_speaker_loc == 0.0)
        horizontal_collocated_threshold = df_su[q]["threshold"]
        horizontal_collocated_threshold = horizontal_collocated_threshold.values[0] if horizontal_collocated_threshold.values.size != 0 else None
        if horizontal_collocated_threshold is not None and horizontal_collocated_threshold > row.threshold:
            output = horizontal_collocated_threshold
    return output


# df_su["threshold"] = df_su.apply(lambda row: equalise_su_collocated(row), axis=1)


def get_su_norm_to_max_threshold(row):
    # max_threshold_idx = df_su[(df_su.subject_id == row.subject_id) & (df_su.plane == row.plane)]["threshold"].idxmax()
    collocated_threshold = None
    if row.plane == "horizontal" or row.plane == "vertical":
        collocated_threshold = df_su[
            (df_su.plane == row.plane) & (df_su.masker_speaker_loc == 0.0)]["threshold"]
    elif row.plane == "distance":
        collocated_threshold = df_su[
            (df_su.plane == row.plane) & (df_su.masker_speaker_loc == 7.0)]["threshold"]
    # max_threshold = df_su.loc[max_threshold_idx]["threshold"]
    # normed_threshold = row["threshold"] - max_threshold
    collocated_threshold = collocated_threshold.values.mean()
    normed_threshold = row["threshold"] - collocated_threshold
    return normed_threshold


df_su["normed_threshold"] = df_su.apply(lambda x: get_su_norm_to_max_threshold(x), axis=1)


def normalise_su_threshold(row):
    q = (df_su.plane == row.plane)
    mean_threshold_per_plane = df_su[q]["threshold"].mean()
    output = row.threshold - mean_threshold_per_plane
    return output


# df_su["threshold"] = df_su.apply(lambda row: normalise_su_threshold(row), axis=1)


df_la["error_loc"] = df_la["resp_loc"] - df_la["stim_loc"]
df_la["abs_error_loc"] = df_la["error_loc"].abs()

df_nj["error"] = df_nj["resp_number"] - df_nj["stim_number"]
df_nj["abs_error"] = df_nj["error"].abs()

df_la["stim_loc_abs"] = df_la["stim_loc"].abs()
df_la[df_la.plane == "distance"]["stim_loc_abs"] = (df_la[df_la.plane == "distance"]["stim_loc"] - 7).abs()
df_su["masker_speaker_loc_abs"] = df_su["masker_speaker_loc"].abs()
df_su.loc[df_su.plane == "distance", "masker_speaker_loc_abs"] = (df_su[df_su.plane == "distance"]["masker_speaker_loc_abs"] - 7).abs()

# sub_109 (Elia): horizontal speaker 15 didn’t work
q = (df_la.subject_id == "sub_109") & (df_la.plane == "horizontal") & (df_la.stim_loc == -17.5)
df_la = df_la[~q]

q = (df_su.subject_id == "sub_109") & (df_su.plane == "horizontal") & (df_su.masker_speaker_loc == -17.5)
df_su = df_su[~q]

q = (df_la.subject_id == "sub_111") & (df_la.plane == "horizontal") & (df_la.stim_loc == -17.5)
df_la = df_la[~q]

q = (df_su.subject_id == "sub_111") & (df_su.plane == "horizontal") & (df_su.masker_speaker_loc == -17.5)
df_su = df_su[~q]

q = (df_la.subject_id == "sub_116") & (df_la.plane == "distance") & (df_la.stim_type == "babble")
df_la = df_la[~q]

# q = (df_nj.subject_id == "sub_106") & (df_nj.plane == "horizontal")
# df_nj = df_nj[~q]


def get_speaker_locs(row):
    speaker_ids = row["speaker_ids"]
    plane = row["plane"]
    speaker_locs = list()
    if plane == "distance":
        speaker_locs = [float(s + 2) for s in speaker_ids]
    elif plane == "horizontal":
        speaker_locs = [pick_speakers(spk)[0].azimuth for spk in speaker_ids]
    elif plane == "vertical":
        speaker_locs = [pick_speakers(spk)[0].elevation for spk in speaker_ids]
    return speaker_locs


def get_speaker_loc_mean(row):
    speaker_locs = get_speaker_locs(row)
    speaker_loc_mean = np.asarray(speaker_locs).mean()
    return speaker_loc_mean


def get_speaker_loc_std(row):
    speaker_locs = get_speaker_locs(row)
    speaker_loc_std = np.asarray(speaker_locs).std()
    return speaker_loc_std


def get_relative_speaker_loc_mean(row):
    speaker_loc_mean = get_speaker_loc_mean(row)
    df_curr = df_nj[(df_nj["plane"] == row["plane"]) & (df_nj["stim_number"] == row["stim_number"])]
    cond_min = df_curr["stim_loc_mean"].min()
    cond_max = df_curr["stim_loc_mean"].max()
    m = scipy.interpolate.interp1d([cond_min, cond_max], [0, 1])
    relative_speaker_loc_mean = float(m(speaker_loc_mean))
    return relative_speaker_loc_mean


def get_relative_speaker_loc_std(row):
    speaker_loc_std = get_speaker_loc_std(row)
    df_curr = df_nj[(df_nj["plane"] == row["plane"]) & (df_nj["stim_number"] == row["stim_number"])]
    cond_min = df_curr["stim_loc_std"].min()
    cond_max = df_curr["stim_loc_std"].max()
    m = scipy.interpolate.interp1d([cond_min, cond_max], [0, 1])
    relative_speaker_loc_mean = float(m(speaker_loc_std))
    return relative_speaker_loc_mean


df_nj["stim_loc_mean"] = df_nj.apply(lambda row: get_speaker_loc_mean(row), axis=1)
df_nj["stim_loc_std"] = df_nj.apply(lambda row: get_speaker_loc_std(row), axis=1)
df_nj["stim_loc_mean_relative"] = df_nj.apply(lambda row: get_relative_speaker_loc_mean(row), axis=1)
df_nj["stim_loc_std_relative"] = df_nj.apply(lambda row: get_relative_speaker_loc_std(row), axis=1)

df_nj["stim_loc_mean_binned"] = pd.cut(df_nj["stim_loc_mean_relative"], 5)
df_nj["stim_loc_std_binned"] = pd.cut(df_nj["stim_loc_std_relative"], 5)

