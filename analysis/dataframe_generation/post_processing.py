from analysis.dataframe_generation.compile_task_results import df_la, df_su, df_nj
import datetime
import pandas as pd
import scipy
from analysis.dataframe_generation.utils import pick_speakers

speaker_rewiring_datetime = pd.to_datetime(datetime.datetime(2024, 7, 9, 12, 10, 6, 570176))

excluded_subs = ["sub_01", "sub_02", "sub_04", "sub_07", "sub_09", "sub_12", "sub_15", "sub_16", "sub_17", "sub_19", "sub_106"]
for e in excluded_subs:
    df_nj = df_nj[~(df_nj.subject_id == e)]
    df_la = df_la[~(df_la.subject_id == e)]


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

'''
columns_performance = ["subject_id", "plane", "stim_type", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
df_performance_la = pd.DataFrame(columns=columns_performance)
for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        for stim_type in df_la.stim_type.unique():
            df_curr = df_la[(df_la.subject_id == subject_id) & (df_la.plane == plane) & (df_la.stim_type == stim_type)]
            if len(df_curr) > 0:
                if plane == "v":
                    x = df_curr.stim_ele.values.astype(float)
                    y = df_curr.resp_ele.values.astype(float)
                elif plane == "h":
                    x = df_curr.stim_azi.values.astype(float)
                    y = df_curr.resp_azi.values.astype(float)
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
'''

df_nj["la_slope"] = 0.0
df_nj["nj_slope"] = 0.0

for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        for stim_type_nj in df_nj.stim_type.unique():
            nj_slope = df_performance_nj[(df_performance_nj.subject_id == subject_id) &
                                         (df_performance_nj.plane == plane) &
                                         (df_performance_nj.stim_type == stim_type_nj)]["slope"].mean()
            q_curr_nj = (df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_type == stim_type_nj)
            df_nj.loc[q_curr_nj, "nj_slope"] = nj_slope


df_la = df_la[~((df_la.plane == "v") & (df_la.stim_loc > 50))]

df_nj = df_nj.dropna(subset=["stim_number", "resp_number"])

df_nj = df_nj[df_nj.resp_number < 7]

dfs = [df_la, df_su, df_nj]


def get_ff_wiring(row):
    datetime_c = row["datetime_c"]
    delta = pd.Timedelta(datetime_c - speaker_rewiring_datetime).seconds
    wiring = "old" if delta > 0 else "new"
    return wiring


def get_speaker_loc(row):
    plane = row["plane"]
    loc = None
    if plane == "distance":
        loc = row["masker_speaker_loc"]
    else:
        speaker_id = row["masker_speaker_id"]
        speaker = pick_speakers(speaker_id)[0]
        if plane == "horizontal":
            loc = speaker.azimuth
        elif plane == "vertical":
            loc = speaker.elevation
    return loc


for df in dfs:
    df["ff_wiring"] = df.apply(lambda x: get_ff_wiring(x), axis=1)

df_su["masker_speaker_loc"] = df_su.apply(lambda x: get_speaker_loc(x), axis=1)
df_la.loc[(df_la.plane == "horizontal") & (df_la.stim_loc == 17.5) & (df_la.ff_wiring == "old"), "stim_loc"] = 8.75
df_su.loc[(df_su.plane == "horizontal") & (df_su.masker_speaker_loc == 17.5) & (df_su.ff_wiring == "old"), "masker_speaker_loc"] = 8.75

q = (df_nj.subject_id == "sub_108") & (df_nj.plane == "horizontal")
df_nj.loc[q, "resp_number"] = df_nj.loc[q, "resp_number"] - 1

# TODO: Correct centre speaker threshold by 3dB until speaker_rewiring_datetime

