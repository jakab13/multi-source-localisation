from analysis.dataframe_generation.compile_task_results import df_la, df_su, df_nj
import datetime
import pandas as pd
import scipy
import numpy as np
import slab
from analysis.dataframe_generation.utils import pick_speakers
from stimuli.tts_models import models
import pathlib
import os
import ast

tts_model = models["tts_models"][16]
DIR = pathlib.Path(os.getcwd())
country_converter = {
    0: "Belgium",
    1: "Britain",
    2: "Congo",
    3: "Cuba",
    4: "Japan",
    5: "Mali",
    6: "Oman",
    7: "Peru",
    8: "Sudan",
    9: "Syria",
    10: "Togo",
    11: "Tonga",
    12: "Yemen",
}

speaker_rewiring_datetime = pd.to_datetime(datetime.datetime(2024, 7, 9, 12, 10, 6, 570176))

excluded_subs = ["sub_01", "sub_02", "sub_04", "sub_05", "sub_07", "sub_09", "sub_12", "sub_15", "sub_16", "sub_17", "sub_19", "sub_111", "sub_108"]
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


df_la.loc[(df_la.plane == "horizontal") & (df_la.stim_loc == 17.5) & (df_la.ff_wiring == "modified"), "stim_loc"] = 8.75
df_su.loc[(df_su.plane == "vertical") & (df_su.masker_speaker_loc == 0.0), "threshold"] += 3
df_su.loc[(df_su.plane == "horizontal") & (df_su.masker_speaker_loc == 17.5) & (df_su.ff_wiring == "modified"), "masker_speaker_loc"] = 8.75

q = (df_nj.subject_id == "sub_108") & (df_nj.plane == "horizontal")
df_nj.loc[q, "resp_number"] = df_nj.loc[q, "resp_number"] - 1


def get_su_norm_to_max_threshold(row):
    max_threshold_idx = df_su[(df_su.subject_id == row.subject_id) & (df_su.plane == row.plane)]["threshold"].idxmax()
    collocated_threshold = None
    if row.plane == "horizontal" or row.plane == "vertical":
        collocated_threshold = df_su[
            (df_su.subject_id == row.subject_id) & (df_su.plane == row.plane) & (df_su.masker_speaker_loc == 0.0)]["threshold"]
    elif row.plane == "distance":
        collocated_threshold = df_su[
            (df_su.subject_id == row.subject_id) & (df_su.plane == row.plane) & (df_su.masker_speaker_loc == 7.0)]["threshold"]
    max_threshold = df_su.loc[max_threshold_idx]["threshold"]
    # normed_threshold = row["threshold"] - max_threshold
    collocated_threshold = collocated_threshold.values[0] if collocated_threshold.values.size != 0 else max_threshold
    normed_threshold = row["threshold"] - collocated_threshold
    return normed_threshold


df_su["normed_threshold"] = df_su.apply(lambda x: get_su_norm_to_max_threshold(x), axis=1)

df_la["error_loc"] = df_la["stim_loc"] - df_la["resp_loc"]
df_la["abs_error_loc"] = df_la["error_loc"].abs()

df_nj["error"] = df_nj["stim_number"] - df_nj["resp_number"]
df_nj["abs_error"] = df_nj["error"].abs()

# sub_106 (Jacqueline): horizontal speaker 15 didn’t work
q = (df_la.subject_id == "sub_106") & (df_la.plane == "horizontal") & (df_la.stim_loc == -17.5)
df_la = df_la[~q]

q = (df_su.subject_id == "sub_106") & (df_su.plane == "horizontal") & (df_su.masker_speaker_loc == -17.5)
df_su = df_su[~q]

# q = (df_nj.subject_id == "sub_106") & (df_nj.plane == "horizontal")
# df_nj = df_nj[~q]

def get_spectral_coverage(row):
    trial_dur = 0.6
    p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
    upper_freq = 11000  # upper frequency limit that carries information for speech
    dyn_range = 65
    talker_ids = row["stim_talker_ids"]
    speaker_ids = row["speaker_ids"]
    country_ids = row["stim_country_ids"]
    country_ids = ast.literal_eval(country_ids) if type(country_ids) == str else country_ids
    talker_ids = ast.literal_eval(talker_ids) if type(talker_ids) == str else talker_ids
    setup = "cathedral" if row["plane"] == "distance" else "freefield"
    stim_type = "countries_forward" if row["stim_type"] == "forward" else "countries_reversed"
    trial_composition = list()
    if setup == "cathedral":
        sub_DIR = DIR / "samples" / "TTS" / f"tts-{stim_type}_cathedral_n13_resamp_24414"
    else:
        if stim_type == "countries_forward":
            sub_DIR = DIR / "samples" / "TTS" / f"tts-countries_n13_resamp_24414"
        else:
            sub_DIR = DIR / "samples" / "TTS" / f"tts-countries-reversed_n13_resamp_24414"
    for n in range(len(talker_ids)):
        talker_id = talker_ids[n]
        talker_id = "p" + talker_id if setup != "cathedral" else talker_id
        sex = tts_model["speaker_genders"][talker_id]
        country_id = country_ids[n]
        if setup == "cathedral":
            speaker_ids = ast.literal_eval(speaker_ids)
            distances = [float(s + 2) for s in speaker_ids]
            distance = distances[n]
            # country_id = country_id + "_reversed" if "reversed" in stim_type else country_id + "_"
            file_name = f"sound-talker-{talker_id}_sex-{sex}_text-{country_id}_mgb-level-27.5_distance-{distance}.wav"
        else:
            country_name = country_converter[country_id]
            if stim_type == "countries_forward":
                file_name = f"talker-{talker_id}_sex-{sex}_text-_{country_name}_.wav"
            else:
                file_name = f"talker-{talker_id}_sex-{sex}_text-_{country_name}_reversed.wav"
        signal = slab.Sound(sub_DIR / file_name)
        trial_composition.append(signal.resize(trial_dur))
        sound = sum(trial_composition)
        sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
        sound = sound.resample(24414)
        freqs, times, power = sound.spectrogram(show=False)
        power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
        power = power[freqs < upper_freq, :]
        dB_max = power.max()
        dB_min = dB_max - dyn_range
        interval = power[np.where((power > dB_min) & (power < dB_max))]
        percentage_filled = interval.shape[0] / power.flatten().shape[0]
        return percentage_filled

df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_spectral_coverage(row), axis=1)