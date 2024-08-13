import pandas as pd
import slab
import pathlib
import os
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import matplotlib
from stimuli.tts_models import models
from analysis.dataframe_generation.post_processing import df_nj

np.seterr(divide='ignore')

DIR = pathlib.Path(os.getcwd())
tts_model = models["tts_models"][16]

trial_dur = 0.6
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
upper_freq = 11000  # upper frequency limit that carries information for speech


def get_filename(stim_type, talker_id, country_id, distance=None):
    sex = tts_model["speaker_genders"][talker_id]
    if distance is not None:
        filename = f"sound-talker-{talker_id}_sex-{sex}_text-{country_id}_mgb-level-27.5_distance-{distance}.wav"
    else:
        if stim_type == "countries_forward":
            filename = f"talker-{talker_id}_sex-{sex}_text-_{country_id}_.wav"
        else:
            filename = f"talker-{talker_id}_sex-{sex}_text-_{country_id}_reversed.wav"
    return filename


def get_sub_DIR(plane, stim_type):
    if plane == "distance":
        sub_DIR = DIR / "samples" / "TTS" / f"tts-{stim_type}_cathedral_n13_resamp_24414"
    else:
        if stim_type == "countries_forward":
            sub_DIR = DIR / "samples" / "TTS" / f"tts-countries_n13_resamp_24414"
        else:
            sub_DIR = DIR / "samples" / "TTS" / f"tts-countries-reversed_n13_resamp_24414"
    return sub_DIR


def compose_trial(sub_DIR, talker_ids, country_ids, distances, stim_type):
    trial_composition = list()
    for n in range(len(talker_ids)):
        talker_id = talker_ids[n]
        country_id = country_ids[n]
        distance = distances[n] if distances is not None else None
        filename = get_filename(stim_type, talker_id, country_id, distance)
        signal = slab.Sound(sub_DIR / filename)
        if distances is None:
            signal.level -= 10
        trial_composition.append(signal.resize(trial_dur))
    sound = sum(trial_composition)
    sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
    sound = sound.resample(24414)
    sound = sound.aweight()
    return sound


def get_sound(row):
    talker_ids = row["stim_talker_ids"]
    speaker_ids = row["speaker_ids"]
    country_ids = row["stim_country_ids"]
    distances = [float(s + 2) for s in speaker_ids] if row["plane"] == "distance" else None
    stim_type = "countries_forward" if row["stim_type"] == "forward" else "countries_reversed"
    sub_DIR = get_sub_DIR(row["plane"], stim_type)
    sound = compose_trial(sub_DIR, talker_ids, country_ids, distances, stim_type)
    return sound


def get_spectral_data(row):
    sound = slab.Sound(row["sound_data"], samplerate=24414)
    freqs, times, power = sound.spectrogram(show=False)
    return [freqs, times, power]


def get_cochleagram(row, bandwidth):
    sound = get_sound(row)
    fbank = slab.Filter.cos_filterbank(bandwidth=bandwidth, low_cutoff=20,
                                       high_cutoff=upper_freq, samplerate=sound.samplerate)
    subbands = fbank.apply(sound.channel(0))
    envs = subbands.envelope()
    envs.data[envs.data < 1e-9] = 0  # remove small values that cause waring with numpy.power
    envs = envs.data ** (1 / 3)  # apply non-linearity (cube-root compression)
    envs = 10 * np.ma.log10(envs / (2e-5 ** 2))
    print(row.name)
    return envs


def get_spectral_coverage(row):
    envs = row["cochleagram_0.2"]
    dB_min = 84 if row.plane != "distance" else 86
    coverage = np.where(envs < dB_min, 0, 1).sum() / envs.size
    print(row.name, coverage)
    return coverage


# def get_spectral_coverage(row):
#     dB_min = -7 if row["plane"] == "distance" else 4
#     freqs = row["spectral_data"][0]
#     power = row["spectral_data"][2]
#     power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
#     power = power[freqs < upper_freq, :]
#     interval = power[np.where(power > dB_min)]
#     percentage_filled = interval.shape[0] / power.flatten().shape[0]
#     return percentage_filled


def get_sound_level(row):
    sound = slab.Sound(row["sound_data"], samplerate=24414)
    return sound.level


def get_relative_spectral_coverage(row):
    df_curr = df_nj[(df_nj["plane"] == row["plane"]) &
                    (df_nj["stim_number"] == row["stim_number"])]
    cond_min = df_curr["spectral_coverage"].min()
    cond_max = df_curr["spectral_coverage"].max()
    m = scipy.interpolate.interp1d([cond_min, cond_max], [0, 1])
    relative_spectral_coverage = float(m(row["spectral_coverage"]))
    return relative_spectral_coverage


df_nj = df_nj[df_nj["round"] == 2]
df_nj["sound_data"] = df_nj.apply(lambda row: get_sound(row), axis=1)
df_nj["cochleagram_0.2"] = df_nj.apply(lambda row: get_cochleagram(row, 0.2), axis=1)

# df_nj["sound_data"] = df_nj.apply(lambda row: get_sound(row), axis=1)
# df_nj["spectral_coverage_relative"] = df_nj.apply(lambda row: get_relative_spectral_coverage(row), axis=1)

# df_nj["spectral_data"] = df_nj.apply(lambda row: get_spectral_data(row), axis=1)
df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_spectral_coverage(row), axis=1)

for plane in df_nj.plane.unique():
    for stim_number in df_nj.stim_number.unique():
        for stim_type in df_nj.stim_type.unique():
            q = (df_nj.plane == plane) & (df_nj.stim_number == stim_number) & (df_nj.stim_type == stim_type)
            df_nj.loc[q, "spectral_coverage_binned"] = pd.cut(df_nj.loc[q, "spectral_coverage"], 5,
                                                              labels=["lowest", "low", "mid", "high", "highest"])


for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        for stim_number in df_nj.stim_number.unique():
            for stim_type in df_nj.stim_type.unique():
                q = (df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_number == stim_number) & (df_nj.stim_type == stim_type)
                df_curr = df_nj[q]
                if len(df_curr) > 0:
                    x = df_curr.spectral_coverage.values.astype(float)
                    y = df_curr.error.values.astype(float)
                    reg = scipy.stats.linregress(x, y)
                    df_nj.loc[q, "spectral_coverage_slope"] = reg.slope


def calculate_spectral_coverage_threshold():
    df_spectral = pd.DataFrame(columns=["dB_min", "plane", "std"])
    for dB_min in [85, 86, 87, 88, 89, 90]:
        df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_spectral_coverage(row, dB_min), axis=1)
        df_curr = df_nj.groupby(["plane"], as_index=False)["spectral_coverage"].std()
        df_curr = df_curr.rename(columns={"spectral_coverage": "std"})
        df_curr["dB_min"] = dB_min
        df_spectral = pd.concat([df_spectral, df_curr], ignore_index=True)
        print(f"Done with min dB: {dB_min}")
    return df_spectral


cmap = matplotlib.cm.get_cmap('Greys')
df_curr = df_nj[df_nj.stim_type == "forward"]
df_curr = df_curr[df_curr.plane == "horizontal"]
for stim_number in df_curr.stim_number.unique():
    df_cond = df_curr[(df_curr.stim_number == stim_number)]
    # row = df_curr.loc[random.choice(df_curr.index)]
    row = df_cond.loc[df_cond.spectral_coverage.idxmax()]
    envs = row["cochleagram_0.2"]
    dB_min = 84 if row.plane != "distance" else 88
    _, axis = plt.subplots()
    # axis.imshow(np.where(envs < 84, 0, 1).T, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    axis.imshow(envs.T, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    title = f'Cochleagram threshold ({row.plane}, n_stim={row.stim_number}, spec_cov={row.spectral_coverage})'
    axis.set(title=title, xlabel='Time [sec]', ylabel='Frequency [Hz]')
    # plt.savefig("figures/" + title + ".png")
    # plt.close()
