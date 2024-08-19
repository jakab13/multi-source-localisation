import pandas as pd
import slab
import pathlib
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pycochleagram.cochleagram as cgram
from stimuli.tts_models import models
from analysis.dataframe_generation.post_processing import df_nj

DIR = pathlib.Path(os.getcwd())
tts_model = models["tts_models"][16]

trial_dur = 0.6
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
upper_freq = 11000  # upper frequency limit that carries information for speech
col_order = ["horizontal", "vertical", "distance"]


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


def get_cochleagram(row):
    sound = get_sound(row)
    cg = cgram.human_cochleagram(signal=sound.data, sr=sound.samplerate, hi_lim=11000, nonlinearity="db")
    return cg


def get_spectral_coverage(row, dB_min):
    envs = get_cochleagram(row)
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
df_nj["cochleagram"] = df_nj.apply(lambda row: get_cochleagram(row), axis=1)

# df_nj["sound_data"] = df_nj.apply(lambda row: get_sound(row), axis=1)
# df_nj["spectral_coverage_relative"] = df_nj.apply(lambda row: get_relative_spectral_coverage(row), axis=1)

# df_nj["spectral_data"] = df_nj.apply(lambda row: get_spectral_data(row), axis=1)
df_nj["spectral_coverage_-40"] = df_nj.apply(lambda row: get_spectral_coverage(row, -40), axis=1)

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
                    x = df_curr.spectral_coverage_normed.values.astype(float)
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


spectral_columns = [column for column in df_nj.columns if "spectral_coverage" in column]
df_spectral = df_nj.groupby(["plane", "stim_type", "stim_number"], as_index=False)[spectral_columns].std()
df_thresholds = df_spectral.melt(id_vars=["plane", "stim_type", "stim_number"], value_vars=spectral_columns, var_name="threshold", value_name="std")
df_thresholds["threshold"] = df_thresholds["threshold"].transform(lambda x: int(x[-3:]))
sns.lineplot(df_thresholds[df_thresholds.stim_type == "forward"], x="threshold", y="std", hue="plane")

g = sns.FacetGrid(
    df_thresholds,
    col="plane",
    row="stim_type",
    hue="stim_number",
    palette="winter",
    col_order=col_order,
    height=4,
    aspect=0.8
)
g.map(sns.lineplot, "threshold", "std")
g.add_legend()
g.set_titles(template="{col_name}")
plt.show()


df_thresholds_max = df_thresholds.groupby(["plane", "stim_number", "stim_type"], as_index=False)["std"].max()
df_thresholds_max = df_thresholds_max.merge(df_thresholds[["std", "threshold"]], left_on="std", right_on="std")
df_nj = df_nj.merge(df_thresholds_max[["plane", "stim_number", "stim_type", "threshold"]], on=["plane", "stim_number", "stim_type"])

df_thresholds_stats = df_nj.groupby(["plane", "stim_number", "stim_type"], as_index=False)["spectral_coverage"].agg(
    {"spectral_coverage_mean": "mean", "spectral_coverage_std": "std", "spectral_coverage_min": "min", "spectral_coverage_max": "max"})
df_nj = df_nj.merge(df_thresholds_stats[["plane", "stim_number", "stim_type",
                                         "spectral_coverage_mean", "spectral_coverage_std",
                                         "spectral_coverage_min", "spectral_coverage_max"
                                         ]], on=["plane", "stim_number", "stim_type"])
df_nj["spectral_coverage_normed"] = (df_nj["spectral_coverage"] - df_nj["spectral_coverage_min"]) / \
                                    (df_nj["spectral_coverage_max"] - df_nj["spectral_coverage_min"])


def get_final_spectral_coverage(row):
    column_name = "spectral_coverage_" + str(row["threshold"])
    final_spectral_coverage = row[column_name]
    return final_spectral_coverage


df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_final_spectral_coverage(row), axis=1)


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


fbank = slab.Filter.cos_filterbank(bandwidth=0.2, low_cutoff=20, samplerate=24414)
freqs = fbank.filter_bank_center_freqs()
cmap = matplotlib.cm.get_cmap('inferno')
_, axis = plt.subplots()
mesh = scipy.ndimage.gaussian_filter(np.ma.filled(envs, envs.min()).T, 1)
# mesh = np.ma.filled(envs, envs.min()).T
# extent = (0, mesh.shape[1] / 24414, freqs.min(), freqs.max())
axis.imshow(mesh, origin='lower', aspect='auto', cmap=cmap, interpolation='none', vmin=75, vmax=90)
# cg.set_clim(vmin=75, vmax=90)
axis.set(title='Cochleagram', xlabel='Time [sec]', ylabel='Frequency [Hz]')

