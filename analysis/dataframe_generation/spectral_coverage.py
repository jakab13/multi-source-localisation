import warnings
# warnings.filterwarnings("error")
warnings.simplefilter(action='ignore', category=FutureWarning)
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
import ast
from analysis.dataframe_generation.utils import pick_speakers
from analysis.dataframe_generation.spectral_coverage_plotting import plot_multisource_coverage_blobs


# from analysis.dataframe_generation.post_processing import df_nj

# df_nj = pd.read_csv("DataFrames/numjudge_post_processed_performance.csv")

df_nj = pd.read_csv("DataFrames/numerosity_judgement_spectral_coverage_otsu.csv")

def convert_cell_to_list(row, column_name):
    output = row[column_name]
    if type(row[column_name]) == str:
        output = ast.literal_eval(row[column_name])
    return output


for column_name in ["stim_talker_ids", "speaker_ids", "stim_country_ids"]:
    df_nj[column_name] = df_nj.apply(lambda row: convert_cell_to_list(row, column_name), axis=1)

DIR = pathlib.Path(os.getcwd())
tts_model = models["tts_models"][16]

trial_dur = 1.2
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
upper_freq = 11000  # upper frequency limit that carries information for speech
col_order = ["horizontal", "vertical", "distance"]

DEFAULT_SAMPLERATE = 24414
slab.set_default_samplerate(DEFAULT_SAMPLERATE)
hrtf = slab.HRTF.kemar()

def spectral_coverage(sound, threshold=-50, low_cutoff=20, high_cutoff=None):
    def otsu_var(data, th):  # helper function to compute Otsu interclass variance
        return np.nansum(
            [np.mean(cls) * np.var(data, where=cls) for cls in [data >= th, data < th]])

    fbank = slab.Filter.cos_filterbank(low_cutoff=low_cutoff, high_cutoff=high_cutoff,
                                  filter_width_factor=0.75,
                                  pass_bands=True, samplerate=sound.samplerate)
    subbands = fbank.apply(sound.channel(0))
    envs = subbands.envelope(kind='dB').data
    if threshold == 'otsu':
        threshold = min(
            range(int(np.min(envs)) + 1, int(np.max(envs))),
            key=lambda th: otsu_var(envs, th))
    coverage = np.where(envs > threshold, 1, 0).sum() / envs.size
    return coverage, threshold


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


def filter_cathedral_noise(sound):
    sound = sound.filter(200, kind="hp")
    sound = sound.filter((266, 344), kind="bs")
    sound = sound.filter(4000, kind="lp")
    return sound


def compose_trial(sub_DIR, talker_ids, country_ids, speaker_ids, stim_type, plane, ax=None):
    trial_composition = list()
    distances = [float(s + 2) for s in speaker_ids] if plane == "distance" else None
    signals = []
    for n in range(len(talker_ids)):
        talker_id = talker_ids[n]
        country_id = country_ids[n]
        speaker_id = speaker_ids[n]
        distance = distances[n] if distances is not None else None
        filename = get_filename(stim_type, talker_id, country_id, distance)
        signal = slab.Sound(sub_DIR / filename)
        if plane != "distance":
            speaker = pick_speakers(speaker_id)[0]
            azi = speaker.azimuth if speaker.azimuth <= 0 else 360 - speaker.azimuth
            ele = speaker.elevation
            filt = hrtf.interpolate(azi, ele, method="triangulate")
            filt = filt.resample(samplerate=signal.samplerate)
            signal_at_loc = filt.apply(signal)
            signal_at_loc = signal_at_loc.resize(0.6)
            trial_composition.append(signal_at_loc)
        else:
            signal_at_loc = filter_cathedral_noise(signal)
            signal_at_loc = signal_at_loc.resize(trial_dur)
            trial_composition.append(signal_at_loc)
        signals.append(signal_at_loc)
    if ax is not None:
        plot_multisource_coverage_blobs(
            signals,
            sr=signals[0].samplerate,
            nperseg=256,
            noverlap=128,  # or sound.samplerate
            coverage_threshold_db=-60,
            blob_smoothing_sigma=3,
            blob_alpha=0.35,
            contrast=3,
            title=f"{len(signals)} sources",
            ax=ax
        )
    sound = sum(trial_composition)
    sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
    sound = sound.resample(24414)
    sound = sound.aweight() if distances is None else sound
    return sound


def get_sound(row, ax=None):
    talker_ids = row["stim_talker_ids"]
    speaker_ids = row["speaker_ids"]
    country_ids = row["stim_country_ids"]
    stim_type = "countries_forward" if row["stim_type"] == "forward" else "countries_reversed"
    sub_DIR = get_sub_DIR(row["plane"], stim_type)
    plane = row["plane"]
    sound = compose_trial(sub_DIR, talker_ids, country_ids, speaker_ids, stim_type, plane, ax=ax)
    return sound


# def get_spectral_data(row):
#     sound = slab.Sound(row["sound_data"], samplerate=24414)
#     freqs, times, power = sound.spectrogram(show=False)
#     return [freqs, times, power]


def get_cochleagram(row):
    sound = get_sound(row)
    cg = cgram.human_cochleagram(signal=sound.data, sr=sound.samplerate, hi_lim=11000, nonlinearity="db")
    return cg


def get_spectral_coverage_cgram(row, dB_min):
    envs = get_cochleagram(row)
    coverage = np.where(envs < dB_min, 0, 1).sum() / envs.size
    print(dB_min, row.name, coverage)
    return coverage


def get_spectral_coverage_slab(row):
    sound = get_sound(row)
    threshold = -115 if row["plane"] == "distance" else -95
    coverage = sound.spectral_coverage(threshold=threshold, high_cutoff=11000)
    print(row.name)
    return coverage


def get_spectral_coverage_local(row, threshold):
    sound = get_sound(row)
    _coverage, _threshold = spectral_coverage(sound, threshold=threshold, high_cutoff=11000)
    print(row.name, _threshold, _coverage)
    return _coverage, _threshold


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


# df_nj = df_nj[df_nj["round"] == 2]
# df_nj["sound_data"] = df_nj.apply(lambda row: get_sound(row), axis=1)
# df_nj["cochleagram"] = df_nj.apply(lambda row: get_cochleagram(row), axis=1)

# df_nj["sound_data"] = df_nj.apply(lambda row: get_sound(row), axis=1)
# df_nj["spectral_coverage_relative"] = df_nj.apply(lambda row: get_relative_spectral_coverage(row), axis=1)

# df_nj["spectral_data"] = df_nj.apply(lambda row: get_spectral_data(row), axis=1)
# for threshold in [20, 23, 24, 25, 30, 35, 40, 45, 46, 47,  50, 55]:
#     key = f"spectral_coverage_neg{threshold}"
#     df_nj.loc[df_nj.plane == "distance", key] = df_nj[df_nj.plane == "distance"].apply(
#         lambda row: get_spectral_coverage_cgram(row, -threshold), axis=1)

# df_nj["spectral_coverage_-40"] = df_nj.apply(lambda row: get_spectral_coverage(row, -40), axis=1)
# df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_spectral_coverage_slab(row), axis='columns')
df_nj["spectral_coverage_2"] = df_nj.apply(lambda row: get_spectral_coverage_slab(row), axis='columns')
# df_nj[["spectral_coverage_otsu", "threshold_otsu"]] = df_nj.apply(lambda row: get_spectral_coverage_local(row, "otsu"),
#                                                                   axis='columns', result_type='expand')

# df_nj.loc[df_nj.plane == "distance", ["spectral_coverage_otsu", "threshold_otsu"]] = \
#     df_nj[df_nj.plane == "distance"].apply(lambda row: get_spectral_coverage_local(row, "otsu"), axis='columns')


for plane in df_nj.plane.unique():
    for stim_number in df_nj.stim_number.unique():
        for stim_type in df_nj.stim_type.unique():
            q = (df_nj.plane == plane) & (df_nj.stim_number == stim_number) & (df_nj.stim_type == stim_type)
            df_nj.loc[q, "spectral_coverage_binned"] = pd.cut(df_nj.loc[q, "spectral_coverage_neg40"], 3,
                                                              labels=["low", "mid", "high"])


for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        for stim_number in df_nj.stim_number.unique():
            for stim_type in df_nj.stim_type.unique():
                q = (df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_number == stim_number) & (df_nj.stim_type == stim_type)
                df_curr = df_nj[q]
                if len(df_curr) > 0:
                    # x = df_curr.spectral_coverage_normed.values.astype(float)
                    x = df_curr.spectral_coverage_neg40.values.astype(float)
                    y = df_curr.resp_number.values.astype(float)
                    reg = scipy.stats.linregress(x, y)
                    df_nj.loc[q, "spectral_coverage_slope"] = reg.slope


def calculate_spectral_coverage_threshold():
    df_spectral = pd.DataFrame(columns=["dB_min", "plane", "std"])
    for dB_min in [85, 86, 87, 88, 89, 90]:
        df_nj["spectral_coverage"] = df_nj.apply(lambda row: get_spectral_coverage_cgram(row, dB_min), axis=1)
        df_curr = df_nj.groupby(["plane"], as_index=False)["spectral_coverage"].std()
        df_curr = df_curr.rename(columns={"spectral_coverage": "std"})
        df_curr["dB_min"] = dB_min
        df_spectral = pd.concat([df_spectral, df_curr], ignore_index=True)
        print(f"Done with min dB: {dB_min}")
    return df_spectral


spectral_columns = [column for column in df_nj.columns if "spectral_coverage_neg" in column]
df_spectral = df_nj.groupby(["plane", "stim_type", "stim_number"], as_index=False)[spectral_columns].std()
df_thresholds = df_spectral.melt(id_vars=["plane", "stim_type", "stim_number"], value_vars=spectral_columns, var_name="threshold", value_name="std")
df_thresholds["threshold"] = df_thresholds["threshold"].transform(lambda x: -int(x[-2:]))
sns.lineplot(df_thresholds[df_thresholds.stim_type == "forward"], x="threshold", y="std", hue="plane")


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


fbank = slab.Filter.cos_filterbank(bandwidth=0.1, low_cutoff=20, high_cutoff=11000, samplerate=24414)
freqs = fbank.filter_bank_center_freqs()
cmap = matplotlib.cm.get_cmap('inferno')
_, axis = plt.subplots()
mesh = scipy.ndimage.gaussian_filter(np.ma.filled(envs, envs.min()).T, 1)
# mesh = np.ma.filled(envs, envs.min()).T
# extent = (0, mesh.shape[1] / 24414, freqs.min(), freqs.max())
axis.imshow(mesh, origin='lower', aspect='auto', cmap=cmap, interpolation='none', vmin=75, vmax=90)
# cg.set_clim(vmin=75, vmax=90)
axis.set(title='Cochleagram', xlabel='Time [sec]', ylabel='Frequency [Hz]')

# PLOTTING


def get_median_idx(rows):
    median = rows.quantile(0.5, 'lower')
    idx = rows.loc[rows == median].index[0]
    return idx


df_spectral_coverage_ids = df_nj[df_nj.stim_type == "forward"].groupby(["plane", "stim_number"], as_index=False)["spectral_coverage_neg30"].agg({
    "idxmin": "idxmin",
    "idxmax": "idxmax",
    "idxmedian": lambda rows: get_median_idx(rows)
}).melt(id_vars=["plane", "stim_number"],
        value_vars=["idxmin", "idxmax", "idxmedian"],
        var_name="idx_type",
        value_name="idx"
        )

df_spectral_coverage_ids["idx_type"] = df_spectral_coverage_ids["idx_type"].apply(lambda x: x[3:])


def plot_cochleagram(row, save=False, threshold=False):
    cg = get_cochleagram(row)
    _, axis = plt.subplots()
    if threshold:
        cg = np.where(cg < row["threshold"], 0, 1)
        cmap = matplotlib.cm.get_cmap('Greys')
        title = f"Cochleagram after threshold \n(n_stim={row.stim_number}, spec_cov={row.spectral_coverage:.2f})"
    else:
        cmap = matplotlib.cm.get_cmap('inferno')
        title = f"Cochleagram \n(n_stim={row.stim_number}, spec_cov={row.spectral_coverage:.2f})"
    axis.imshow(cg, origin='lower', aspect='auto', cmap=cmap, interpolation='none')
    axis.set(title=title, xlabel='Time [sec]', ylabel='Frequency [Hz]')
    if save:
        title = title.rstrip()
        print(title)
        plt.savefig(f"figures/cochleagrams/{title}.png", dpi=400)
        plt.close()
    else:
        plt.show()


df_spectral_coverage_ids[df_spectral_coverage_ids.plane == "distance"]["idx"].apply(lambda idx: plot_cochleagram(df_nj.iloc[idx], save=True, threshold=True))


fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(16, 4))
# forbidden = ["Mali", "Sudan", "Syria", "Tonga", "Yemen"]
for idx, stim_number in enumerate(sorted(df_nj.stim_number.unique())):
    df_curr = df_nj[
        (df_nj.plane == "horizontal") &
        (df_nj.stim_type == "forward") &
        (df_nj.stim_number == stim_number)
        ]
    # mask = df_curr["stim_country_ids"].apply(lambda lst: not any(x in forbidden[:0] for x in lst))
    idxmax = df_curr.nlargest(100, "spectral_coverage").sample(1).index[0]
    idxmin = df_curr["spectral_coverage"].idxmin()
    row_max = df_nj.iloc[idxmax]
    row_min = df_nj.iloc[idxmin]
    print("max", row_max.spectral_coverage)
    print("min", row_min.spectral_coverage)
    # forbidden.extend(row_max.stim_country_ids)
    sound_min = get_sound(row_min, axes[1][int(stim_number-2)])
    sound_max = get_sound(row_max, axes[0][int(stim_number-2)])
fig.tight_layout()
fig.savefig("spectral_coverage_high_low.svg", format="svg", dpi=400)

