import matplotlib.pyplot as plt
import slab
import pathlib
import os
import pandas as pd
import re
from os.path import join
import numpy as np
from librosa.feature import spectral_centroid

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(os.getcwd())

MSL_stimuli_directory = DIR / 'experiment' / 'samples' / 'VEs' / 'vocoded'


def get_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


def get_from_file_name(file_path, pre_string):
    sub_val = None
    file_name = file_path.name
    sub_string = file_name[file_name.find(pre_string) + len(pre_string):file_name.rfind('.wav')]
    if sub_string:
        sub_val = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", sub_string)[0]
        sub_val = int(sub_val) if sub_val.isdigit() else float(sub_val)
    return sub_val


def get_duration(file_path):
    sound = slab.Sound(file_path)
    return sound.duration


def get_spectral_feature(file_path, feature_name, mean="rms", duration=None, control=False):
    sound = slab.Sound(file_path)
    if duration is not None:
        duration = slab.Signal.in_samples(duration, sound.samplerate)
        sound.data = sound.data[:duration]
    feature = np.asarray(sound.spectral_feature(feature_name, mean=mean))
    feature_avg = feature.mean()
    return feature_avg


def get_spectral_slope(file_path, duration=None):
    sound = slab.Sound(file_path)
    if duration is not None:
        duration = slab.Signal.in_samples(duration, sound.samplerate)
        sound.data = sound.data[:duration]
    win_length = slab.Signal.in_samples(0.3, sound.samplerate)
    centroids = spectral_centroid(y=sound.data.T, sr=sound.samplerate, n_fft=win_length)
    centroids = centroids.transpose(2, 0, 1)
    centroids = centroids.squeeze()
    slope = np.gradient(centroids, axis=0).mean()
    return slope


def get_onset_slope(file_path):
    sound = slab.Sound(file_path)
    return sound.onset_slope()


def get_time_cog(file_path, duration=1.0, resamplerate=48828):
    sound = slab.Sound(file_path)
    if sound.samplerate != resamplerate:
        sound = sound.resample(resamplerate)
    duration = slab.Signal.in_samples(duration, sound.samplerate)
    sound.data = sound.data[:duration]
    env = sound.envelope()
    cog_l = np.average(np.arange(0, len(env[:, 0])), weights=env[:, 0])
    cog_r = np.average(np.arange(0, len(env[:, 1])), weights=env[:, 1])
    cog = np.average([cog_l, cog_r])
    cog /= sound.samplerate
    return cog


MSL_file_paths = [f for f in get_file_paths(MSL_stimuli_directory)]
MSL_stimuli_features = {f.name: {} for f in get_file_paths(MSL_stimuli_directory)}

COLUMN_NAMES = [
    "vocalist",
    "duration",
    "centroid",
    "flatness"
]

for MSL_file_path in MSL_file_paths:
    MSL_stimuli_features[MSL_file_path.name] = {
        "centroid": get_spectral_feature(MSL_file_path, "centroid"),
        "flatness": get_spectral_feature(MSL_file_path, "flatness")
    }

df = pd.DataFrame.from_dict(MSL_stimuli_features, columns=COLUMN_NAMES, orient="index")
df = df.round(decimals=5)
df.to_csv('analysis/acoustics/MSL_stimuli_features.csv')