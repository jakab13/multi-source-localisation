import numpy as np
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
sns.set_theme(style="white", palette="Set1")
plt.rcParams['text.usetex'] = True  # TeX rendering
import slab

goal_sf = 48828
belgium = slab.Sound.read("/home/max/labplatform/male_example_belgium.wav").resample(goal_sf)
cuba = slab.sound.Sound.read("/home/max/labplatform/female_example_cuba.wav").resample(goal_sf)
combined = cuba.resize(1.0)+belgium.resize(1.0)

layout = """
AB
CC
DD
"""
figs, axs = plt.subplot_mosaic(mosaic=layout)
cuba.waveform(axis=axs["A"])
belgium.waveform(axis=axs["A"])
cuba.spectrum(axis=axs["B"])
belgium.spectrum(axis=axs["B"])
cuba.spectrogram(axis=axs["C"])
belgium.spectrogram(axis=axs["D"])

axs["C"].sharex(axs["D"])
axs["C"].set_title("Female Cuba")
axs["D"].set_title("Male Belgium")
plt.tight_layout()
