import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use("science")
plt.set_cmap("Greys")
plt.ion()

import slab
import pickle as pkl

goal_sf = 48828
belgium = slab.Sound.read("/home/max/labplatform/male_example_belgium.wav").resample(goal_sf).resize(1.0)
cuba = slab.sound.Sound.read("/home/max/labplatform/female_example_cuba.wav").resample(goal_sf).resize(1.0)
combined = cuba+belgium
diff = cuba - belgium

layout = """
AB
"""
figs, axs = plt.subplot_mosaic(mosaic=layout)
plt.subplots_adjust(wspace=0.25, hspace=0.2)

cuba.waveform(axis=axs["A"])
belgium.waveform(axis=axs["A"])
cuba.spectrum(axis=axs["B"])
belgium.spectrum(axis=axs["B"])

axs["B"].set_xlabel("Frequency [Hz]")
for key, ax in axs.items():
    axs[key].set_title("")

plt.savefig("/home/max/labplatform/plots/MA_thesis/materials_methods/talker_description.png",
            dpi=800)


rifle = pkl.load(open("/home/max/labplatform/sound_files/locaaccu_machine_gun_noise.pkl", "rb"))[0]
layout = """
ab
cc
"""
figs, axs = plt.subplot_mosaic(mosaic=layout)
rifle.waveform(axis=axs["a"])
rifle.spectrum(axis=axs["b"])
rifle.spectrogram(axis=axs["c"])

axs["a"].sharex(axs["b"])
axs["b"].set_title("Female Cuba")
plt.tight_layout()


