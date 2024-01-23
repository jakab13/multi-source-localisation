import slab
import matplotlib.pyplot as plt
import librosa
import numpy as np

import scienceplots
plt.style.use("science")
plt.ion()


upper_frequency = None
dyn_range = 120
p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

sound = slab.Sound.read("/home/max/labplatform/male_example_belgium.wav")
sr = sound.samplerate
freqs, times, power = sound.spectrogram(show=False)
power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
shape = power.shape
fh = int(shape[1]/2)
power = power[:, :fh]

# set lower bound of colormap (vmin) from dynamic range.
dB_max = power.max()
vmin = dB_max - dyn_range
extent = (times.min(initial=0), times.max(initial=0), freqs.min(initial=0),
          upper_frequency or freqs.max(initial=0))
dB_max = power.max()
vmin = dB_max - dyn_range
extent = (times.min(initial=0), times.max(initial=0)/2, freqs.min(initial=0),
          upper_frequency or freqs.max(initial=0))
plt.imshow(power, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=None)
plt.title("")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])

plt.savefig("/home/max/labplatform/plots/MA_thesis/materials_methods/harmonics.png",
            dpi=400, bbox_inches="tight")
