import random
import slab
import numpy as np
import matplotlib.pyplot as plt

# load calib sound
chirp = slab.Sound.chirp(0.05, samplerate=48828, to_frequency=24000)
# power_full = chirp.data.mean()
# chirp_norm = slab.Sound(chirp.data - chirp.data.mean())
# z, freqs = chirp_norm.spectrum(show=False)
# freqs = np.linspace(0, 20000, len(freqs))
# plt.plot(freqs, z)
# plt.xscale("log")

# inverse filtering example
freqs = [f * 400 for f in range(10)]
gain = [random.random() + .4 for _ in range(10)]
tf = slab.Filter.band(frequency=freqs, gain=gain, samplerate=48828)
tf.tf()
recording = tf.apply(chirp)
recording.spectrum()

inverse = slab.Filter.equalizing_filterbank(reference=chirp, sound=recording, length=3052, bandwidth=1/16)
inverse.tf()
equalized = inverse.apply(recording)

fix, ax = plt.subplots(2, sharex=True)
chirp.spectrum(axis=ax[0], show=False)
equalized.spectrum(axis=ax[1], show=False)
ax[0].set(title="Original")
ax[1].set(title="Reconstructed")
plt.show()
