from Speakers.FFCalibrator import FFCalibrator
import matplotlib.pyplot as plt
import slab

# initialize
cal = FFCalibrator("FREEFIELD")

cal.calibrate(speakers=[19, 20, 21, 22, 23, 24, 25, 26], save=False)

# test equalization for different speakers
ele_speakers = cal.speakerArray.pick_speakers(picks=[19, 20, 21, 22, 23, 24, 25, 26])
azi_speakers = cal.speakerArray.pick_speakers(picks=[2, 8, 15, 23, 31, 38, 44])

raw, level, full = cal.test_equalization(ele_speakers)  # ele or azi speakers or "all"

cal.spectral_range(raw)
cal.spectral_range(level)
cal.spectral_range(full)

# plotting relevant info

chirp = slab.Sound.chirp(duration=0.1, samplerate=48828, level=35, from_frequency=100, to_frequency=20000,
                         kind="linear")
rawsamp = raw.channel(3)
levelsamp = level.channel(3)
fullsamp = full.channel(3)

_, axis = plt.subplots()
chirp.spectrum(axis=axis, show=False)
rawsamp.spectrum(axis=axis, show=False)
plt.legend(["original", "12.5°"])

# plot power spectrum difference between target and recording
# diff = chirp - rawsamp  # take one channel for illustration
# x, freqs = diff.spectrum(show=False)
# tf_diff = slab.Filter.band(frequency=freqs.data.tolist(), gain=x.flatten().tolist(), samplerate=48828)

_, axis = plt.subplots()
# axis.set_xscale("log")
inverse = slab.Filter.equalizing_filterbank(reference=chirp, sound=levelsamp, length=1000, bandwidth=1 / 50)
inverse.tf(axis=axis)
chirp.spectrum(axis=axis)
levelsamp.spectrum(axis=axis)
plt.legend(["Inverse Filter", "Original", "Level equalized recording"])

equalized = inverse.apply(levelsamp)
_, axis = plt.subplots()
# axis.set_xscale("log")
inverse.tf(axis=axis)
chirp.spectrum(axis=axis)
levelsamp.spectrum(axis=axis)
equalized.spectrum(axis=axis)
axis.grid(True)
plt.legend(["Inverse Filter", "Original", "Level equalized recording", "Equalized signal"])

_, axis = plt.subplots()
# axis.set_xscale("log")
inverse = slab.Filter.equalizing_filterbank(reference=chirp, sound=levelsamp, length=1000, bandwidth=1 / 50)
inverse.tf(axis=axis)
axis.set_xscale("log")