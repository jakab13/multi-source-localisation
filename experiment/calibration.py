from Speakers.FFCalibrator import FFCalibrator

# calibrate setup
cal = FFCalibrator("FREEFIELD")
cal.calibrate()

# test equalization for different speakers
ele_speakers = cal.speakerArray.pick_speakers(picks=[19, 20, 21, 22, 23, 24, 25, 26])
azi_speakers = cal.speakerArray.pick_speakers(picks=[2, 8, 15, 23, 31, 38, 44])

raw, level, full = cal.test_equalization(ele_speakers)  # ele or azi speakers or "all"

cal.spectral_range(raw)
cal.spectral_range(level)
cal.spectral_range(full)
