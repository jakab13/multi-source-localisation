from Speakers import FFCalibrator
import slab

cal = FFCalibrator.FFCalibrator("FREEFIELD")

sound = slab.Sound.chirp(duration=0.1, kind="linear", level=85, from_frequency=0,
                         to_frequency=20000, samplerate=cal.device.RP2.GetSFreq())
sound = sound.ramp(duration=sound.duration/50)

speaker = cal.speakerArray.pick_speakers(23)

cal.play_and_record(speaker=speaker, sound=sound, equalize=False)


# test calibration
censpeaks = cal.speakerArray.pick_speakers(picks=[20, 21, 22, 23, 24, 25, 26])
cal.calibrate(speakers=censpeaks)

raw, level, full = cal.test_equalization(speakers=censpeaks)

cal.spectral_range(signal=full)