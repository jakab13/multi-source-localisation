import slab
import pathlib
import os
import random
import numpy as np


gender = "M"  # F or M
talker = "max"  # number of talker
nums_rec = 5

root = pathlib.Path.cwd()/"data"
if not os.path.exists(root):
    os.mkdir(root)
duration = 1.0
samplerate = 48828
for number in range(nums_rec):  # record sound files
    filename = pathlib.Path(f"talker-{talker}_number-{number}_gender-{gender}.wav")
    filepath = pathlib.Path(root/f"{talker}")
    input(f"Press any key to start recording number {number}")
    sound = slab.Sound.record(duration=duration, samplerate=samplerate)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    sound.write(filepath/filename)
    print(f"Successfully saved sound {number} from talker {talker}!")

# Alternatively, use google TTS API to generate stimuli.

gender = "F"  # F or M
talker = "gTTS-de"  # number of talker
nums_rec = 5
root = pathlib.Path.cwd() / "data" / "sounds"
language = "de"

if not os.path.exists(root):
    os.mkdir(root)

for number in range(1, nums_rec+1):  # record sound files
    filename = pathlib.Path(f"talker-{talker}_number-{number}_gender-{gender}")
    filepath = pathlib.Path(root/f"{talker}")
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    tts = gtts.gTTS(str(number), lang=language)
    tts.save(str(filepath/filename) + ".wav")
    print(f"Successfully saved sound {number} from talker {talker}!")

# generate and save different lengths of stimuli
samplerate = 48828
root = pathlib.Path("C:\\Users\\neurobio\\Desktop\\tts-numbers")
sound_list = list()
sound_list_resampled = list()
fp = os.listdir(f"{root}")
for file in fp:
    sound_list.append(slab.Sound.read(root/file))

# resample
for sound in sound_list:
    sound = sound.resample(samplerate)
    sound_list_resampled.append(sound)

for i, sound in enumerate(sound_list_resampled):
    slab.Sound.write(sound, filename=f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-harvard-5\\{fp[i]}")

shortstims = list(x.resize(duration=0.3) for x in sound_list_resampled.copy())

for i, stim in enumerate(shortstims):
    slab.Sound.write(stim, filename=f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers\\0.3s\\{fp[i]}")

for s in range(5):
    talker = random.randint(1, 108)
    medstims = sound_list_resampled[talker*5:(talker+1)*5].copy()
    random.shuffle(medstims)
    sample = slab.Sound.sequence(medstims[0], medstims[1], medstims[2], medstims[3], medstims[4])
    sample.write(f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers\\5s\\sample_{s}.wav")

trial_duration = slab.Signal.in_samples(1.5, samplerate)

# 10 s stimuli
for s in range(5):
    talker = random.randint(1, 108)
    longstims = sound_list_resampled[talker*5:(talker+1)*5].copy()
    random.shuffle(longstims)
    sample_choices = [random.randint(0, 4) for i in range(10)]
    for sample_choice in sample_choices:
        silence_duration = trial_duration - longstims[sample_choice].n_samples
        if silence_duration > 0:
            silence = slab.Sound.silence(duration=silence_duration, samplerate=samplerate)
            longstims[sample_choice] = slab.Sound.sequence(longstims[sample_choice], silence)
    sample = slab.Sound.sequence(longstims[sample_choices[0]], longstims[sample_choices[1]], longstims[sample_choices[2]], longstims[sample_choices[3]], longstims[sample_choices[4]], longstims[sample_choices[5]], longstims[sample_choices[6]], longstims[sample_choices[7]], longstims[sample_choices[8]], longstims[sample_choices[9]])
    sample.write(f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers\\10s\\sample_{s}.wav")


longstims = sound_list_resampled.copy() * 2

random.shuffle(longstims)
sample = slab.Sound.sequence(longstims[0], longstims[1], longstims[2], longstims[3], longstims[4], longstims[5], longstims[6], longstims[7], longstims[8], longstims[9])
sample.write(f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers\\10s\\sample_5.wav")

# load stimuli to be reversed
root = pathlib.Path("C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers\\10s")
sound_list = list()
sound_list_reversed = list()
fp = os.listdir(f"{root}")
for file in fp:
    sound_list.append(slab.Sound.read(root/file))

# reverse stimuli
for sound in sound_list:
    sound.data = sound.data[::-1]
    sound_list_reversed.append(sound)

# save reversed stimuli
for i, sound in enumerate(sound_list_reversed):
    slab.Sound.write(sound, filename=f"C:\\Users\\neurobio\\Desktop\\sounds_resampled\\tts-numbers_reversed\\10s\\{fp[i]}")