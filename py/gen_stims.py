import slab
import pathlib
import os
# import gtts
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
talker = "gTTS-de"  # number of talker
samplerate = 48828
root = pathlib.Path.cwd() / "data" / "sounds"
sound_list = list()
sound_list_resampled = list()
fp = os.listdir(f"{root}/{talker}")
for file in fp:
    sound_list.append(slab.Sound.read(root/talker/file))

# resample
for sound in sound_list:
    sound = sound.resample(samplerate)
    sound_list_resampled.append(sound)

for i, sound in enumerate(sound_list_resampled):
    slab.Sound.write(sound, filename=f"C:\\Users\\neurobio\\Desktop\\sounds\\1s\\sample_{i+1}.wav")

shortstims = list(x.resize(duration=0.3) for x in sound_list_resampled.copy())

for i, stim in enumerate(shortstims):
    slab.Sound.write(stim, filename=f"C:\\Users\\neurobio\\Desktop\\sounds\\0.3s\\sample_{i+1}.wav")

medstims = sound_list_resampled.copy()
random.shuffle(medstims)
sample = slab.Sound.sequence(medstims[0], medstims[1], medstims[2], medstims[3], medstims[4])
sample.write("C:\\Users\\neurobio\\Desktop\\sounds\\5s\\sample_5.wav")


longstims = sound_list_resampled.copy() * 2

random.shuffle(longstims)
sample = slab.Sound.sequence(longstims[0], longstims[1], longstims[2], longstims[3], longstims[4], longstims[5], longstims[6], longstims[7], longstims[8], longstims[9])
sample.write("C:\\Users\\neurobio\\Desktop\\sounds\\10s\\sample_5.wav")