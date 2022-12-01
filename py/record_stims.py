import slab
import pathlib
import os
import gtts


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
talker = "gTTS"  # number of talker
nums_rec = 5
root = pathlib.Path.cwd() / "data" / "sounds"
language = "de"

if not os.path.exists(root):
    os.mkdir(root)

for number in range(1, nums_rec+1):  # record sound files
    filename = pathlib.Path(f"talker-{talker}_number-{number}_gender-{gender}.wav")
    filepath = pathlib.Path(root/f"{talker}")
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    tts = gtts.gTTS(str(number), lang=language)
    tts.save(str(filepath/filename) + ".mp3")
    print(f"Successfully saved sound {number} from talker {talker}!")