import slab
import pathlib
import os

gender = "M"  # F or M
talker = "max"  # number of talker
root = pathlib.Path(os.getcwd())/"data"
if not os.path.exists(root):
    os.mkdir(root)
duration = 2.0
samplerate = 48828
for number in list(range(1, 11)):
    filename = pathlib.Path(f"talker-{talker}_number-{number}_gender-{gender}.wav")
    filepath = pathlib.Path(root/f"{talker}")
    input(f"Press any key to start recording number {number}")
    sound = slab.Sound.record(duration=duration, samplerate=samplerate)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    sound.write(filepath/filename)
    print(f"Successfully saved sound {number} from talker {talker}!")