import pickle
import pathlib
import slab
import freefield
import time

DIR = pathlib.Path("C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Anaconda3_64\\envs\\freefield\\Lib\\site-packages\\freefield")
with open(DIR / 'data' / 'calibration_dome.pkl', "rb") as f:
    cal = pickle.load(f)

freefield.initialize("dome", default="play_rec")

noise = slab.Sound.pinknoise(duration=1.0)

for i in range(47):
    freefield.set_signal_and_speaker(noise, i)
    freefield.play()
    freefield.wait_to_finish_playing()
    time.sleep(0.5)

