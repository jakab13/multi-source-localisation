import freefield
import slab
from pathlib import Path
import os
import random
import time

# TODO: set unused channels on processors to 99.


freefield.initialize(setup="dome", device=['RX81', 'RX8', 'E:\projects\multi-source-localisation\data\play_buf_msl.rcx'])
filepath = Path("E:\projects\multi-source-localisation\data\max")
# freefield.write(tag='bitmask', value=1, processors='RX81')

speaker_list = list(x for x in range(20, 27) if x is not 23)
sound_list = list()
samplerate = 48828 * 2
play_duration = 2.0

for file in os.listdir(filepath):
    sound_list.append(slab.Sound.read(filepath/file))


speaker_list = list(x for x in range(20, 27))
talker_count = list(x for x in range(1, 6))
trials = 20
freefield.write(tag="playbuflen", value=samplerate, processors="RX81")


for trial in range(trials):
    num_talkers = random.choice(talker_count)
    speaker_ids = random.sample(speaker_list, num_talkers)
    sound_list_copy = sound_list.copy()
    for i, speaker_id in enumerate(speaker_ids):
        speaker = freefield.pick_speakers(picks=speaker_id)[0]
        signal = random.choice(sound_list_copy)
        sound_list_copy.remove(signal)
        freefield.write(tag=f"data{i}", value=signal.data.flatten(), processors=speaker.analog_proc)
        freefield.write(tag=f"chan{i}", value=speaker.analog_channel, processors=speaker.analog_proc)
    freefield.play()
    while freefield.read(tag="playback", n_samples=1, processor=speaker.analog_proc):
        time.sleep(0.01)

freefield.play()
freefield.halt()

