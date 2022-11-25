import freefield
import slab
from pathlib import Path
import os

# TODO: set unused channels on processors to 99.


freefield.initialize(setup="dome", device=['RX81', 'RX8', 'E:\projects\multi-source-localisation\data\play_buf_msl.rcx'])
# freefield.write(tag='bitmask', value=1, processors='RX81')

speaker_list = list(x for x in range(20, 27) if x is not 23)
sound_list = list()
filepath = Path("E:\projects\multi-source-localisation\data\max")
target_speaker = freefield.pick_speakers(picks=23)[0]


for file in os.listdir(filepath):
    sound_list.append(slab.Sound.read(filepath/file))


vowels = ["a", "e", "i", "o"]
speakers_list = list(x for x in range(20, 27, 2))
signal = slab.Sound.vowel(duration=10.0)
freefield.write(tag="playbuflen", value=signal.n_samples, processors="RX81")

for vowel, speaker in enumerate(speakers_list):
    speaker = freefield.pick_speakers(picks=speaker)[0]
    signal = slab.Sound.vowel(vowels[vowel], duration=10.0)
    freefield.write(tag=f"data{vowel}", value=signal.data, processors=speaker.analog_proc)
    freefield.write(tag=f"chan{vowel}", value=speaker.analog_channel, processors=speaker.analog_proc)



freefield.play()
freefield.halt()

for i, speaker in enumerate(speakers):
    freefield.write()