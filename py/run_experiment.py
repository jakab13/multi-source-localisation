import freefield
import slab

# TODO: set unused channels on processors to 99.


freefield.initialize(setup="dome", device=['RX81', 'RX8', 'E:\projects\multi-source-localisation\data\play_buf_msl.rcx'])
# freefield.write(tag='bitmask', value=1, processors='RX81')

vowels = ["a", "e", "i", "o", "u"]
speakers_list = list(x for x in range(20, 27, 2))
freefield.write(tag="playbuflen", value=signal.n_samples, processors="RX81")

for vowel, speaker in enumerate(speakers_list):
    speaker = freefield.pick_speakers(picks=speaker)[0]
    signal = slab.Sound.vowel(vowels[vowel], duration=10.0)
    freefield.write(tag=f"data{vowel+1}", value=signal.data, processors=speaker.analog_proc)
    freefield.write(tag=f"chan{vowel+1}", value=speaker.analog_channel, processors=speaker.analog_proc)



freefield.play()
freefield.halt()

for i, speaker in enumerate(speakers):
    freefield.write()