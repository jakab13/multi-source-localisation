import freefield
import slab
import pathlib
import os
import random
import time
import numpy as np

# TODO: set unused channels on processors to 99.

# initialize FF
DIR = pathlib.Path(os.getcwd()).absolute()
proc_list = [['RP2', 'RP2', DIR / 'data' / "rcx" / 'button_rec.rcx'],
             ['RX81', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx'],
             ['RX82', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx']]
freefield.initialize(setup="dome", device=proc_list)  # initialize freefield

# pick speakers and sounds
speaker_list = list(x for x in range(20, 27))
stim_dur = 1  # can be 0.3,1,5 or 10s
filepath = pathlib.Path(DIR / f"data/sounds/{stim_dur}s")
sound_list = slab.Precomputed(slab.Sound.read(filepath/file) for file in os.listdir(filepath))

# number of maximum talkers
freefield.write(tag="playbuflen", value=100000, processors="RX81")

# initialize sequence and response object
seq = slab.Trialsequence(conditions=4, n_reps=5)
results = slab.ResultsFile()

# loop through sequence
for trial in seq:
    speaker_ids = random.sample(speaker_list, trial)
    sound_list_copy = sound_list.copy()
    response = None
    for i, speaker_id in enumerate(speaker_ids):
        speaker = freefield.pick_speakers(picks=speaker_id)[0]
        signal = random.choice(sound_list_copy)
        signal.data = signal.data[::-1]
        sound_list_copy.remove(signal)
        freefield.write(tag=f"data{i}", value=signal.data, processors=speaker.analog_proc)
        freefield.write(tag=f"chan{i}", value=speaker.analog_channel, processors=speaker.analog_proc)
    start_time = time.time()
    freefield.play()
    freefield.wait_to_finish_playing()
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.01)
    curr_response = int(freefield.read(tag="response", processor="RP2"))
    if curr_response != 0:
        reaction_time = int(round(time.time() - start_time, 3) * 1000)
        response = int(np.log2(curr_response))
    solution = trial + 1
    correct_response = True if solution / response == 1 else False
    results.write(response, "response")
    results.write(solution, "solution")
    results.write(reaction_time, "rc")
    results.write(correct_response, "is_correct")
    while freefield.read(tag="playback", n_samples=1, processor="RP2"):
        time.sleep(0.01)


freefield.halt()
