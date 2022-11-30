import slab
import pathlib
import os
import freefield
import random
import time
import numpy as np


# initialization + checking for existing ID
id = func.randId()

#print(id)  # for debugging

trials = input("Pls enter Value for Amount of trials, press enter for default\n")

if trials == "":
    trials = 50
else:
    # failsafe
    try:
        int(trials)
    except:
        print("your input is invalid pls enter an numeric value")
        exit(1)

print(trials)  #for debugging


#TODO
# test for loudness threshold and write a quick savefile
DIR = pathlib.Path(os.getcwd()).absolute()
proc_list = [['RP2', 'RP2', DIR / 'data' / "rcx" / 'button_rec.rcx'],
             ['RX81', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx'],
             ['RX82', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx']]
freefield.initialize(setup="dome", device=proc_list)  # initialize freefield

stim_dur = 2.0
samplerate = 48828 / 2
masker_sound = slab.Sound.pinknoise(duration=stim_dur)  # masker sound is always noise
stairs_dict = dict()
for e in range(1, 5):
    stairs_dict[f"stairs{e}"] = slab.Staircase(start_val=65, n_reversals=2, step_sizes=[4, 1])
stairs_dict_copy = stairs_dict.copy()

seq = slab.Trialsequence(conditions=4, kind="infinite")
speaker_list = list(x for x in range(20, 27) if x != 23)  # central dome speakers without speaker 23 (ele:0°)
sound_list = list()  # list of sounds to choose target from (numbers 1-10)
filepath = pathlib.Path(os.getcwd()) / "data" / "sounds" / "max"  # example file path
target_speaker = freefield.pick_speakers(picks=23)[0]  # pick central speaker

for file in os.listdir(filepath):  # load sound files into list
    sound_list.append(slab.Sound.read(filepath/file))

target_sounds = slab.Precomputed(sound_list)


freefield.write(tag="chan0", value=target_speaker.analog_channel, processors=target_speaker.analog_proc)  # target speaker location
freefield.write(tag="data1", value=masker_sound.data, processors=target_speaker.analog_proc)  # masker sound
freefield.write(tag="playbuflen", value=masker_sound.n_samples, processors="RX81")  # 2 seconds duration

results = dict()
for x in range(1, 5):
    results[str(x)] = slab.ResultsFile()
threshs = list()
# TODO: how to implement interleaved staircases?
for trial in seq:
    seq.__next__()
    if seq.this_trial == 1:
        masker_speaker = freefield.pick_speakers(picks=speaker_list[seq.this_trial-1])[0]  # pick random masker speaker
    elif seq.this_trial == 2:
        masker_speaker = freefield.pick_speakers(picks=speaker_list[seq.this_trial-1])[0]  # pick random masker speaker
    elif seq.this_trial == 3:
        masker_speaker = freefield.pick_speakers(picks=speaker_list[seq.this_trial-1])[0]  # pick random masker speaker
    elif seq.this_trial == 4:
        masker_speaker = freefield.pick_speakers(picks=speaker_list[seq.this_trial-1])[0]  # pick random masker speaker
    level = stairs_dict_copy[f"stairs{seq.this_trial}"]._next_intensity
    start_time = time.time()
    reaction_time = None
    response = None
    target_i = random.choice(range(len(sound_list)))
    target_sound = target_sounds[target_i]  # choose random number from sound_list
    freefield.write(tag="data1", value=masker_sound.data, processors=target_speaker.analog_proc)  # masker sound
    target_sound.level = level
    freefield.write(tag="data0", value=target_sound.data, processors=target_speaker.analog_proc)  # load target sound
    freefield.write(tag="chan1", value=masker_speaker.analog_channel, processors=masker_speaker.analog_proc)  # load masker speaker
    freefield.play()  # play trial
    freefield.wait_to_finish_playing()
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.01)
    curr_response = int(freefield.read(tag="response", processor="RP2"))
    if curr_response != 0:
        reaction_time = int(round(time.time() - start_time, 3) * 1000)
        response = int(np.log2(curr_response))
    solution = target_i + 1
    correct_response = True if solution / response == 1 else False
    stairs_dict_copy[f"stairs{seq.this_trial}"].add_response(1) if correct_response is True else stairs_dict_copy[f"stairs{seq.this_trial}"].add_response(0)
    results[str(seq.this_trial)].write(solution, "solution")
    results[str(seq.this_trial)].write(response, "response")
    results[str(seq.this_trial)].write(masker_speaker.elevation, "elevation")
    results[str(seq.this_trial)].write(reaction_time, "rt")
    if stairs_dict[f"stairs{seq.this_trial}"].finished is True:
        for stairs in range(1, len(stairs_dict_copy)+1):
            if stairs_dict_copy[f"stairs{seq.this_trial}"].finished:
                stairs_dict[f"stairs{seq.this_trial}"] = stairs_dict_copy[f"stairs{seq.this_trial}"].pop(f"stairs{seq.this_trial}")
        stair_num = seq.this_trial
        results[str(seq.this_trial)].write(stairs_dict_copy[f"stairs{seq.this_trial}"].data, "iscorrect")
        results[str(seq.this_trial)].write(stairs_dict_copy[f"stairs{seq.this_trial}"].threshold(), "threshold")
        results[str(seq.this_trial)].write(stairs_dict[f"stairs{seq.this_trial}"].intensities, "intensities")
        results[str(seq.this_trial)].write(stairs_dict[f"stairs{seq.this_trial}"].reversal_points, "reversal_points")
        results[str(seq.this_trial)].write(stairs_dict[f"stairs{seq.this_trial}"].reversal_intensities, "reversal_intensities")
    while freefield.read(tag="playback", n_samples=1, processor="RP2"):
        time.sleep(0.01)




freefield.halt()


# 1. checking if loudness test is needed + setting boolean
# 2. set up adaptive staircase
# 3. running the test
# 4. writing savefile and closing it afterwards

#TODO
# generating a masker noise
# setting up a staircase for level manipulation


#TODO
# setting up the actual trial


# Task: find the loudness threshold of a noise that masks the content of a speech sound
# Use an "adaptive staircase" method from slab to find this loudness threshold for every participant
# generate a unique participant ID DONE
# load sound files of speech sounds (talker counting from 1-10)
# generate a masker noise
# set up an adaptive staircase that changes the level of the masker on each step
# each round of the staircase should play a random speech sound and the masker at the same time
# ask for participant to indicate which number they have heard
# check if participant response was correct
# save data once the threshold is reached (don't forget to add participant ID)
# extra 1: rove the loudness of both the talker and the masker
# extra 2: extend the above for speech sounds of multiple talkers
# extra 3: vocode the speech sounds and use those as maskers
