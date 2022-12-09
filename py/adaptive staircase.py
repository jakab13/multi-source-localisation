import slab
import pathlib
import os
import freefield
import time
import numpy as np
import random


# initialize freefield setup
DIR = pathlib.Path(os.getcwd()).absolute()
proc_list = [['RP2', 'RP2', DIR / 'data' / "rcx" / 'button_rec.rcx'],
             ['RX81', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx'],
             ['RX82', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx']]
freefield.initialize(setup="dome", device=proc_list)  # initialize freefield

# define samplerate
samplerate = 48828

# load sound files for target sounds
sound_fp = pathlib.Path(os.getcwd()) / "data" / "sounds" / "demo" / "numbers" / "single" / "normal"# example file path
target_sounds = slab.Precomputed(slab.Sound.read(sound_fp / file) for file in os.listdir(sound_fp))

# pick target speaker (ele: 0, azi: 0)
target_speaker = freefield.pick_speakers(picks=23)[0]  # pick central speaker

# specify masker speakers without the central speaker
speaker_list = list(x for x in range(20, 27) if x != 23)  # central dome speakers without speaker 23 (ele:0°)

# write data to nametags in the MSL RCX file
freefield.write(tag="chan0", value=target_speaker.analog_channel, processors=target_speaker.analog_proc)  # target speaker location

# initialize ResultsFile to handle recording and saving of behavioral data
subject = "max"
results = slab.ResultsFile(subject=subject)

# initialize trialsequence with according conditions
seq = slab.Trialsequence(conditions=speaker_list, n_reps=1, kind="random_permutation")

# TODO: Set playbuflen outside loop as soon as stimuli have equal durations.

solution_converter = {
    1: 5,
    2: 4,
    3: 1,
    4: 3,
    5: 2
}

for trial in seq:
    masker_speaker = freefield.pick_speakers(picks=seq.this_trial)[0]
    ele = masker_speaker.elevation
    stairs = slab.Staircase(start_val=70, n_reversals=2, step_sizes=[4, 1], n_up=1, n_down=1)
    talker = random.randint(1, 108)
    selected_target_sounds = target_sounds[talker*5:(talker+1)*5]
    for level in stairs:
        reaction_time = None
        response = None
        target_sound_i = random.choice(range(len(selected_target_sounds)))
        target_sound = selected_target_sounds[target_sound_i]  # choose random number from sound_list
        stim_dur_masker = 1.2
        masker_sound = slab.Sound.pinknoise(duration=stim_dur_masker)  # masker sound is always noise
        freefield.write(tag="data1", value=masker_sound.data, processors=target_speaker.analog_proc)  # masker sound
        target_sound.level = level  # adjust target level according to staircase
        freefield.write(tag="data0", value=target_sound.data, processors=target_speaker.analog_proc)  # load target sound
        freefield.write(tag="chan1", value=masker_speaker.analog_channel, processors=masker_speaker.analog_proc)  # load masker speaker
        freefield.write(tag="playbuflen", value=masker_sound.n_samples, processors="RX81")
        start_time = time.time()
        freefield.play()  # play trial
        freefield.wait_to_finish_playing()
        stairs.print_trial_info()
        while not freefield.read(tag="response", processor="RP2"):
            time.sleep(0.01)
        curr_response = int(freefield.read(tag="response", processor="RP2"))
        if curr_response != 0:
            reaction_time = int(round(time.time() - start_time, 3) * 1000)
            response = int(np.log2(curr_response))
        # solution = target_sound_i + 1
        solution = solution_converter[target_sound_i + 1]
        correct_response = True if solution / response == 1 else False
        stairs.add_response(1) if correct_response is True else stairs.add_response(0)
        results.write(response, "response")
        results.write(solution, "solution")
        results.write(reaction_time, "reaction_time")
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            time.sleep(0.01)
    results.write(ele, "elevation")
    results.write(stairs.data, "iscorrect")
    results.write(stairs.threshold(), "threshold")
    results.write(stairs.intensities, "intensities")
    results.write(stairs.reversal_points, "reversal_points")
    results.write(stairs.reversal_intensities, "reversal_intensities")


freefield.halt()




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
