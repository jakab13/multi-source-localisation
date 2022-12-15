import freefield
import slab
import pathlib
import os
import random
import time
import numpy as np
from headpose_estimation.meta_motion import mm_pose as motion_sensor
# from headpose_estimation.meta_motion import scan_connect
samplerate = 24414


# start headpose sensor and calibrate
sensor = motion_sensor.start_sensor()

# initialize FF
DIR = pathlib.Path(os.getcwd()).absolute()
proc_list = [['RP2', 'RP2', DIR / 'data' / "rcx" / 'button_rec.rcx'],
             ['RX81', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx'],
             ['RX82', 'RX8', DIR / 'data' / "rcx" / 'play_buf_msl.rcx']]
freefield.initialize(setup="dome", device=proc_list)  # initialize freefield

# pick speakers and sounds
speaker_list = list(x for x in range(20, 27))
filepath = pathlib.Path("E:\\projects\\multi-source-localisation\\data\\sounds\\tts-numbers_resamp")
sound_list = slab.Precomputed(slab.Sound.read(filepath/file) for file in os.listdir(filepath))
starttone = slab.Sound.read(DIR / "data" / "sounds" / "bell.wav")
buzztone = slab.Sound.vowel(samplerate=samplerate, duration=0.5)


# initialize sequence and response object
seq = slab.Trialsequence(conditions=5, n_reps=3)
results = slab.ResultsFile()

# calibrate sensor
offset = motion_sensor.calibrate_pose(sensor)

# play starting tone
freefield.write(tag=f"data0", value=starttone.data, processors=["RX81", "RX82"])
freefield.write(tag=f"chan0", value=23, processors=["RX81", "RX82"])
freefield.play()
freefield.wait_to_finish_playing()
freefield.wait_for_button()

# run experiment
for trial in seq:
    speaker_ids = random.sample(speaker_list, trial)
    response = None
    for i, speaker_id in enumerate(speaker_ids):
        speaker = freefield.pick_speakers(picks=speaker_id)[0]
        signal = random.choice(sound_list)
        # sound_list.remove(signal)
        freefield.write(tag=f"data{i}", value=signal.data, processors=speaker.analog_proc)
        freefield.write(tag=f"chan{i}", value=speaker.analog_channel, processors=speaker.analog_proc)
    freefield.write(tag="playbuflen", value=samplerate, processors=["RX81", "RX82"])
    start_time = time.time()
    response = 0
    while not response:
        pose = motion_sensor.get_pose(sensor, 30)  # set initial isi based on pose-target difference
        if all(pose):
            pose = pose - offset
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        # if pose[1] > 5:  # subject head position check
            # freefield.write(tag=f"data0", value=buzztone.data, processors=["RX81", "RX82"])
            # freefield.write(tag=f"chan0", value=23, processors=["RX81", "RX82"])
            # freefield.play()
            # freefield.wait_to_finish_playing()
            # freefield.wait_for_button()
        response = freefield.read('response', processor='RP2')
    if all(pose):
        print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    freefield.play()
    freefield.wait_to_finish_playing()
    freefield.wait_for_button()
    curr_response = int(freefield.read(tag="response", processor="RP2"))
    if curr_response != 0:
        reaction_time = int(round(time.time() - start_time, 3) * 1000)
        response = int(np.log2(curr_response))
    correct_response = True if trial / response == 1 else False
    results.write(response, "response")
    results.write(trial, "solution")
    results.write(reaction_time, "rc")
    results.write(correct_response, "is_correct")
    results.write(round(pose[0], 2), "azimuth")
    results.write(round(pose[1], 2), "elevation")

for channel in range(5):
    freefield.write(tag=f"data{channel}", value=np.zeros(500000), processors=["RX81", "RX82"])

motion_sensor.disconnect(sensor)
freefield.halt()
