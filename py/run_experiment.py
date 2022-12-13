import freefield
import slab
import pathlib
import os
import random
import time
import numpy as np
from meta_motion import mm_pose as motion_sensor
from meta_motion import scan_connect

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
buzztone = slab.Sound.read(DIR / "data" / "sounds" / "bell.wav")
# samplerate = 24414

# initialize sequence and response object
seq = slab.Trialsequence(conditions=5, n_reps=3)
results = slab.ResultsFile()

# start headpose sensor
sensor = motion_sensor.start_sensor()

# play starting tone
freefield.write(tag=f"data0", value=starttone.data, processors="RX81")
freefield.write(tag=f"chan0", value=23, processors="RX81")
freefield.play()
freefield.wait_to_finish_playing()
freefield.wait_for_button()

# run experiment
for trial in seq:
    offset = motion_sensor.calibrate_pose(sensor)
    tracking = True
    while tracking:
        pose = motion_sensor.get_pose(sensor, 30)  # set initial isi based on pose-target difference
        if all(pose):
            for x, y in zip(pose, offset):
                if np.abs(x - y) > 5:
                    freefield.write(tag=f"data0", value=buzztone.data, processors="RX81")
                    freefield.write(tag=f"chan0", value=23, processors="RX81")
                    freefield.play()
                    freefield.wait_to_finish_playing()
                    freefield.wait_for_button()
            pose = pose - offset
            print('head pose at azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
            tracking = False if freefield.read('response', processor='RP2') is True else False
        else:
           print('no head pose detected', end="\r", flush=True)
    speaker_ids = random.sample(speaker_list, trial)
    response = None
    for i, speaker_id in enumerate(speaker_ids):
        speaker = freefield.pick_speakers(picks=speaker_id)[0]
        signal = random.choice(sound_list)
        freefield.write(tag="playbuflen", value=signal.n_samples, processors=["RX81", "RX82"])  # set playbuflen tag
        freefield.write(tag=f"data{i}", value=signal.data, processors=speaker.analog_proc)
        freefield.write(tag=f"chan{i}", value=speaker.analog_channel, processors=speaker.analog_proc)
    start_time = time.time()
    freefield.play()
    freefield.wait_to_finish_playing()
    freefield.wait_for_button()
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
    results.write(pose, "headpose")

for channel in range(5):
    freefield.write(tag=f"data{channel}", value=np.zeros(500000), processors=speaker.analog_proc)

motion_sensor.disconnect(sensor)
freefield.halt()
