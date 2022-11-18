import slab
#import random # for ID generation
#import string # for ID generation


# Task: find the loudness threshold of a noise that masks the content of a speech sound

# Use an "adaptive staircase" method from slab to find this loudness threshold for every participant

# https://pypi.org/project/slab/ (first textbox, second line)
#stimulus = slab.Sound.tone(frequency=500, duration=0.5) # make a 0.5 sec pure tone of 500 Hz
#stairs = slab.Staircase(start_val=50, n_reversals=10) # set up the adaptive staircase
#for level in stairs: # the staircase object returns a value between 0 and 50 dB for each trial
#    stimulus.level = level
#    stairs.present_tone_trial(stimulus) # plays the tone and records a keypress (1 for 'heard', 2 for 'not heard')
#print(stairs.threshold()) # print threshold when done



# generate a unique participant ID

# TODO
#https://www.geeksforgeeks.org/generating-random-ids-python/ (third example)
# assign ID to variable for later saving

# load sound files of speech sounds (talker counting from 1-10)
# generate a masker noise
# set up an adaptive staircase that changes the level of the masker on each step
# each round of the staircase should play a random speech sound and the masker at the same time
# ask for participant to indicate which number they have heard

#TODO
#maybe tell participant to press an input after each trial or do it via a noise that has to be played beforehand

# check if participant response was correct

#TODO 
#checking if input equals x via if clause

# save data once the threshold is reached (don't forget to add participant ID)
#threshhold = open ( id , w)
#threshhold.write ()
  


# extra 1: rove the loudness of both the talker and the masker
# extra 2: extend the above for speech sounds of multiple talkers
# extra 3: vocode the speech sounds and use those as maskers
