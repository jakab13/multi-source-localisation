import slab
import pathlib
import os
import random

# TODO: aling sounds by talker

def load_sounds(dir):
    """
    Given a non-empty directory, load all sound files (.wav) within that directory.

    Args:
        dir: the directory containing sound files.

    Returns:
        sound_list: list of sounds within the specified directory.
    """
    sound_list = list()
    if len(fp) == 0:
        print("Empty directory")
    for file in dir:
        sound_list.append(slab.Sound.read(dir/file))
    return sound_list

def resample(sound, samplerate):
    """
    Resample sound.

    Args:
        sound: slab.sound.Sound instance or a list of the instance
        samplerate: Desired samplerate

    Returns:
        slab.sound.Sound instance
    """
    if type(sound) == list:
        sound_list_resamp = list()
        for e in sound_list:
            s = e.resample(samplerate)
            sound_list_resamp.append(s)
        return sound_list_resamp
    if type(sound) == slab.sound.Sound:
        sound_resamp = sound.resample(samplerate)
        return sound_resamp
    else:
        raise TypeError("Only single instances of slab.sound.Sound or a list of these objects are allowed as input!")


def reverse(sound):
    """
    Reverse time course of sounds in a given slab.sound.Sound object or a list of this instance.

    Args:
        sound: slab.sound.Sounde instance of a list of it.

    Returns:
        Single or list of slab.sound.Sound instance.
    """
    if type(sound) == list:
        sound_list_reversed = list()
        for s in sound:
            s.data = s.data[::-1]
            sound_list_reversed.append(s)
        return sound_list_reversed
    if type(sound) == slab.sound.Sound:
        sound.data = sound.data[::-1]
        return sound


def align_sound_duration(sound_list, sound_duration, samplerate=48828):
    """
    Align the duration of sounds in a given list of slab.sound.Sound instances.

    Args:
        sound_list: list of slab.sound.Sound instances to be aligned.
        sound_duration: desired sound_duration (float).
        samplerate: samplerate of the sounds in sound_list. Has to be equal among all instances in the list.

    Returns:
        Duration-aligned list of slab.sound.Sound instances.
    """
    trial_duration = slab.Signal.in_samples(sound_duration, samplerate)
    for sound in sound_list:
        silence_duration = trial_duration - sound_list[sound].n_samples
        if silence_duration > 0:
            silence = slab.Sound.silence(duration=silence_duration, samplerate=samplerate)
            sound_list[sound] = slab.Sound.sequence(sound_list[sound], silence)
    return sound_list


for s in range(5):
    talker = random.randint(1, 108)
    medstims = sound_list_resampled[talker*5:(talker+1)*5].copy()
    random.shuffle(medstims)
    sample = slab.Sound.sequence(medstims[0], medstims[1], medstims[2], medstims[3], medstims[4])
    sample.write(f"E:\\projects\\multi-source-localisation\\data\\sounds\\demo\\numbers\\5_reps\\normal\\sample_{s}.wav")
