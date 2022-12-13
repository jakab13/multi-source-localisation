import slab
import pathlib
import os
import random
random.seed = 50


def load(DIR):
    """
    Given a non-empty directory, load all sound files (.wav) within that directory.

    Args:
        dir: the directory containing sound files.

    Returns:
        sound_list: list of sounds within the specified directory.
    """
    DIR = pathlib.Path(DIR)
    sound_list = list()
    if os.listdir(DIR).__len__():
        print("Empty directory")
    for file in os.listdir(DIR):
        file = pathlib.Path(file)
        sound_list.append(slab.Sound.read(pathlib.Path(DIR/file)))
    return sound_list

def resample(sounds, samplerate):
    """
    Resample sound.

    Args:
        sound: slab.sound.Sound instance or a list of the instance
        samplerate: Desired samplerate

    Returns:
        slab.sound.Sound instance
    """
    if type(sounds) == list:
        sound_list_resamp = list()
        for e in sounds:
            s = e.resample(samplerate)
            sound_list_resamp.append(s)
        return sound_list_resamp
    if type(sounds) == slab.sound.Sound:
        sound_resamp = sounds.resample(samplerate)
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


def talker_data(data, DIR, pattern):
    """
    Searches for a pattern in a list of sound names and sound data and retrieves the data which matches the pattern.

    Args:
        data: list of slab.sound.Sound data.
        pattern: searching pattern, talker id for instance.
        DIR: sound file directory, has to match data list.

    Returns:
        sound data matching the searching pattern.

    """
    talker_files = list()
    for i, sound_name in enumerate(os.listdir(DIR)):
        if pattern in sound_name:
            talker_files.append(data[i])
    return talker_files

def concatenate(sounds, n_concatenate=5):
    """
    Randomly concatenates sounds from a list into a slab.Sound.sequence object without permutation.

    Args:
        sounds: list of sounds to concatenate
        n_concatenate: number of total sounds in a sequence

    Returns:
        slab.Sound.sequence
    """
    sample = random.sample(sounds, k=n_concatenate)
    sequence = slab.Sound.sequence(*sample)
    return sequence


if __name__ == "__main__":
    DIR = pathlib.Path("D:\Projects\multi-source-localisation\data\sounds\\tts-harvard-5")
    sounds_data = load(DIR)
    pattern = "p227"
    talker_files = talker_data(data=sounds_data, pattern=pattern, DIR=DIR)
    sequence = concatenate(talker_files, n_concatenate=len(talker_files))

