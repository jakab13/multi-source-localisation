from Speakers.speaker_config import *
import os

# TODO: how to add speakers??
basedir = get_config(setting="CAL_ROOT")
filename = "dome_speakers.txt"
file = os.path.join(basedir, filename)
spk_array = SpeakerArray(file=file)
spk_array.load_speaker_table()

spks = spk_array.pick_speakers(picks=list(range(19, 28)))  # pick all speakers at azimuth 0°
