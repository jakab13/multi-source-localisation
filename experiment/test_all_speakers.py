from labplatform.config import get_config
from experiment.RP2 import RP2Device
from experiment.RX8 import RX8Device
from Speakers.speaker_config import SpeakerArray
import os
import slab
import logging
import time

log = logging.getLogger(__name__)
log = logging.getLogger()
log.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)
rp2 = RP2Device()
rx8 = RX8Device()

filename = "dome_speakers.txt"
basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
filepath = os.path.join(basedir, filename)
spk_array = SpeakerArray(file=filepath)
spk_array.load_speaker_table()
noise = slab.Sound.pinknoise(duration=2.0, samplerate=24414)
rx8.handle.write("playbuflen", noise.samplerate*noise.duration, procs=rx8.handle.procs)


def play_sound(speaker):
    rx8.handle.write(tag="chan0", value=speaker.channel_analog, procs=f"{speaker.TDT_analog}{speaker.TDT_idx_analog}")
    rx8.handle.write(tag="data0", value=noise.data.flatten(), procs=f"{speaker.TDT_analog}{speaker.TDT_idx_analog}")
    rx8.start()
    rx8.pause()
    clear_channels()


def clear_channels():
    for idx in range(5):  # clear all speakers before loading warning tone
        rx8.handle.write(f"chan{idx}", 99, procs=["RX81", "RX82"])


if __name__ == "__main__":
    for speaker in spk_array.speakers:
        play_sound(speaker)
        time.sleep(1)

