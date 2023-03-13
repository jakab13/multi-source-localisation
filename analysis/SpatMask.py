from analysis.utils.plotting import *
from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Speakers.speaker_config import SpeakerArray
sns.set_theme()


# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "SpatMask"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")
sub_ids_h = extract_subject_ids_from_dataframe(dfh)

# calculate threshold for all staircases
threshs = list()
order = list()
speaker_ids = [2, 8, 15, 31, 38, 44]
basedir = os.path.join(get_config(setting="BASE_DIRECTORY"), "speakers")
filepath = os.path.join(basedir, "dome_speakers.txt")
spks = SpeakerArray(file=filepath)
spks.load_speaker_table()
spks = spks.pick_speakers(picks=speaker_ids)


for sub in sub_ids_h:
    reversal_intensities = list()
    order.append(dfh.loc[sub].sequence.dropna()[0]["trials"])
    for stair in dfh.loc[sub].stairs.dropna():
        reversal_intensities.append(np.mean(stair["reversal_intensities"]))
    threshs.append(reversal_intensities)

