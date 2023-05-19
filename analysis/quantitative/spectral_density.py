from analysis.utils.plotting import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
sns.set_theme()

# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# get sub ids
sub_ids = extract_subject_ids_from_dataframe(dfh)

# get talker files
talker_files_path = os.path.join(get_config("DATA_ROOT"), "MSL", "used_sounds_pkl", "numjudge_talker_files_clear.pkl")
with open(talker_files_path, "rb") as files:
    sounds = pkl.load(files)

# get info from trials
speakers_sample = dfh.speakers_sample  # indices of speakers from speakertable
signals_sample = dfh.signals_sample  # talker IDs
country_idxs = dfh.country_idxs  # indices of the country names from a talker

# extract envelope
fig, ax = plt.subplots()
for sound in sounds["229"]:
    sound.envelope().plot_samples(show=False)

