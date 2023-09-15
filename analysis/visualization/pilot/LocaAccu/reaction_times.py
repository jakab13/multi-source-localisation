from analysis.utils.plotting import *
from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

'''
PLOTTING
'''
# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# get subject ids from dataframe
sub_ids = extract_subject_ids_from_dataframe(dfh)

# divide noise vs babble blocks
filledh = dfh["mode"].ffill()
noiseh = dfh[np.where(filledh=="noise", True, False)]  # True where reversed_speech is True
babbleh = dfh[np.where(filledh=="babble", True, False)]  # True where reversed_speech is False

# divide noise vs babble blocks
filledv = dfv["mode"].ffill()
noisev = dfv[np.where(filledv=="noise", True, False)]  # True where reversed_speech is True
babblev= dfv[np.where(filledv=="babble", True, False)]  # True where reversed_speech is False

for sub in sub_ids:
    plt.scatter(noisev.loc[sub].rt.mean(), noiseh.loc[sub].rt.mean())
plt.legend(sub_ids)
plt.xlabel("Reaction time vertical plane [s]")
plt.ylabel("Reaction time horizontal plane [s]")
plt.title("Mean Pinknoise Localization Reaction Time")

for sub in sub_ids:
    plt.scatter(babblev.loc[sub].rt.mean(), babbleh.loc[sub].rt.mean())
plt.legend(sub_ids)
plt.xlabel("Reaction time vertical plane [s]")
plt.ylabel("Reaction time horizontal plane [s]")
plt.title("Mean Babble Noise Localization Reaction Time")
