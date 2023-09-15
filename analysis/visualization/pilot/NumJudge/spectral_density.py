from analysis.utils.plotting import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from labplatform.config import get_config
import librosa
from sklearn.metrics import auc
sns.set_theme()

# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
revspeech_h = revspeech_h.sort_index()
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False
clearspeech_h = clearspeech_h.sort_index()

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
revspeech_v = revspeech_v.sort_index()
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False
clearspeech_v = clearspeech_v.sort_index()

# load dataframe containing coverage
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# plot results
sns.lineplot(x=clearspeech_h.solution, y=coverage.loc["clearspeech_h"]["coverage"], label="clear speech_h")
sns.lineplot(x=revspeech_h.solution, y=coverage.loc["revspeech_h"]["coverage"], label="reversed speech_h")
plt.xlabel("Sound source perceptional discrepancy")
plt.ylabel("Coverage")

# vertical
sns.lineplot(x=clearspeech_v.solution, y=coverage.loc["clearspeech_v"]["coverage"], label="clear speech_v")
sns.lineplot(x=revspeech_v.solution, y=coverage.loc["revspeech_v"]["coverage"], label="reversed speech_v")
plt.xlabel("Solution")
plt.ylabel("Coverage")
