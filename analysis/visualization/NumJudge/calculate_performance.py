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

# calculate performance
performance = clearspeech_h.solution - clearspeech_h.response
# some processing
performance = performance.abs().fillna(0)  # fill NaN with 0 and make everything positive

df = pd.DataFrame(performance, columns=["performance_clear_h"])
tofp = "/home/max/labplatform/data/csv/numjudge_performance_clearspeech_h.csv"
df.to_csv(tofp)
