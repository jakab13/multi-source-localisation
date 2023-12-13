import numpy as np
import os
from labplatform.config import get_config
from analysis.utils.misc import load_dataframe
from analysis.utils import stats, math
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", palette="Dark2")
plt.rcParams['text.usetex'] = True  # TeX rendering


# permutation test
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")

# divide reversed speech blocks from clear speech
filledh = dfh.reversed_speech.ffill()
revspeechh = dfh[np.where(filledh == True, True, False)]  # True where reversed_speech is True
clearspeechh = dfh[np.where(filledh == False, True, False)]  # True where reversed_speech is False

filledv = dfv.reversed_speech.ffill()
revspeechv = dfv[np.where(filledv == True, True, False)]  # True where reversed_speech is True
clearspeechv = dfv[np.where(filledv == False, True, False)]  # True where reversed_speech is False

# compare group means
group1h = revspeechh.response.fillna(0)
group2h = clearspeechh.response.fillna(0)

group1v = revspeechv.response.fillna(0)
group2v = clearspeechv.response.fillna(0)


pvalh = stats.permutation_test(group1h, group2h)
pvalv = stats.permutation_test(group1v, group2v)


# wilcoxon test
wilcoxon(group1h, group2h)
wilcoxon(group1v, group2v)

