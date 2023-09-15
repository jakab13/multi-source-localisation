import numpy as np
import os
from labplatform.config import get_config
from analysis.utils.misc import load_dataframe
from analysis.utils import stats
from scipy.stats import wilcoxon

# permutation test
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfh = load_dataframe(fp, exp_name=exp_name, plane="v")
# divide reversed speech blocks from clear speech
filled = dfh.reversed_speech.ffill()
revspeech = dfh[np.where(filled == True, True, False)]  # True where reversed_speech is True
clearspeech = dfh[np.where(filled == False, True, False)]  # True where reversed_speech is False

# compare group means
group1 = revspeech.response.fillna(0)
group2 = clearspeech.response.fillna(0)

# permutation test
pval = stats.permutation_test(group1, group2)

# wilcoxon test
wilcoxon(group1, group2)
