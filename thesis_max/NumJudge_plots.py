from analysis.utils.plotting import *
import os
from labplatform.config import get_config
import seaborn as sns
import scienceplots
plt.style.use(['science', 'nature'])


# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

sub_ids_h = extract_subject_ids_from_dataframe(dfh)
sub_ids_v = extract_subject_ids_from_dataframe(dfv)

# divide reversed speech blocks from clear speech
filledv = dfv.reversed_speech.ffill()
revspeechv = dfv[np.where(filledv==True, True, False)]  # True where reversed_speech is True
clearspeechv = dfv[np.where(filledv==False, True, False)]  # True where reversed_speech is False

filledh = dfh.reversed_speech.ffill()
revspeechh = dfh[np.where(filledh==True, True, False)]  # True where reversed_speech is True
clearspeechh = dfh[np.where(filledh==False, True, False)]  # True where reversed_speech is False

layout = []
