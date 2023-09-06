from analysis.utils.plotting import *
import os
import seaborn as sns
import pickle as pkl
from labplatform.config import get_config
sns.set_theme()
from statsmodels.formula.api import ols


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

# get covariate
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# prepare dataframe for the model
df = pd.DataFrame(columns=["solution", "response", "coverage", "performance"])
df.solution = clearspeech_h.solution
df.response = clearspeech_h.response
df.coverage = coverage.loc["clearspeech_h"]["coverage"]
performance = pd.read_csv("/home/max/labplatform/data/csv/numjudge_performance_clearspeech_h.csv", index_col=1)
performance.pop("Sub_ID")
performance = [x[0] for x in performance.values.tolist()]
df.performance = performance
df = df.fillna(0)


"""
The interesting fact is that the ANCOVA is literally the same procedure as doing a linear regression:
response = Beta0 + Beta1 x solution + Beta2 x coverage
The corresponding values for the coefficients can be taken from the ANOVA table.
"""

# linear model
model = ols('response ~ solution + coverage', data=df).fit()
model.summary()
