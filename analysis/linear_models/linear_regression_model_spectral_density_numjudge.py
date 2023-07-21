from analysis.utils.plotting import *
import os
import seaborn as sns
import pickle as pkl
from labplatform.config import get_config
sns.set_theme()
from sklearn.linear_model import LinearRegression


# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False

# get sub ids
sub_ids = extract_subject_ids_from_dataframe(dfh)

# get covariate
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# prepare dataframe for the model
df = pd.DataFrame(columns=["solution", "response", "coverage"])
df.solution = clearspeech_v.solution
df.response = clearspeech_v.response
df.coverage = coverage.loc["clearspeech_v"]["coverage"]
df = df.fillna(0)

# model
x = df.solution.values.reshape(-1, 1)
y = df.coverage.values.reshape(-1, 1)
model = LinearRegression(fit_intercept=False)
model.fit(x, y)
ypred = model.predict(x)
model.score(x, y)

plt.scatter(x, y)
plt.plot(x, ypred)
