import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from analysis.dataframe_generation.post_processing import df_nj, df_la, df_su

# LOCALISATION ACCURACY =============
df_curr = df_la[df_la["round"] == 2]
# df_curr = df_la
g = sns.FacetGrid(df_curr, col="plane", hue="stim_type", sharex=False, sharey=False)
g.map(sns.lineplot, "stim_loc", "resp_loc")

df_curr = df_curr[df_curr.plane == "vertical"]
g = sns.FacetGrid(df_curr, col="plane", hue="subject_id", sharex=False, sharey=False)
g.map(sns.lineplot, "stim_loc", "resp_loc")
g.add_legend()

# SPATIAL UNMASKING =============
df_curr = df_su[df_su["round"] == 2]
df_curr = df_curr[df_curr.plane == "horizontal"]
df_curr = df_curr[df_curr.plane == "vertical"]
df_curr = df_curr[df_curr.plane == "distance"]
# df_curr = df_su
g = sns.FacetGrid(df_curr, col="plane", sharex=False)
g.map(sns.lineplot, "masker_speaker_loc", "threshold")

df_curr = df_curr[df_curr.plane == "vertical"]
sns.lineplot(df_curr, x="masker_speaker_loc", y="threshold")
sns.lineplot(df_curr, x="masker_speaker_loc", y="threshold", hue="subject_id")

# NUMEROSITY JUDGEMENT =============
df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_su
g = sns.FacetGrid(df_curr, col="plane", hue="stim_type")
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()


df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr,
    col="plane",
    hue="subject_id"
)
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()

df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr,
    col="subject_id",
    col_wrap=4,
    hue="plane"
)
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()

df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_nj
df_curr = df_curr[df_curr["plane"] == "horizontal"]
g = sns.FacetGrid(
    df_curr,
    col="subject_id",
    col_wrap=4,
    hue="stim_type"
)
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()

sns.lineplot(df_nj[(df_nj["round"] == 2) & (df_nj["stim_type"] == "forward")], x="stim_number", y="resp_number", hue="plane")


lm_glob = smf.ols('resp_number ~ stim_number', df_nj).fit()
print("MSE=%.3f" % lm_glob.mse_resid)
print(lm_glob.t_test('stim_number'))
sns.lmplot(x="stim_number", y="resp_number", data=df_nj)

lmm = smf.mixedlm("resp_number ~ stim_number",
                  data=df_nj,
                  groups="subject_id").fit()
print(lmm.summary())

lmm_lin = smf.mixedlm("resp_number ~ stim_number + C(stim_type) + C(plane)", df_nj, groups="subject_id",
                       re_formula="~1 + stim_number").fit()
print(lmm_lin.summary())

lmm_log = smf.mixedlm("resp_number ~ np.log(stim_number) + C(stim_type) + C(plane)", df_nj, groups="subject_id",
                       re_formula="~1 + np.log(stim_number) + C(stim_type) + C(plane)").fit()
print(lmm_log.summary())
