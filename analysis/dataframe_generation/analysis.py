import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from analysis.dataframe_generation.post_processing import df_nj, df_la

sns.pointplot(df_nj, x="stim_number", y="resp_number", hue="subject_id")

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
