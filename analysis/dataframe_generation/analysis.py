import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from analysis.dataframe_generation.post_processing import df_nj, df_la, df_su

col_order = ["horizontal", "vertical", "distance"]

hue_order_nj = df_nj[
    (df_nj["round"] == 2) &
    (df_nj.plane == "horizontal") &
    (df_nj["stim_type"] == "forward")].groupby(
    ["subject_id", "nj_slope"],
    as_index=False)["nj_slope"].mean().sort_values(by="nj_slope")["subject_id"].values

# LOCALISATION ACCURACY =============
df_curr = df_la[df_la["round"] == 2]
# df_curr = df_la
df_curr = df_curr.groupby(["subject_id", "plane", "stim_type", "stim_loc"], as_index=False)["resp_loc"].mean()
g = sns.FacetGrid(df_curr, col="plane", hue="stim_type", sharex=False, sharey=False)
g.map(sns.lineplot, "stim_loc", "resp_loc")

# df_curr = df_curr[df_curr.plane == "vertical"]
g = sns.FacetGrid(
    df_curr,
    col="plane",
    # row="stim_type",
    row_order=["babble", "noise"],
    hue="subject_id",
    # hue_order=hue_order_nj,
    palette="copper",
    sharex=False,
    sharey=False,
    col_order=col_order
)
g.map(sns.lineplot, "stim_loc", "resp_loc")
g.add_legend()

df_curr = df_la[df_la["round"] == 2]
g = sns.FacetGrid(
    df_curr,
    col="plane",
    sharex=False,
    sharey=False,
    col_order=col_order
)
g.map(sns.regplot, "nj_slope", "rmse")
g.add_legend()

# SPATIAL UNMASKING =============
df_curr = df_su[df_su["round"] == 2]
# df_curr = df_su
g = sns.FacetGrid(
    df_curr,
    col="plane",
    col_order=col_order,
    hue="subject_id",
    sharex=False
)
g.map(sns.lineplot, "masker_speaker_loc", "normed_threshold")
g.add_legend()


df_curr = df_su[df_su["round"] == 2]
# df_curr = df_su
g = sns.FacetGrid(
    df_curr,
    col="plane",
    col_order=col_order,
    # hue="subject_id",
    sharex=False
)
g.map(sns.lineplot, "masker_speaker_loc", "normed_threshold", errorbar=("ci", 95))
g.add_legend()

# NUMEROSITY JUDGEMENT =============
df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_nj
df_curr = df_curr.groupby(["subject_id", "plane", "stim_type", "stim_number"], as_index=False)["resp_number"].mean()
g = sns.FacetGrid(
    df_curr,
    col="plane",
    col_order=col_order,
    hue="stim_type"
)
g.map(sns.lineplot, "stim_number", "resp_number", errorbar="sd")
g.add_legend()
plt.ylim(1.8, 6.2)
plt.xlim(1.8, 6.2)


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
    # col="subject_id",
    # col_wrap=4,
    hue="plane"
)
g.map(sns.lineplot, "stim_number", "resp_number")
plt.ylim(1.8, 6.2)
plt.xlim(1.8, 6.2)
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

df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr.sort_values(by="nj_slope"),
    col="plane",
    # row="stim_type",
    hue="subject_id",
    # hue_order=hue_order_nj,
    # palette="copper",
    col_order=col_order,
    row_order=["forward", "reversed"]
)
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()
g.set_titles(template="{col_name}")
g.set_xlabels(label="", clear_inner=True)
g.set_ylabels(label='n perceived')
plt.ylim(1.8, 6.2)
plt.xlim(1.8, 6.2)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Auditory Numerosity Judgement in 3D")
g.fig.supxlabel("n presented", fontsize=13)

df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_curr[df_curr["stim_number"] == 5]
df_curr = df_curr[df_curr["subject_id"] == "sub_110"]
g = sns.FacetGrid(
    df_curr,
    hue="stim_number",
    # hue_order=hue_order_nj,
    # palette="copper",
    col="plane",
    # row="stim_number",
    col_order=col_order,
    # sharex=False,
    # sharey=False
)
g.map(sns.regplot, "spectral_coverage", "abs_error", scatter=False)
g.add_legend()



df_curr_group = df_curr.groupby(["subject_id", "plane"], as_index=False)["nj_slope"].mean()

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
