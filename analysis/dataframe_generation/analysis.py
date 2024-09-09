import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplcursors import cursor
import scipy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from analysis.dataframe_generation.utils import add_grid_line_of_equality
from analysis.dataframe_generation.post_processing import df_nj, df_la, df_su

col_order = ["horizontal", "vertical", "distance"]

hue_order_nj = df_nj[
    (df_nj["round"] == 2) &
    (df_nj.plane == "horizontal") &
    (df_nj["stim_type"] == "forward")].groupby(
    ["subject_id", "nj_rmse"],
    as_index=False)["nj_rmse"].mean().sort_values(by="nj_rmse", ascending=False)["subject_id"].values

hue_order_su = df_nj[
    (df_nj["round"] == 2) &
    (df_nj.plane == "horizontal") &
    (df_nj["stim_type"] == "forward")].groupby(
    ["subject_id", "su_slope"],
    as_index=False)["su_slope"].mean().sort_values(by="su_slope")["subject_id"].values

# LOCALISATION ACCURACY =============
df_curr = df_la[df_la["round"] == 2]
# df_curr = df_la
g = sns.FacetGrid(
    df_curr,
    col="plane",
    # row="stim_type",
    # row_order=["babble", "noise"],
    hue="subject_id",
    sharex=False,
    sharey=False,
    col_order=col_order
)
g.map(sns.lineplot, "stim_loc", "resp_loc", errorbar=None)
add_grid_line_of_equality(g)
g.add_legend()
cursor(hover=True)


df_curr = df_la[df_la["round"] == 2]
df_curr = df_curr.groupby(["subject_id", "plane", "stim_loc"], as_index=False)["resp_loc"].mean()
g = sns.FacetGrid(df_curr, col="plane", sharex=False, sharey=False, col_order=col_order)
g.map(sns.lineplot, "stim_loc", "resp_loc", errorbar="sd", color="black")
add_grid_line_of_equality(g, task_name="la")
g.set_titles(template="{col_name}")
g.set_xlabels(label="Presented location")
g.set_ylabels(label='Perceived location')
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

# SPATIAL UNMASKING ===============================================================================
df_curr = df_su[df_su["round"] == 2]
# df_curr = df_curr[df_curr["subject_id"] == "sub_109"]
# df_curr = df_su
for subject_id in df_curr.subject_id.unique():
    g = sns.FacetGrid(
        df_curr[df_curr.subject_id == subject_id],
        col="plane",
        col_order=col_order,
        sharex=False
    )
    g.map(sns.lineplot, "masker_speaker_loc", "normed_threshold")
    g.add_legend()
    plt.title(subject_id)
    plt.show()


df_curr = df_su[df_su["round"] == 2]
# df_curr = df_su[~(df_su["subject_id"] == "sub_102")]
g = sns.FacetGrid(
    df_curr,
    col="plane",
    col_order=col_order,
    hue="subject_id",
    hue_order=hue_order_su,
    palette="copper",
    sharex=False,
    height=6,
    aspect=.8
)
g.map(plt.axhline, y=0, ls='--', c='red', alpha=.2)
# g.map(sns.lineplot, "masker_speaker_loc", "normed_threshold", errorbar=("ci", 95), color="black")
g.map(sns.lineplot, "masker_speaker_loc_abs", "threshold", errorbar=None)
# for ax in g.axes.flatten():
#     ax_min = 1.5 if "distance" in ax.get_title() else -55
#     ax_max = 12.5 if "distance" in ax.get_title() else 55
#     ax.set_xlim(ax_min, ax_max)
#     ax.set_ylim(-15, 3)
g.add_legend()
g.set_titles(template="{col_name}")
title = f"Spatial unmasking thresholds in 3D"
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle(title)
# plt.savefig(title + ".svg", format="svg")
cursor(hover=True)

# NUMEROSITY JUDGEMENT ===============================================================================
def get_stim_type_diff(row):
    other_stim_resp = df_curr[(df_curr.subject_id == row.subject_id) &
                              (df_curr.plane == row.plane) &
                              (df_curr.stim_type != row.stim_type) &
                              (df_curr.stim_number == row.stim_number)]["resp_number"]
    diff = other_stim_resp.values[0] - row["resp_number"]
    return diff

df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_nj
df_curr = df_curr.groupby(["subject_id", "plane", "stim_type", "stim_number"], as_index=False)["resp_number"].mean()
# df_curr["resp_diff"] = df_curr.apply(lambda row: get_stim_type_diff(row), axis=1)
df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr,
    # col="plane",
    # col_order=col_order,
    hue="plane",
    # hue_order=hue_order_nj,
    # palette="copper"
)
g.map(sns.lineplot, "stim_number", "resp_number")
g.add_legend()
# plt.ylim(1.8, 6.2)
# plt.xlim(1.8, 6.2)


df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr.groupby(["subject_id", "plane", "stim_type", "stim_number"], as_index=False)["resp_number"].mean(),
    col="plane",
    col_order=col_order,
    # hue="stim_type",
    height=6,
    aspect=.8
)
g.map(sns.lineplot, "stim_number", "resp_number", errorbar="sd", color="black")
add_grid_line_of_equality(g, task_name="nj")
g.set_titles(template="{col_name}")
g.add_legend()
title = f"Numerosity judgement in 3D"
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle(title)
# plt.savefig(title + ".svg", format="svg")



df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_curr[df_curr["stim_type"] == "forward"]
g = sns.FacetGrid(
    df_curr,
    col="stim_type",
    # col_wrap=4,
    hue="plane"
)
g.map(sns.lineplot, "stim_number", "resp_number")
plt.ylim(1.8, 6.2)
plt.xlim(1.8, 6.2)
g.add_legend()
add_grid_line_of_equality(g)

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
df_curr = df_curr.sort_values(by="nj_slope")
g = sns.FacetGrid(
    df_curr,
    col="plane",
    # row="stim_type",
    # row_order=["forward", "reversed"],
    hue="subject_id",
    hue_order=hue_order_nj,
    palette="copper",
    col_order=col_order
)
g.map(sns.lineplot, "stim_number", "resp_number", errorbar=None, linewidth=2)
g.set(xticks=[2, 3, 4, 5, 6])
g.set(yticks=[2, 3, 4, 5, 6])
add_grid_line_of_equality(g)
g.add_legend()
g.set_titles(template="{col_name}")
g.set_xlabels(label="n presented")
g.set_ylabels(label='n perceived')
plt.ylim(1.8, 6.2)
plt.xlim(1.8, 6.2)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Auditory Numerosity Judgement in 3D")
cursor(hover=True)


df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
for subject_id in df_curr.subject_id.unique():
    g = sns.FacetGrid(
        # df_curr[df_curr.subject_id == subject_id],
        df_curr,
        col="plane",
        hue="spectral_coverage_binned",
        palette="winter",
        col_order=col_order,
        height=5,
        aspect=0.8
    )
    g.map(sns.lineplot, "stim_number", "resp_number", errorbar=None, linewidth=2)
    g.set(xticks=[2, 3, 4, 5, 6])
    g.set(yticks=[2, 3, 4, 5, 6])
    add_grid_line_of_equality(g, task_name="nj")
    g.add_legend()
    g.set_titles(template="{col_name}")
    g.set_xlabels(label="n presented")
    g.set_ylabels(label='n perceived')
    plt.ylim(1.8, 6.2)
    plt.xlim(1.8, 6.2)
    g.fig.subplots_adjust(top=0.85)
    # title = f"Auditory Numerosity Judgement in 3D ({subject_id})"
    title = f"Auditory Numerosity Judgement in 3D (all subjects)"
    g.fig.suptitle(title)
    cursor(hover=True)
    # plt.savefig(title)


df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
df_curr = df_curr.groupby(["subject_id", "plane"], as_index=False)["su_slope"].mean().sort_values(by="su_slope")
g = sns.catplot(
    df_curr,
    y="su_slope",
    col="plane",
    col_order=col_order,
    hue="subject_id",
    hue_order=hue_order_su,
    palette="copper",
    aspect=0.5,
    size=10
)
g.set_titles(template="{col_name}")
cursor(hover=True)


df_curr = df_nj[df_nj["round"] == 2]
# df_curr = df_curr[df_curr["stim_type"] == "forward"]
for subject_id in df_curr.subject_id.unique():
    g = sns.FacetGrid(
        df_curr[df_curr.subject_id == subject_id],
        # df_curr,
        col="plane",
        row="stim_type",
        hue="stim_number",
        palette="winter",
        col_order=col_order,
        # row_order=col_order,
        height=4,
        aspect=0.8
    )
    g.map(sns.regplot, "spectral_coverage_neg40", "resp_number", scatter=False)
    g.add_legend()
    g.set_titles(template="{col_name}")
    title = f"Dependence on spectral_coverage ({subject_id})"
    # title = "Error vs spectral_coverage"
    # title = "Dependence on spectral_coverage"
    g.fig.suptitle(title)
    plt.savefig("figures/" + title + ".png")
    plt.close()
    # plt.show()



df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
df_curr = df_curr.groupby(["subject_id", "plane", "stim_type", "stim_number"], as_index=False)["spectral_coverage_slope"].mean()
g = sns.FacetGrid(
    df_curr[df_curr["stim_type"] == "forward"],
    palette="copper",
    # hue="subject_id",
    # hue_order=hue_order_nj,
    col="plane",
    col_order=col_order
)
g.map(sns.pointplot, "stim_number", "spectral_coverage_slope")
g.add_legend()
cursor(hover=True)


df_curr = df_nj[df_nj["round"] == 2]
df_curr = df_curr[df_curr["stim_type"] == "forward"]
sns.displot(df_curr, x="spectral_coverage", hue="stim_number", kind="kde", col="plane", palette="winter", col_order=col_order)
g.add_legend()
cursor(hover=True)

# SYNTHESIS ===================================================================================================

df_curr = df_la[df_la["round"] == 2]
df_curr = df_curr.groupby(["subject_id", "plane", "nj_slope"], as_index=False)["la_slope"].mean().dropna()
g = sns.FacetGrid(
    df_curr,
    col="plane",
    sharex=False,
    sharey=False,
    col_order=col_order
)
g.map(sns.regplot, "la_slope", "nj_slope")
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

df_curr = df_nj[df_nj["round"] == 2]
lmm_log = smf.mixedlm("resp_number ~ np.log(stim_number) + C(stim_type) + C(plane)", df_curr, groups="subject_id",
                       re_formula="~1 + np.log(stim_number) + C(stim_type) + C(plane)").fit()
print(lmm_log.summary())
