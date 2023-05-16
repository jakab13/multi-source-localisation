from analysis.utils.plotting import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

sub_ids_h = extract_subject_ids_from_dataframe(dfh)
sub_ids_v = extract_subject_ids_from_dataframe(dfv)

layout = [["A", "B"],
          ["C", "C"]]

# create mosaic plot
fig, ax = mosaic_plot(layout)

# plot response vs solution lineplot
draw_lineplot_solution_vs_response(data=dfh, sub_ids=sub_ids_h, ax=ax["A"])
ax["A"].set(title="Horizontal")

# vertical
draw_lineplot_solution_vs_response(data=dfv, sub_ids=sub_ids_v, ax=ax["B"])
ax["B"].set(title="Vertical")

# plot mean
sns.lineplot(x=dfv.solution.reset_index(drop=True),
             y=dfv.response.reset_index(drop=True),
             err_style="bars",
             errorbar=("se", 1),
             label="vertical",
             ax=ax["C"])
sns.lineplot(x=dfh.solution.reset_index(drop=True),
             y=dfh.response.reset_index(drop=True),
             err_style="bars",
             errorbar=("se", 1),
             label="horizontal",
             ax=ax["C"])
plt.xticks(range(2, 7))
plt.yticks(range(2, 7))
x0, x1 = plt.xlim()
y0, y1 = plt.ylim()
lims = [max(x0, y0), min(x1, y1)]
sns.lineplot(x=lims, y=lims, color='grey', linestyle="dashed", ax=ax["C"])
for val in ax.values():
    val.invert_yaxis()

ax["A"].sharex(ax["B"])
ax["A"].sharey(ax["B"])
# plt.tight_layout()
plt.legend()
plt.show()

# plot confusion matrix
layout = """
ab
"""
fig, ax = mosaic_plot(layout)
fig.suptitle("NumJudge Crosstab")

# vertical
cmv = crosstab(index=dfv["response"], columns=dfv["solution"], rownames=["response"], colnames=["solution"])
cmv = cmv.drop(index=1)
sns.heatmap(cmv, annot=True, ax=ax["a"])
ax["a"].set_title("Vertical")

# horizontal
cmh = crosstab(index=dfh["response"], columns=dfh["solution"], rownames=["response"], colnames=["solution"])
cmh = cmh.drop(index=8)
cmh = cmh.drop(index=1)
cmh = cmh.drop(index=9)
sns.heatmap(cmh, annot=True, ax=ax["b"])
ax["b"].set_title("Horizontal")
