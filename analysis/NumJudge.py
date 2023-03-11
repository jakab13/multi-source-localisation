from analysis.utils.misc import *
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

# create mosaic plot
fig, ax = plt.subplot_mosaic(layout=[["A", "B"],
                                     ["C", "C"]])
for idx, sub_id in enumerate(sub_ids_h):
    sub = dfh.loc[sub_id].reset_index()
    sns.lineplot(x=sub.solution,
                 y=sub.response,
                 err_style="bars",
                 errorbar=("se", 1),
                 label=sub_id,
                 ax=ax["A"]).set(title="Horizontal")
for idx, sub_id in enumerate(sub_ids_v):
    sub = dfv.loc[sub_id].reset_index()
    sns.lineplot(x=sub.solution,
                 y=sub.response,
                 err_style="bars",
                 errorbar=("se", 1),
                 label=sub_id,
                 ax=ax["B"]).set(title='Vertical')

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
plt.xticks(range(2, 6))
plt.yticks(range(2, 6))
x0, x1 = plt.xlim()
y0, y1 = plt.ylim()
lims = [max(x0, y0), min(x1, y1)]
sns.lineplot(x=lims, y=lims, color='grey', linestyle="dashed", ax=ax["C"])
for val in ax.values():
    val.invert_yaxis()
plt.tight_layout()
plt.legend()
plt.show()
