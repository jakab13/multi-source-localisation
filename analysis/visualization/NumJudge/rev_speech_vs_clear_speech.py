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

# divide reversed speech blocks from clear speech
filled = dfv.reversed_speech.ffill()
revspeech = dfv[np.where(filled==True, True, False)]  # True where reversed_speech is True
clearspeech = dfv[np.where(filled==False, True, False)]  # True where reversed_speech is False

# for each plane, plot revspeech and clearspeech results
layout = [["A", "C"],
          ["B", "C"]]
fig, ax = mosaic_plot(layout)

# plot response vs solution lineplot
draw_lineplot_solution_vs_response(data=revspeech, sub_ids=sub_ids_h, ax=ax["A"])
ax["A"].set(title="reversed speech vertical")

# vertical
draw_lineplot_solution_vs_response(data=clearspeech, sub_ids=sub_ids_h, ax=ax["B"])
ax["B"].set(title="clear speech vertical")

# plot mean
sns.lineplot(x=revspeech.solution.reset_index(drop=True),
             y=revspeech.response.reset_index(drop=True),
             err_style="bars",
             errorbar=("se", 1),
             label="reversed speech",
             ax=ax["C"])
sns.lineplot(x=clearspeech.solution.reset_index(drop=True),
             y=clearspeech.response.reset_index(drop=True),
             err_style="bars",
             errorbar=("se", 1),
             label="clear speech",
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
ax["A"].legend("")
ax["B"].legend("")
plt.tight_layout
plt.show()
