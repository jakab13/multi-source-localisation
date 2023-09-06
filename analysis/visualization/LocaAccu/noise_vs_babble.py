from analysis.utils.plotting import *
from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

'''
PLOTTING
'''
# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# get subject ids from dataframe
sub_ids = extract_subject_ids_from_dataframe(dfh)

# divide noise vs babble blocks
filled = dfv["mode"].ffill()
noise = dfv[np.where(filled=="noise", True, False)]  # True where reversed_speech is True
babble = dfv[np.where(filled=="babble", True, False)]  # True where reversed_speech is False

layout = [["A", "B"],
          ["C", "D"]]

# set up mosaic plot
fig, ax = mosaic_plot(layout=layout)

# horizontal
draw_lineplot_actual_vs_perceived(data=noise, plane="vertical", sub_ids=sub_ids, ax=ax["A"])
ax["A"].set_title("Vertical noise localization accuracy")
ax["A"].sharey(ax["B"])

# vertical
draw_lineplot_actual_vs_perceived(data=babble, plane="vertical", sub_ids=sub_ids, ax=ax["B"])
ax["B"].set_title("Vertical babble localization accuracy")

# plot regression line horizontal
draw_linear_regression_actual_vs_perceived(data=noise, plane="vertical", ax=ax["C"])
ax["C"].set_title("Regression fit noise")

# plot regression line horizontal
draw_linear_regression_actual_vs_perceived(data=babble, plane="vertical", ax=ax["D"])
ax["D"].set_title("Regression fit babble")

# show
ax["A"].legend("")
ax["B"].legend("")
fig.tight_layout()
fig.show()
