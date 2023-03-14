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
sub_ids_h = extract_subject_ids_from_dataframe(dfh)
sub_ids_v = extract_subject_ids_from_dataframe(dfv)

# define mosaic layout
layout = [["A", "B"],
          ["C", "D"],
          ["E", "E"]]

# set up mosaic plot
fig, ax = mosaic_plot(layout=layout)

# horizontal
draw_lineplot_actual_vs_perceived(data=dfh, plane="horizontal", sub_ids=sub_ids_h, ax=ax["A"])
ax["A"].set_title("Horizontal localization accuracy")
ax["A"].sharey(ax["B"])

# vertical
draw_lineplot_actual_vs_perceived(data=dfv, plane="vertical", sub_ids=sub_ids_v, ax=ax["B"])
ax["B"].set_title("Vertical localization accuracy")

# plot regression line horizontal
draw_linear_regression_actual_vs_perceived(data=dfh, plane="horizontal", ax=ax["C"])
ax["C"].set_title("Regression fit horizontal")

# plot regression line horizontal
draw_linear_regression_actual_vs_perceived(data=dfv, plane="vertical", ax=ax["D"])
ax["D"].set_title("Regression fit vertical")

# plot regression line of all planes and subjects
df = pd.concat([dfh, dfv])
draw_linear_regression_actual_vs_perceived(data=df, plane="all", ax=ax["E"])
ax["E"].set_title("Regression fit mean")

# show
fig.tight_layout()
fig.show()
