from analysis.utils.plotting import *
from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# get subject ids from dataframe
sub_ids_h = extract_subject_ids_from_dataframe(dfh)
sub_ids_v = extract_subject_ids_from_dataframe(dfv)

# plot localization accuracy for single subjects
fig, ax = mosaic_plot(layout=[["A", "B"],
                              ["C", "D"],
                              ["E", "E"]])

# horizontal
draw_lineplot_actual_vs_perceived(data=dfh, sub_ids=sub_ids_h, ax=ax["A"])
ax["A"].set_title("Horizontal localization accuracy")

# vertical
draw_lineplot_actual_vs_perceived(data=dfv, sub_ids=sub_ids_v, ax=ax["B"])
ax["B"].set_title("Vertical localization accuracy")

# plot regression line horizontal and vertical
x = get_azimuth_from_df(dfh.actual)
y = get_azimuth_from_df(dfh.perceived)
for i, val in enumerate(x):
    if val == None:
        x[i] = 0
for i, val in enumerate(y):
    if val == None:
        y[i] = 0
sns.regplot(x=x,
            y=y,
            ax=ax["C"])
ax["C"].set_title("Regression fit horizontal")
ax["C"].set_ylim([-75, 75])

x = get_elevation_from_df(dfv.actual)
y = get_elevation_from_df(dfv.perceived)
for i, val in enumerate(x):
    if val == None:
        x[i] = 0
for i, val in enumerate(y):
    if val == None:
        y[i] = 0
sns.regplot(x=x,
            y=y,
            ax=ax["D"])
ax["D"].set_title("Regression fit vertical")

# plot regression line of all planes and subjects
df = pd.concat([dfh, dfv])
x = get_azimuth_from_df(df.actual)
y = get_azimuth_from_df(df.perceived)
for i, val in enumerate(x):
    if val == None:
        x[i] = 0
for i, val in enumerate(y):
    if val == None:
        y[i] = 0
sns.regplot(x=x,
            y=y,
            ax=ax["E"])
ax["E"].set_title("Regression fit mean")
ax["E"].set_ylim([-75, 75])

# show
fig.tight_layout()
fig.show()
