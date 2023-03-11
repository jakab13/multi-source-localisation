import seaborn as sns
import matplotlib.pyplot as plt
from analysis.utils.misc import *


def mosaic_plot(layout):
    fig, ax = plt.subplot_mosaic(layout=layout)
    return fig, ax


def draw_lineplot_actual_vs_perceived(data, sub_ids, ax, errorbar=("se", 1)):
    for sub_id in sub_ids:
        sub = df.loc[sub_id].reset_index()
        actual = get_azimuth_from_df(sub.actual)
        perceived = get_azimuth_from_df(sub.perceived)
        sns.lineplot(x=actual,
                     y=perceived,
                     err_style="bars",
                     errorbar=errorbar,
                     label=sub_id,
                     ax=ax)
