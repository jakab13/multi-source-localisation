import seaborn as sns
import matplotlib.pyplot as plt
from analysis.utils.misc import *


def mosaic_plot(layout, **kwargs):
    fig, ax = plt.subplot_mosaic(mosaic=layout, **kwargs)
    return fig, ax


def draw_lineplot_actual_vs_perceived(data, sub_ids, plane, ax=None, errorbar=("sd", 1), **kwargs):
    for sub_id in sub_ids:
        sub = data.loc[sub_id].reset_index()
        if plane == "horizontal":
            actual = get_azimuth_from_df(sub.actual)
            perceived = get_azimuth_from_df(sub.perceived)
        elif plane == "vertical":
            actual = get_elevation_from_df(sub.actual)
            perceived = get_elevation_from_df(sub.perceived)
        else:
            raise ValueError("Plane parameter must be vertical or horizontal")
        if ax:
            sns.lineplot(x=actual,
                         y=perceived,
                         err_style="bars",
                         errorbar=errorbar,
                         label=sub_id,
                         ax=ax,
                         **kwargs)
        else:
            sns.lineplot(x=actual,
                         y=perceived,
                         err_style="bars",
                         errorbar=errorbar,
                         label=sub_id,
                         **kwargs)


def draw_lineplot_solution_vs_response(data, sub_ids, ax=None, errorbar=("sd", 1), **kwargs):
    for sub_id in sub_ids:
        sub = data.loc[sub_id].reset_index()
        if ax:
            sns.lineplot(x=sub.solution,
                         y=sub.response,
                         err_style="bars",
                         errorbar=errorbar,
                         label=sub_id,
                         ax=ax,
                         **kwargs)
        else:
            sns.lineplot(x=sub.solution,
                         y=sub.response,
                         err_style="bars",
                         errorbar=errorbar,
                         label=sub_id,
                         **kwargs)


def draw_boxplot_actual_vs_perceived(data, sub_ids, plane, ax=None, **kwargs):
    for sub_id in sub_ids:
        sub = data.loc[sub_id].reset_index()
        if plane == "horizontal":
            actual = get_azimuth_from_df(sub.actual)
            perceived = get_azimuth_from_df(sub.perceived)
        elif plane == "vertical":
            actual = get_elevation_from_df(sub.actual)
            perceived = get_elevation_from_df(sub.perceived)
        else:
            raise ValueError("Plane parameter must be vertical or horizontal")
        actual = replace_in_array(actual)
        perceived = replace_in_array(perceived)
        if ax:
            sns.boxplot(x=actual,
                        y=perceived,
                        ax=ax,
                        **kwargs)
        else:
            sns.boxplot(x=actual,
                        y=perceived,
                        **kwargs)


def draw_linear_regression_actual_vs_perceived(data, plane, ax=None, axlim=(-75, 75), **kwargs):
    if plane == "horizontal":
        x = get_azimuth_from_df(data.actual)
        y = get_azimuth_from_df(data.perceived)
    elif plane == "vertical":
        x = get_elevation_from_df(data.actual)
        y = get_elevation_from_df(data.perceived)
    elif plane == "all":
        x = list()
        y = list()
        x.append(get_elevation_from_df(data.actual))
        y.append(get_elevation_from_df(data.perceived))
        x.append(get_azimuth_from_df(data.actual))
        y.append(get_azimuth_from_df(data.perceived))
        xflat = [item for sublist in x for item in sublist]
        yflat = [item for sublist in y for item in sublist]
    else:
        raise ValueError("Plane parameter must be vertical or horizontal")
    if plane == "all":
        xflat = replace_in_array(xflat)
        yflat = replace_in_array(yflat)
        if ax:
            sns.regplot(x=xflat,
                        y=yflat,
                        ax=ax,
                        **kwargs)
            ax.set_ylim(axlim)
        else:
            sns.regplot(x=xflat,
                        y=yflat,
                        ax=ax,
                        **kwargs)
    else:
        x = replace_in_array(x)
        y = replace_in_array(y)
        if ax:
            sns.regplot(x=x,
                        y=y,
                        ax=ax,
                        **kwargs)
            ax.set_ylim(axlim)
        else:
            sns.regplot(x=x,
                        y=y,
                        ax=ax,
                        **kwargs)
