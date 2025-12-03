import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
# from analysis.dataframe_generation.post_processing import df_nj, df_la, df_su

df_la = pd.read_csv("DataFrames/locaaccu_post_processed.csv")
df_su = pd.read_csv("DataFrames/spatmask_post_processed.csv")
df_nj = pd.read_csv("DataFrames/numjudge_post_processed.csv")

def get_slope(presented, perceived):
    # print(presented, perceived)
    linreg = scipy.stats.linregress(presented, perceived)
    slope = linreg.slope
    return slope


def get_rmse(presented, perceived):
    mse = mean_squared_error(presented, perceived)
    rmse = np.sqrt(mse)
    return rmse


def get_power_exponent(presented, perceived):
    k, a = scipy.optimize.curve_fit(lambda x_, k_, a_: k_ * np.power(x_, a_),  presented,  perceived)[0]
    return a


df_la = df_la[df_la["round"] == 2]
df_la_group = df_la.groupby(["subject_id", "plane"])
df_performance = df_la_group.apply(lambda row: get_slope(row.stim_loc, row.resp_loc)).reset_index(name='la_slope')
df_performance["la_rmse"] = df_la_group.apply(
    lambda row: get_rmse(row.stim_loc, row.resp_loc)).reset_index(name='la_rmse')["la_rmse"]
df_performance["la_power_exponent"] = df_la_group.apply(
    lambda row: get_power_exponent(row.stim_loc, row.resp_loc)).reset_index(name='la_power_exponent')["la_power_exponent"]

df_su = df_su[df_su["round"] == 2]
df_su = df_su.dropna(subset=["masker_speaker_loc"])
df_su_group = df_su.groupby(["subject_id", "plane"])
df_performance["su_slope"] = df_su_group.apply(
    lambda row: get_slope(row.masker_speaker_loc_abs, row.normed_threshold)).reset_index(name='su_slope')["su_slope"]

df_nj = df_nj[df_nj["round"] == 2]
df_nj_group = df_nj.groupby(["subject_id", "plane"])
df_performance["nj_slope"] = df_nj_group.apply(
    lambda row: get_slope(row.stim_number, row.resp_number)).reset_index(name='nj_slope')["nj_slope"]
df_performance["nj_rmse"] = df_nj_group.apply(
    lambda row: get_rmse(row.stim_number, row.resp_number)).reset_index(name='nj_rmse')["nj_rmse"]

df_performance = df_performance.merge(df_su_group["normed_threshold"].mean().reset_index(name="su_mean_threshold"), on=["subject_id", "plane"])

df_nj = df_nj.merge(df_performance, on=["subject_id", "plane"])

df_la.to_csv("DataFrames/locaaccu_post_processed_performance.csv")
df_su.to_csv("DataFrames/spatmask_post_processed_performance.csv")
df_nj.to_csv("DataFrames/numjudge_post_processed_performance.csv")