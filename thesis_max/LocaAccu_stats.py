import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from analysis.utils.misc import *

# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# get subject ids from dataframe
sub_ids = extract_subject_ids_from_dataframe(dfh)

# divide noise vs babble blocks
filledh = dfh["mode"].ffill()
noiseh = dfh[np.where(filledh=="noise", True, False)]  # True where reversed_speech is True
babbleh = dfh[np.where(filledh=="babble", True, False)]  # True where reversed_speech is False

# divide noise vs babble blocks
filledv = dfv["mode"].ffill()
noisev = dfv[np.where(filledv=="noise", True, False)]  # True where reversed_speech is True
babblev= dfv[np.where(filledv=="babble", True, False)]  # True where reversed_speech is False

# MAD
madnoiseh = np.mean(np.abs(replace_in_array(get_azimuth_from_df(noiseh.accuracy))))
madbabbleh = np.mean(np.abs(replace_in_array(get_azimuth_from_df(babbleh.accuracy))))
madnoisev = np.mean(np.abs(replace_in_array(get_elevation_from_df(noisev.accuracy))))
madbabblev = np.mean(np.abs(replace_in_array(get_elevation_from_df(babblev.accuracy))))

print(f"AZIMUTH: \n"
      f"MAD babble noise: {madbabbleh.round(2)} \n"
      f"MAD rifle noise: {madnoiseh.round(2)} \n"
      "### \n"
      f"ELEVATION: \n"
      f"MAD babble noise: {madbabblev.round(2)} \n"
      f"MAD rifle noise: {madnoisev.round(2)}")

# stats
print(f"WILCOXON SIGNED RANK TEST MAD: \n"
      f"Babble noise elevation vs. azimuth: {stats.wilcoxon(replace_in_array(get_azimuth_from_df(babbleh.accuracy)), replace_in_array(get_elevation_from_df(babblev.accuracy)))} \n"
      f"Rifle noise elevation vs. azimuth: {stats.wilcoxon(replace_in_array(get_azimuth_from_df(noiseh.accuracy)), replace_in_array(get_elevation_from_df(noisev.accuracy)))} \n"
      f"Babble vs rifle elevation: {stats.wilcoxon(replace_in_array(get_elevation_from_df(babblev.accuracy)), replace_in_array(get_elevation_from_df(noisev.accuracy)))} \n"
      f"Babble vs rifle azimuth: {stats.wilcoxon(replace_in_array(get_azimuth_from_df(babbleh.accuracy)), replace_in_array(get_azimuth_from_df(noiseh.accuracy)))} \n")

pvals = [stats.wilcoxon(replace_in_array(get_azimuth_from_df(babbleh.accuracy)), replace_in_array(get_elevation_from_df(babblev.accuracy)))[1],
         stats.wilcoxon(replace_in_array(get_azimuth_from_df(noiseh.accuracy)), replace_in_array(get_elevation_from_df(noisev.accuracy)))[1],
         stats.wilcoxon(replace_in_array(get_elevation_from_df(babblev.accuracy)), replace_in_array(get_elevation_from_df(noisev.accuracy)))[1],
         stats.wilcoxon(replace_in_array(get_azimuth_from_df(babbleh.accuracy)), replace_in_array(get_azimuth_from_df(noiseh.accuracy)))[1]]

multitest_method = "bonferroni"
print(f"Bonferroni-corrected: {multipletests(pvals, method=multitest_method)}")

# regression performance elevation vs azimuth
fromfph = "/home/max/labplatform/data/linear_model/final_df_revspeech_h.csv"
dfh = pd.read_csv(fromfph, index_col=0)
fromfpv = "/home/max/labplatform/data/linear_model/final_df_revspeech_v.csv"
dfv = pd.read_csv(fromfpv, index_col=0)

x1 = np.unique(dfh.lababble).reshape((-1, 1))
y1 = np.unique(dfv.lababble)

x2 = np.unique(dfh.lanoise).reshape((-1, 1))
y2 = np.unique(dfv.lanoise)

# fit model
model1 = LinearRegression().fit(x1, y1)
model2 = LinearRegression().fit(x1, y2)

print(f"LINEAR REGRESSION \n"
      f"AZIMUTH VS ELEVATION BABBLE NOISE MAD: \n"
      f"Line slope: {model1.coef_} \n"
      f"Correlation: {model1.score(x1, y1)} \n"
      f"AZIMUTH VS ELEVATION RIFLE NOISE MAD \n"
      f"Line slope: {model2.coef_} \n"
      f"Correlation: {model2.score(x2, y2)}")

# reaction times
rtnoiseh = np.mean(noiseh.rt)
rtbabbleh = np.mean(babbleh.rt)
rtnoisev = np.mean(noisev.rt)
rtbabblev = np.mean(babblev.rt)

print(f"MEAN REACTION TIMES \n"
      f"AZIMUTH: \n"
      f"Babble noise: {rtbabbleh} ms\n"
      f"Rifle noise: {rtnoiseh} ms\n"
      f"ELEVATION: \n"
      f"Babble noise: {rtbabblev} ms\n"
      f"Rifle noise: {rtnoisev} ms\n")

print(f"WILCOXON SIGNED RANK TEST REACTION TIMES ELEVATION VS. AZIMUTH\n"
      f"{stats.wilcoxon(replace_in_array(np.append(noiseh.rt, babbleh.rt)), replace_in_array(np.append(noisev.rt, babblev.rt)))}")

