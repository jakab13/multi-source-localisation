from analysis.utils.plotting import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
sns.set_theme(style="white", palette="inferno")
plt.rcParams['text.usetex'] = True  # TeX rendering


boxplot_params = dict(saturation=0.75,
                      width=.8,
                      dodge=True,
                      fliersize=5,
                      linewidth=None)


# get dataframes
root = "/home/max/labplatform/data/DataFrames/"
su_fph = os.path.join(root, "spatmask_abs_distance_h.pkl")
su_fpv = os.path.join(root, "spatmask_abs_distance_v.pkl")

suh = pkl.load(open(su_fph, "rb"))

suv = pkl.load(open(su_fpv, "rb"))


all = pd.concat([suh, suv], keys=["Azimuth", "Elevation"], axis=0)
# suh["dimension"] = ["azimuth"] * len(suh.index)

sns.boxplot(data=all.loc["Azimuth"], palette="twilight", **boxplot_params)
sns.boxplot(data=all.loc["Elevation"], palette="twilight_shifted", **boxplot_params)
sns.despine(offset=10, trim=True)

plt.xlabel("Speaker-Target Difference [degrees]")
plt.ylabel("50 Percent Hearing Threshold (TMR) [dB]")
plt.legend(labels=["Azimuth", "Elevation"], loc="upper left")
plt.show()
