from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
import string
import scienceplots
plt.style.use("science")
plt.ion()


raincloud_params = dict(bw=0.2,
                        width_viol=.6,
                        pointplot=True,
                        alpha=1.0,
                        dodge=True)
# get dataframes
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# mad
root = "/home/max/labplatform/data/csv"  # root

# data paths
labfpv = os.path.join(root, "mad_babble_v.csv")  # locaaccu babble
lanfpv = os.path.join(root, "mad_noise_v.csv")  # locaaccu noise
labfph = os.path.join(root, "mad_babble_h.csv")  # locaaccu babble
lanfph = os.path.join(root, "mad_noise_h.csv")  # locaaccu noise

labh = pd.read_csv(labfph)
lanh = pd.read_csv(lanfph)
labv = pd.read_csv(labfpv)
lanv = pd.read_csv(lanfpv)

# get subject ids from dataframe
sub_ids = extract_subject_ids_from_dataframe(dfh)

data = dict()

# vertical
filledv = dfv["mode"].ffill()
noisev = dfv[np.where(filledv=="noise", True, False)]  # True where reversed_speech is True
babblev = dfv[np.where(filledv=="babble", True, False)]  # True where reversed_speech is

# horizontal
filledh = dfh["mode"].ffill()
noiseh = dfh[np.where(filledh=="noise", True, False)]  # True where reversed_speech is True
babbleh = dfh[np.where(filledh=="babble", True, False)]  # True where reversed_speech is

data["noisev"] = [replace_in_array(get_elevation_from_df(noisev.actual)), replace_in_array(get_elevation_from_df(noisev.perceived))]
data["babblev"] = [replace_in_array(get_elevation_from_df(babblev.actual)), replace_in_array(get_elevation_from_df(babblev.perceived))]
data["noiseh"] = [replace_in_array(get_azimuth_from_df(noiseh.actual)), replace_in_array(get_azimuth_from_df(noiseh.perceived))]
data["babbleh"] = [replace_in_array(get_azimuth_from_df(babbleh.actual)), replace_in_array(get_azimuth_from_df(babbleh.perceived))]
data["rth"] = babbleh.rt.append(noiseh.rt)
data["rtv"] = babblev.rt.append(noisev.rt)


layout = """
AB
"""
_, axes = plt.subplot_mosaic(layout, sharex=True, sharey=True, figsize=(6, 3))
axes["A"].set_xlabel("Actual Position (degrees)")
axes["A"].set_ylabel("Judged Position (degrees)")
axes["B"].set_xlabel("Actual Position (degrees)")
axes["A"].text(-0.1, 1.0, string.ascii_uppercase[0], transform=axes["A"].transAxes,
               size=20, weight='bold')
axes["B"].text(-0.1, 1.0, string.ascii_uppercase[1], transform=axes["B"].transAxes,
               size=20, weight='bold')

sns.lineplot(data["noisev"][0], data["noisev"][1], ax=axes["B"], label="Pink Noise")
sns.lineplot(data["babblev"][0], data["babblev"][1], ax=axes["B"], label="Babble Noise")


sns.lineplot(data["noiseh"][0], data["noiseh"][1], ax=axes["A"], label="Pink Noise")
sns.lineplot(data["babbleh"][0], data["babbleh"][1], ax=axes["A"], label="Babble Noise")

axes["A"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
axes["B"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")

plt.savefig("/home/max/labplatform/plots/MA_thesis/results/localization_performance.png",
            dpi=400, bbox_inches="tight")

sns.set_palette("ocean")


sns.regplot(labh, lanh, label="Azimuth")
sns.regplot(labv, lanv, label="Elevation")
plt.legend()
plt.xlabel("Babble Noise Mean Absolute Error (degrees)")
plt.ylabel("Pink Noise Mean Absolute Error (degrees)")
plt.gca().figure.set_figheight(4)
plt.gca().figure.set_figwidth(5)
plt.savefig("/home/max/labplatform/plots/MA_thesis/results/mae_plane_comparison.png",
            dpi=800)



sns.distplot(data["rth"], label="Azimuth")
sns.distplot(data["rtv"], label="Elevation")
plt.xlim((0, 5000))
plt.xlabel("Reaction Time [ms]")
plt.legend()
plt.gca().figure.set_figheight(4)
plt.gca().figure.set_figwidth(5)
plt.savefig("/home/max/labplatform/plots/MA_thesis/results/reaction_times.png",
            dpi=800)


"""
pt.RainCloud(data["noisev"][0], data["noisev"][1], ax=axes["A"], **raincloud_params)
pt.RainCloud(data["noiseh"][0], data["noiseh"][1], ax=axes["B"], **raincloud_params)
sns.scatterplot(x=labh.babbleh.abs(), y=labv.babblev.abs(), ax=axes["C"])
"""
