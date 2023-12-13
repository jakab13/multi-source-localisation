from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
sns.set_theme(style="white", palette="viridis")
plt.rcParams['text.usetex'] = True  # TeX rendering


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
axes = plt.figure().subplot_mosaic(layout, sharex=True, sharey=True)
axes["A"].set_title("Elevation Performance")
axes["B"].set_title("Azimuth Performance")
axes["A"].set_xlabel("Actual Position (degrees)")
axes["A"].set_ylabel("Judged Position (degrees)")
axes["B"].set_xlabel("Actual Position (degrees)")


sns.lineplot(data["noisev"][0], data["noisev"][1], ax=axes["A"], label="Noise")
sns.lineplot(data["babblev"][0], data["babblev"][1], ax=axes["A"], label="Babble")


sns.lineplot(data["noiseh"][0], data["noiseh"][1], ax=axes["B"], label="Noise")
sns.lineplot(data["babbleh"][0], data["babbleh"][1], ax=axes["B"], label="Babble")

axes["A"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
axes["B"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.show()

sns.distplot(data["rth"], label="Horizontal")
sns.distplot(data["rtv"], label="Vertical")
plt.xlim((0, 5000))
plt.xlabel("Reaction Time [ms]")
plt.legend()


"""
pt.RainCloud(data["noisev"][0], data["noisev"][1], ax=axes["A"], **raincloud_params)
pt.RainCloud(data["noiseh"][0], data["noiseh"][1], ax=axes["B"], **raincloud_params)
sns.scatterplot(x=labh.babbleh.abs(), y=labv.babblev.abs(), ax=axes["C"])
"""
