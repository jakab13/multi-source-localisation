from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import ptitprince as pt
plt.style.use(['science', 'nature'])

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


layout = """
AAC
BBC
"""
axes = plt.figure(layout="constrained").subplot_mosaic(layout, sharex=False, sharey=True)
plt.ylim((-60, 60))
pt.RainCloud(data["noisev"][0], data["noisev"][1], ax=axes["A"], **raincloud_params)
pt.RainCloud(data["noiseh"][0], data["noiseh"][1], ax=axes["B"], **raincloud_params)
sns.scatterplot(x=labh.babbleh.abs(), y=labv.babblev.abs(), ax=axes["C"])
