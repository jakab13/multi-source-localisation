from analysis.utils.misc import *
from labplatform.config import get_config
import pickle as pkl

"""
Extract all the parameters from the data and put them into single .csv files. After that, put all into a dataframe 
and save it.
"""

# LOCAACCU MEAN ABSOLUTE DISTANCE
# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "LocaAccu"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")
filled_h = dfh["mode"].ffill()
filled_v = dfv["mode"].ffill()

noise_h = dfh[np.where(filled_h=="noise", True, False)]  # True where reversed_speech is True
babble_h = dfh[np.where(filled_h=="babble", True, False)]  # True where reversed_speech is False
noise_v = dfv[np.where(filled_v=="noise", True, False)]  # True where reversed_speech is True
babble_v = dfv[np.where(filled_v=="babble", True, False)]  # True where reversed_speech is False

sub_ids = extract_subject_ids_from_dataframe(dfh)  # subject IDs

mad = dict(noiseh=dict(),
           babbleh=dict(),
           babblev=dict(),
           noisev=dict())

# subject wise
for sub in sub_ids:
    # horizontal noise
    actualh = replace_in_array(get_azimuth_from_df(noise_h.actual.loc[sub]))
    perceivedh = replace_in_array(get_azimuth_from_df(noise_h.perceived.loc[sub]))
    diffh = np.subtract(actualh, perceivedh)
    mad["noiseh"][sub] = np.mean(np.abs(diffh))

    # vertical noise
    actualv = replace_in_array(get_elevation_from_df(noise_v.actual.loc[sub]))
    perceivedv = replace_in_array(get_elevation_from_df(noise_v.perceived.loc[sub]))
    diffv = np.subtract(actualv, perceivedv)
    mad["noisev"][sub] = np.mean(np.abs(diffv))

    # horizontal babble
    actualh = replace_in_array(get_azimuth_from_df(babble_h.actual.loc[sub]))
    perceivedh = replace_in_array(get_azimuth_from_df(babble_h.perceived.loc[sub]))
    diffh = np.subtract(actualh, perceivedh)
    mad["babbleh"][sub] = np.mean(np.abs(diffh))

    # vertical babble
    actualv = replace_in_array(get_elevation_from_df(babble_v.actual.loc[sub]))
    perceivedv = replace_in_array(get_elevation_from_df(babble_v.perceived.loc[sub]))
    diffv = np.subtract(actualv, perceivedv)
    mad["babblev"][sub] = np.mean(np.abs(diffv))

# save into .csv
csvpath = "/home/max/labplatform/data/csv"
df = pd.DataFrame(mad)
df.noiseh.to_csv(os.path.join(csvpath, "mad_noise_h.csv"), index=False)
df.noisev.to_csv(os.path.join(csvpath, "mad_noise_v.csv"), index=False)
df.babbleh.to_csv(os.path.join(csvpath, "mad_babble_h.csv"), index=False)
df.babblev.to_csv(os.path.join(csvpath, "mad_babble_v.csv"), index=False)


# NUMJUDGE
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

# horizontal
filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
revspeech_h = revspeech_h.sort_index()
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False
clearspeech_h = clearspeech_h.sort_index()

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
revspeech_v = revspeech_v.sort_index()
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False
clearspeech_v = clearspeech_v.sort_index()

# save responses
clearspeech_h.response.fillna(0).to_csv(os.path.join(csvpath, "numjudge_response_clearspeech_h.csv"))
clearspeech_v.response.fillna(0).to_csv(os.path.join(csvpath, "numjudge_response_clearspeech_v.csv"))
revspeech_h.response.fillna(0).to_csv(os.path.join(csvpath, "numjudge_response_revspeech_h.csv"))
revspeech_v.response.fillna(0).to_csv(os.path.join(csvpath, "numjudge_response_revspeech_v.csv"))


if __name__ == "__main__":
    # test load
    loaded = pd.read_csv(os.path.join(csvpath, "mad_noise_h.csv"))
