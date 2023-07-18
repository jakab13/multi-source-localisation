from analysis.utils.misc import *
import os
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
sns.set_theme()

"""
For each subject and paradigm in a plane, calculate and plot the performance slope. 
For LocaAccu for instance, if the person performs badly horizontally, we would expect them to perform equally bad in the
other plane as well.
"""

fp = os.path.join(get_config("DATA_ROOT"), "MSL")  # filepath

# get data for all paradigms
locaaccu_v = load_dataframe(fp, exp_name="LocaAccu", plane="v")
filled_v = locaaccu_v["mode"].ffill()
noise_v = locaaccu_v[np.where(filled_v=="noise", True, False)]  # True where reversed_speech is True
babble_v = locaaccu_v[np.where(filled_v=="babble", True, False)]  # True where reversed_speech is False
spatmask_v = load_dataframe(fp, exp_name="SpatMask", plane="v")


# get data for all paradigms
locaaccu_h = load_dataframe(fp, exp_name="LocaAccu", plane="h")
# divide noise vs babble blocks
filled_h = locaaccu_h["mode"].ffill()
noise_h = locaaccu_h[np.where(filled_h=="noise", True, False)]  # True where reversed_speech is True
babble_h = locaaccu_h[np.where(filled_h=="babble", True, False)]  # True where reversed_speech is False

spatmask_h = load_dataframe(fp, exp_name="SpatMask", plane="h")

sub_ids = extract_subject_ids_from_dataframe(locaaccu_v)  # subject IDs

model = LinearRegression(fit_intercept=True)  # linear regression model

babble_data_v = dict(gain=[], r2=[], mse=[])  # dict to store data
babble_data_h = dict(gain=[], r2=[], mse=[])  # dict to store data
noise_data_v = dict(gain=[], r2=[], mse=[])  # dict to store data
noise_data_h = dict(gain=[], r2=[], mse=[])  # dict to store data

"""
LOCALIZATION ACCURACY
"""

for sub in sub_ids:  # iterate through all subjects
    ### vertical plane babble ###
    actual_v = get_elevation_from_df(babble_v.loc[sub].actual)
    perceived_v = get_elevation_from_df(babble_v.loc[sub].perceived)
    actual_v = replace_in_array(actual_v)  # get right shape
    perceived_v = replace_in_array(perceived_v)  # get right shape
    # fit locaaccu data to model for first subject
    model.fit(np.reshape(actual_v, (-1, 1)), perceived_v)  # fit model
    babble_data_v["r2"].append(model.score(np.reshape(actual_v, (-1, 1)), perceived_v))  # R²
    babble_data_v["gain"].append(model.coef_)  # slope
    babble_data_v["mse"].append(np.mean(np.abs(np.subtract(perceived_v, actual_v)) ** 2))  # mean squared error

    ### horizontal plane babble ###
    actual_h = get_azimuth_from_df(babble_h.loc[sub].actual)
    perceived_h = get_azimuth_from_df(babble_h.loc[sub].perceived)
    actual_h = replace_in_array(actual_h)  # get right shape
    perceived_h = replace_in_array(perceived_h)  # get right shape
    model.fit(np.reshape(actual_h, (-1, 1)), perceived_h)  # fit model
    babble_data_h["r2"].append(model.score(np.reshape(actual_h, (-1, 1)), perceived_h))  # R²
    babble_data_h["gain"].append(model.coef_)  # slope
    babble_data_h["mse"].append(np.mean(np.abs(np.subtract(perceived_h, actual_h)) ** 2))  # mean squared error

    ### vertical plane noise ###
    actual_v = get_elevation_from_df(noise_v.loc[sub].actual)
    perceived_v = get_elevation_from_df(noise_v.loc[sub].perceived)
    actual_v = replace_in_array(actual_v)  # get right shape
    perceived_v = replace_in_array(perceived_v)  # get right shape
    # fit locaaccu data to model for first subject
    model.fit(np.reshape(actual_v, (-1, 1)), perceived_v)  # fit model
    noise_data_v["r2"].append(model.score(np.reshape(actual_v, (-1, 1)), perceived_v))  # R²
    noise_data_v["gain"].append(model.coef_)  # slope
    noise_data_v["mse"].append(np.mean(np.abs(np.subtract(perceived_v, actual_v)) ** 2))  # mean squared error

    ### horizontal plane noise ###
    actual_h = get_azimuth_from_df(noise_h.loc[sub].actual)
    perceived_h = get_azimuth_from_df(noise_h.loc[sub].perceived)
    actual_h = replace_in_array(actual_h)  # get right shape
    perceived_h = replace_in_array(perceived_h)  # get right shape
    model.fit(np.reshape(actual_h, (-1, 1)), perceived_h)  # fit model
    noise_data_h["r2"].append(model.score(np.reshape(actual_h, (-1, 1)), perceived_h))  # R²
    noise_data_h["gain"].append(model.coef_)  # slope
    noise_data_h["mse"].append(np.mean(np.abs(np.subtract(perceived_h, actual_h)) ** 2))  # mean squared error

plt.scatter(babble_data_h["gain"], babble_data_v["gain"])
plt.scatter(noise_data_h["gain"], noise_data_v["gain"])
for i, txt in enumerate(sub_ids):
    plt.annotate(txt, (babble_data_h["gain"][i], babble_data_v["gain"][i]))
    plt.annotate(txt, (noise_data_h["gain"][i], noise_data_v["gain"][i]))
plt.xlabel("Horizontal")
plt.ylabel("Vertical")
plt.title("Localization Accuracy")
plt.legend(["Babble noise", "Pink noise"])

"""
NUMEROSITY JUDGEMENT
"""

numjudge_v = load_dataframe(fp, exp_name="NumJudge", plane="v")
filled_v = numjudge_v.reversed_speech.ffill()
clear_v = numjudge_v[np.where(filled_v==False, True, False)]  # True where reversed_speech is True
reversed_v = numjudge_v[np.where(filled_v==True, True, False)]  # True where reversed_speech is False

numjudge_h = load_dataframe(fp, exp_name="NumJudge", plane="h")
filled_h = numjudge_h.reversed_speech.ffill()
clear_h = numjudge_h[np.where(filled_h==False, True, False)]  # True where reversed_speech is True
reversed_h = numjudge_h[np.where(filled_h==True, True, False)]  # True where reversed_speech is False

clear_data_v = dict(gain=[], correct=[])  # dict to store data
clear_data_h = dict(gain=[], correct=[])  # dict to store data
reversed_data_v = dict(gain=[], correct=[])  # dict to store data
reversed_data_h = dict(gain=[], correct=[])  # dict to store data

for sub in sub_ids:
    # horizontal
    # fit clear numjudge data to model
    x = list(clear_h.loc[sub].solution.values)
    y = list(clear_h.loc[sub].response.fillna(0).values)
    clear_data_h["gain"].append(np.polyfit(np.log(x), y, 1)[1])  # slope
    correct_responses = [x for x in clear_h.loc[sub].is_correct if x == True]
    clear_data_h["correct"].append(correct_responses.__len__() / clear_h.loc[sub].is_correct.__len__())

    # fit reversed numjudge data to model
    x = list(reversed_h.loc[sub].solution.values)
    y = list(reversed_h.loc[sub].response.fillna(0).values)
    reversed_data_h["gain"].append(np.polyfit(np.log(x), y, 1)[1])  # slope
    correct_responses = [x for x in reversed_h.loc[sub].is_correct if x == True]
    reversed_data_h["correct"].append(correct_responses.__len__() / reversed_h.loc[sub].is_correct.__len__())

    # vertical
    # fit clear numjudge data to model
    x = list(clear_v.loc[sub].solution.values)
    y = list(clear_v.loc[sub].response.fillna(0).values)
    clear_data_v["gain"].append(np.polyfit(np.log(x), y, 1)[1])  # slope
    correct_responses = [x for x in clear_v.loc[sub].is_correct if x == True]
    clear_data_v["correct"].append(correct_responses.__len__() / clear_v.loc[sub].is_correct.__len__())

    # fit reversed numjudge data to model
    x = list(reversed_v.loc[sub].solution.values)
    y = list(reversed_v.loc[sub].response.fillna(0).values)
    reversed_data_v["gain"].append(np.polyfit(np.log(x), y, 1)[1])  # slope
    correct_responses = [x for x in reversed_v.loc[sub].is_correct if x == True]
    reversed_data_v["correct"].append(correct_responses.__len__() / reversed_v.loc[sub].is_correct.__len__())

# gain
plt.scatter(clear_data_h["gain"], clear_data_v["gain"])
plt.scatter(reversed_data_h["gain"], reversed_data_v["gain"])
plt.title("Numerosity Judgement performance gain")
plt.xlabel("Horizontal")
plt.ylabel("Vertical")
plt.legend(["Forward speech", "Reversed speech"])

# percent correct
plt.scatter(clear_data_h["correct"], clear_data_v["correct"])
plt.scatter(reversed_data_h["correct"], reversed_data_v["correct"])
plt.title("Numerosity Judgement percentage correct")
plt.xlabel("Horizontal")
plt.ylabel("Vertical")
plt.legend(["Forward speech", "Reversed speech"])
