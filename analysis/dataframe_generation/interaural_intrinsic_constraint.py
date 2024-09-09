import slab
import time
import os
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import scipy
import datetime
from mplcursors import cursor
import copy

ils = pickle.load(open('ils.pickle', 'rb'))

output_path = "interaural_intrinsic_constraint.csv"
subject_id = "jakab"
standard_cue = "itd"
comparison_cue = "itd"
stim_type = "broadband"  # "broadband" or "hp" or "lp"
filter_frequency = 2000
standard_angle = 2
isi = 0.3
save = True
n_reps = 10

df = pd.DataFrame(columns=[
    "subject_id",
    "datetime_onset",
    "stim_type",
    "standard_cue",
    "comparison_cue",
    "trial_type",
    "standard_angle",
    "comparison_angle",
    "standard_order",
    "comparison_order",
    "isi",
    "response",
    "reaction_time",
    "is_correct",
    "score"
])

trials = slab.Trialsequence(conditions=[-45, -35, -25, -15, -5, 0, 5, 15, 25, 35, 45], n_reps=n_reps)
# trials = slab.Trialsequence(conditions=[-45, -30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30, 45], n_reps=n_reps)


def apply_cue(stim, cue, angle, filter_frequency, head_radius=7.5):
    sound = copy.deepcopy(stim)
    if cue == "itd" or cue == "both":
        itd_val = slab.Binaural.azimuth_to_itd(angle, frequency=filter_frequency, head_radius=head_radius)
        sound = sound.itd(itd_val).externalize()
    elif cue == "ild" or cue == "both":
        ild_val = slab.Binaural.azimuth_to_ild(angle, frequency=filter_frequency, ils=ils)
        sound = sound.ild(ild_val).externalize()
    return sound


for comparison_angle in trials:
    noise = slab.Binaural.pinknoise(duration=0.3, samplerate=44100).ramp(duration=0.05)
    if stim_type != "broadband":
        filt = slab.Filter.band(frequency=filter_frequency, kind=stim_type, samplerate=44100)
        noise = filt.apply(noise)
    standard = apply_cue(noise, standard_cue, standard_angle, filter_frequency)
    comparison = apply_cue(noise, comparison_cue, comparison_angle, filter_frequency)
    presentations = [standard, comparison]
    order = np.random.permutation(len(presentations))
    expected_standard = np.where(order == 0)[0][0]
    is_comparison_to_the_right = comparison_angle > standard_angle
    solution = None
    if expected_standard == 0 and is_comparison_to_the_right == 0:
        solution = 1
    elif expected_standard == 0 and is_comparison_to_the_right == 1:
        solution = 2
    elif expected_standard == 1 and is_comparison_to_the_right == 0:
        solution = 2
    elif expected_standard == 1 and is_comparison_to_the_right == 1:
        solution = 1
    datetime_onset = datetime.datetime.now()
    for i, idx in enumerate(order):
        stim = presentations[idx]
        stim.play()
        if i == 0:
            time.sleep(isi)
    datetime_offset = datetime.datetime.now()
    while True:
        response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trials.this_n + 1}/{trials.n_trials})")
        if response not in ["1", "2"]:
            print(f"You entered: '{response}' response. You have to select <----- 1 | 2 ----->")
            continue
        else:
            break
    response = int(response)
    reaction_time = datetime.datetime.now() - datetime_offset
    is_correct = response == solution
    score = int(is_correct) if is_comparison_to_the_right else 1 - int(is_correct)
    if save:
        row = {
            "subject_id": subject_id,
            "datetime_onset": datetime_onset,
            "stim_type": stim_type,
            "standard_cue": standard_cue,
            "comparison_cue": comparison_cue,
            "trial_type": standard_cue + "-->" + comparison_cue,
            "standard_angle": standard_angle,
            "comparison_angle": comparison_angle,
            "standard_order": np.where(order == 0)[0][0],
            "comparison_order": np.where(order == 1)[0][0],
            "isi": isi,
            "response": response,
            "reaction_time": reaction_time,
            "is_correct": is_correct,
            "score": score
        }
        df_curr = pd.DataFrame.from_dict([row])
        df_curr.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


# =============== PLOTTING ==================

df_final = pd.read_csv(output_path)
df_final = df_final[~((df_final.subject_id == "jakab2") & (df_final.standard_angle == 2.5))]
df_final = df_final.replace("jakab2", "jakab")


def get_sigma(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])[0]
    return sigma


def get_mu(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])[0]
    return mean


df_group = df_final.groupby(["subject_id", "standard_angle", "stim_type", "trial_type"])
df_model = df_group.apply(lambda g: get_sigma(g)).reset_index(name='sigma')
df_model["mu"] = df_group.apply(lambda g: get_mu(g)).reset_index(name='mu')["mu"]

df_final = df_final.merge(df_model[["subject_id", "standard_angle", "stim_type", "trial_type", "sigma", "mu"]],
                          on=["subject_id", "standard_angle", "stim_type", "trial_type"])


df_cue_ratio = df_final.groupby(["subject_id", "stim_type",
                                 "standard_angle",
                                 "trial_type"], as_index=False)["sigma", "mu"].mean()

for subject_id in df_final.subject_id.unique():
    for stim_type in df_final.stim_type.unique():
        for trial_type in df_final.trial_type.unique():
            for standard_angle in df_final.standard_angle.unique():
                sigma_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_type == stim_type) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                        (df_cue_ratio.trial_type == "itd-->itd")]["sigma"]
                sigma_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_type == stim_type) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                         (df_cue_ratio.trial_type == "ild-->ild")]["sigma"]
                mu_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_type == stim_type) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                         (df_cue_ratio.trial_type == "itd-->itd")]["mu"]
                mu_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                         (df_cue_ratio.stim_type == stim_type) &
                                         (df_cue_ratio.standard_angle == standard_angle) &
                                         (df_cue_ratio.trial_type == "ild-->ild")]["mu"]
                if len(sigma_itd):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type) &
                                 (df_final.stim_type == stim_type), "sigma_itd"] = sigma_itd.values[0]
                if len(sigma_ild):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type) &
                                 (df_final.stim_type == stim_type), "sigma_ild"] = sigma_ild.values[0]
                if len(mu_itd):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type) &
                                 (df_final.stim_type == stim_type), "mu_itd"] = mu_itd.values[0]
                if len(mu_ild):
                    df_final.loc[(df_final.subject_id == subject_id) &
                                 (df_final.standard_angle == standard_angle) &
                                 (df_final.trial_type == trial_type) &
                                 (df_final.stim_type == stim_type), "mu_ild"] = mu_ild.values[0]

df_final = df_final.fillna(0)

def get_PSE_mid(row):
    x_0 = row["standard_angle"]
    sigma_i = "sigma_" + str(row["standard_cue"])
    sigma_j = "sigma_" + str(row["comparison_cue"])
    mu_i = "mu_" + str(row["standard_cue"])
    mu_j = "mu_" + str(row["comparison_cue"])
    sigma_i = row[sigma_i]
    sigma_j = row[sigma_j]
    mu_i = row[mu_i]
    mu_j = row[mu_j]
    sin_theta_i = 1 / np.sqrt(1 + sigma_i ** 2)
    sin_theta_j = 1 / np.sqrt(1 + sigma_j ** 2)
    # Q = np.sqrt(1 + sigma_j ** 2) / np.sqrt(1 + sigma_i ** 2)
    PSE = x_0 - mu_j + mu_i * (sin_theta_j / sin_theta_i)
    return PSE

def get_PSE_high(row):
    x_0 = row["standard_angle"]
    sigma_i = "sigma_" + str(row["standard_cue"])
    sigma_j = "sigma_" + str(row["comparison_cue"])
    mu_i = "mu_" + str(row["standard_cue"])
    mu_j = "mu_" + str(row["comparison_cue"])
    sigma_i = row[sigma_i]
    sigma_j = row[sigma_j]
    mu_i = row[mu_i]
    mu_j = row[mu_j]
    sin_theta_i = 1 / np.sqrt(1 + sigma_i ** 2)
    sin_theta_j = 1 / np.sqrt(1 + sigma_j ** 2)
    PSE = (mu_i + sigma_i) * sin_theta_i + (x_0 - mu_j) * sin_theta_j
    return PSE

def get_PSE_low(row):
    x_0 = row["standard_angle"]
    x = row["comparison_angle"]
    sigma_i = "sigma_" + str(row["standard_cue"])
    sigma_j = "sigma_" + str(row["comparison_cue"])
    mu_i = "mu_" + str(row["standard_cue"])
    mu_j = "mu_" + str(row["comparison_cue"])
    sigma_i = row[sigma_i]
    sigma_j = row[sigma_j]
    mu_i = row[mu_i]
    mu_j = row[mu_j]
    Q = np.sqrt(1 + sigma_j ** 2) / np.sqrt(1 + sigma_i ** 2)
    PSE = Q * (x - x_0 + mu_i) + x_0 - mu_j
    return PSE


# df_final["PSE_base"] = df_final.apply(lambda row: get_PSE(row), axis=1)

df_final["PSE_mid"] = df_final.apply(lambda row: get_PSE_mid(row), axis=1)
df_final["PSE_mid_y"] = 0.5
df_final["PSE_high"] = df_final.apply(lambda row: get_PSE_high(row), axis=1)
df_final["PSE_high_y"] = 0.85
df_final["PSE_low"] = df_final.apply(lambda row: get_PSE_low(row), axis=1)


df_plot = df_final[(df_final.standard_angle < 10) &
                   (df_final.stim_type == "broadband")
                   & (df_final.subject_id == "jakab")
                    & (df_final.standard_angle != 2.5)
                    ]
# df_plot = df_final[(df_final.standard_angle < 10) & (df_final.stim_type == "broadband") & (df_final.trial_type == "ild-->ild")]
# df_plot = df_final[(df_final.standard_angle < 10) & (df_final.subject_id == "jakab")]
g = sns.FacetGrid(df_plot, hue="trial_type", col="subject_id", row="standard_angle", height=1.7, aspect=2.5, sharex=True, sharey=True)
g.map(sns.regplot, "comparison_angle", "score", logistic=True, scatter=False, ci=None)
# g.map(sns.scatterplot, "PSE_mid", "PSE_mid_y", s=100)
# g.map(sns.scatterplot, "PSE_high", "PSE_high_y", s=100)
# g.map(sns.scatterplot, "PSE_low", "PSE_low_y", s=100)
g.set(xlim=(-65, 65))
g.add_legend()
[ax.axhline(0.5, ls='--', c="red", alpha=.2) for ax in g.axes.flatten()]
[ax.axvline(float(ax.title.get_text().split("standard_angle = ")[1].split("|")[0]), ls='--', c="red", alpha=.2) for ax in g.axes.flatten()]
cursor(hover=True)

print(df_final.groupby(["subject_id", "trial_type", "standard_angle", "stim_type"]).count())

#
# df.groupby(["isi", "trial_type"]).count()
#
# stairs = slab.Staircase(start_val=5, n_reversals=18, step_sizes=[5, 5, 3, 2, 2, 1])
#
# for comparison_angle in stairs:
#     noise = slab.Binaural.pinknoise(samplerate=44100).ramp().externalize()
#     standard = noise.itd(slab.Binaural.azimuth_to_itd(10, head_radius=7.5))
#     comparison = noise.ild(slab.Binaural.azimuth_to_ild(comparison_angle, ils=ils))
#     presentations = [standard, comparison]  # assuming two sound objects
#     order = np.random.permutation(len(stims))
#     for idx in order:
#         stim = presentations[idx]
#         stim.play()
#         time.sleep(0.1)
#     response = input("Which sound was on the right?")
#     response = int(response)
#     interval = np.where(order == 0)[0][0]
#     response = response == (interval + 1)
#     print(not response)
#     stairs.add_response(int(not response))  # initiates calculation of next stimulus value
#     stairs.plot()

