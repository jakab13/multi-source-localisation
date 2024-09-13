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
import matplotlib.pyplot as plt
import random

# ==============================================================

output_path = "interaural_intrinsic_constraint_4.csv"
subject_id = "jakab"
standard_cue = "ild"  # itd or ild or both
comparison_cue = "both"  # itd or ild or both
stim_type = "lp"  # "broadband" or "hp" or "lp" or "lp_200"
centre_frequency = 2000
standard_angle = 3  # [0, 1, 2, 3, 4]
save = True

# ==============================================================

ils = pickle.load(open('ils.pickle', 'rb'))
isi = 0.2
n_reps = 10
stim_name = stim_type + "_" + str(centre_frequency)

df = pd.DataFrame(columns=[
    "subject_id",
    "datetime_onset",
    "stim_type",
    "filter_frequency",
    "stim_name",
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

trials = slab.Trialsequence(conditions=[-45, -35, -25, -15, -5, 5, 15, 25, 35, 45], n_reps=n_reps)
# trials = slab.Staircase(start_val=45, step_sizes=[5, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2], n_down=2, n_up=2, n_reversals=12)
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


trial_counter = 0
n_trials = trials.n_trials if hasattr(trials, 'n_trials') else None

for comparison_angle in trials:
    noise = slab.Binaural.pinknoise(duration=0.3, samplerate=44100).ramp(duration=0.05)
    if stim_type != "broadband":
        filt = slab.Filter.band(frequency=centre_frequency, kind=stim_type, samplerate=44100)
        noise = filt.apply(noise)
        noise = noise.ramp(when="offset", duration=0.09)
    # direction = 1 if random.random() < 0.5 else -1
    # standard_angle *= direction
    standard = apply_cue(noise, standard_cue, standard_angle, centre_frequency)
    comparison = apply_cue(noise, comparison_cue, comparison_angle, centre_frequency)
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
        response = input(f"Which way did the sound move? <----- 1 | 2 -----> ({trial_counter + 1}/{n_trials})")
        if response not in ["1", "2", "49", "51"]:
            print(f"You entered: '{response}' response. You have to select <----- 1 | 2 ----->")
            continue
        else:
            break
    if response == "49" or response == "1":
        response = 1
    elif response == "51" or response == "2":
        response = 2
    reaction_time = datetime.datetime.now() - datetime_offset
    is_correct = response == solution
    score = int(is_correct) if is_comparison_to_the_right else 1 - int(is_correct)
    trials.add_response(score)
    # if hasattr(trials, "plot"):
    #     trials.plot()
    trial_counter += 1
    if save:
        row = {
            "subject_id": subject_id,
            "datetime_onset": datetime_onset,
            "stim_type": stim_type,
            "filter_frequency": centre_frequency,
            "stim_name": stim_name,
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

df_final = pd.read_csv("interaural_intrinsic_constraint_4.csv")
df_final = df_final[~(df_final.trial_type == "both-->both")]
# df_final = df_final[~(df_final.subject_id == "jakab2")]
# df_final = df_final.replace("jakab2", "jakab")


def get_sigma(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])[0]
    return sigma


def get_mu(row):
    cdf = scipy.stats.norm.cdf
    mean, sigma = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])[0]
    return mean

def get_mu_err(row):
    cdf = scipy.stats.norm.cdf
    [popt, pcov] = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])
    perr = np.sqrt(np.diag(pcov))
    return perr[0]

def get_sigma_err(row):
    cdf = scipy.stats.norm.cdf
    [popt, pcov] = scipy.optimize.curve_fit(cdf, row.comparison_angle, row.score, p0=[0, 100])
    perr = np.sqrt(np.diag(pcov))
    return perr[1]

df_group = df_final.groupby(["subject_id", "standard_angle", "stim_name", "trial_type"])
df_model = df_group.apply(lambda g: get_mu(g)).reset_index(name='mu')
df_model["sigma"] = df_group.apply(lambda g: get_sigma(g)).reset_index(name='sigma')["sigma"]
df_model["mu_sd"] = df_group.apply(lambda g: get_mu_err(g)).reset_index(name='mu_sd')["mu_sd"]
df_model["sigma_sd"] = df_group.apply(lambda g: get_sigma_err(g)).reset_index(name='sigma_sd')["sigma_sd"]


df_final = df_final.merge(df_model[["subject_id", "standard_angle", "stim_name", "trial_type", "mu", "sigma", "mu_sd", "sigma_sd"]],
                          on=["subject_id", "standard_angle", "stim_name", "trial_type"])

def get_cue_combination(row):
    if (row["standard_cue"] == "both") or (row["comparison_cue"] == "both"):
        cue_combination = "combined"
    elif row["standard_cue"] == row["comparison_cue"]:
        cue_combination = "within"
    else:
        cue_combination = "across"
    return cue_combination

df_final["cue_combination"] = df_final.apply(lambda row: get_cue_combination(row), axis=1)


df_cue_ratio = df_final.groupby(["subject_id",
                                 "stim_name",
                                 "trial_type"], as_index=False)["mu", "sigma"].mean()

df_final["sigma_itd"] = None
df_final["sigma_ild"] = None

for subject_id in df_final.subject_id.unique():
    for stim_name in df_final.stim_name.unique():
        for trial_type in df_final.trial_type.unique():
            sigma_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                     (df_cue_ratio.stim_name == stim_name) &
                                     (df_cue_ratio.trial_type == "itd-->itd")]["sigma"]
            sigma_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                     (df_cue_ratio.stim_name == stim_name) &
                                     (df_cue_ratio.trial_type == "ild-->ild")]["sigma"]
            mu_itd = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                  (df_cue_ratio.stim_name == stim_name) &
                                  (df_cue_ratio.trial_type == "itd-->itd")]["mu"]
            mu_ild = df_cue_ratio[(df_cue_ratio.subject_id == subject_id) &
                                  (df_cue_ratio.stim_name == stim_name) &
                                  (df_cue_ratio.trial_type == "ild-->ild")]["mu"]
            if len(sigma_itd):
                df_final.loc[(df_final.subject_id == subject_id) &
                             (df_final.stim_name == stim_name) &
                             (df_final.trial_type == trial_type), "sigma_itd"] = sigma_itd.values[0]
            if len(sigma_ild):
                df_final.loc[(df_final.subject_id == subject_id) &
                             (df_final.stim_name == stim_name) &
                             (df_final.trial_type == trial_type), "sigma_ild"] = sigma_ild.values[0]
            if len(mu_itd):
                df_final.loc[(df_final.subject_id == subject_id) &
                             (df_final.stim_name == stim_name) &
                             (df_final.trial_type == trial_type), "mu_itd"] = mu_itd.values[0]
            if len(mu_ild):
                df_final.loc[(df_final.subject_id == subject_id) &
                             (df_final.stim_name == stim_name) &
                             (df_final.trial_type == trial_type), "mu_ild"] = mu_ild.values[0]

# df_final = df_final.fillna(0)

#
# def get_PSE_mid(row):
#     x_0 = row["standard_angle"]
#     sigma_i = "sigma_" + str(row["standard_cue"])
#     sigma_j = "sigma_" + str(row["comparison_cue"])
#     mu_i = "mu_" + str(row["standard_cue"])
#     mu_j = "mu_" + str(row["comparison_cue"])
#     sigma_i = row[sigma_i]
#     sigma_j = row[sigma_j]
#     mu_i = row[mu_i]
#     mu_j = row[mu_j]
#     sin_theta_i = 1 / np.sqrt(1 + sigma_i ** 2)
#     sin_theta_j = 1 / np.sqrt(1 + sigma_j ** 2)
#     # Q = np.sqrt(1 + sigma_j ** 2) / np.sqrt(1 + sigma_i ** 2)
#     # PSE = x_0 - mu_j + mu_i * (sin_theta_j / sin_theta_i)
#     # PSE = x_0 * sin_theta_i / sin_theta_j
#     PSE = None
#     if row["standard_cue"] == row["comparison_cue"]:
#         PSE = mu_i
#     if sigma_i == 0 or sigma_j == 0:
#         PSE = None
#     else:
#         # if sigma_j > sigma_i:
#         PSE = x_0 * sigma_j / sigma_i + mu_i
#         # elif sigma_j < sigma_i:
#         #     PSE = x_0 * sigma_i / sigma_j + mu_j
#     return PSE
#
#
# # df_final["PSE_base"] = df_final.apply(lambda row: get_PSE(row), axis=1)
#
# df_final["PSE_mid"] = df_final.apply(lambda row: get_PSE_mid(row), axis=1)
# df_final["PSE_mid_y"] = 0.5
# df_final["PSE_high"] = df_final.apply(lambda row: get_PSE_high(row), axis=1)
# df_final["PSE_high_y"] = 0.85
# df_final["PSE_low"] = df_final.apply(lambda row: get_PSE_low(row), axis=1)

# df_final["delta_x"] =
df_final["delta_mu"] = df_final["mu"] - df_final["standard_angle"]
df_final["delta_mu_min"] = df_final["delta_mu"] - df_final["mu_sd"]
df_final["delta_mu_max"] = df_final["delta_mu"] + df_final["mu_sd"]


def get_pred_bayes(row):
    if row["cue_combination"] != "combined":
        sigma_i = row["sigma_" + str(row["standard_cue"])]
        sigma_j = row["sigma_" + str(row["comparison_cue"])]
        w_i = sigma_j ** 2 / (sigma_j ** 2 + sigma_i ** 2)
        w_j = sigma_i ** 2 / (sigma_j ** 2 + sigma_i ** 2)
        standard_pred = w_i * row["standard_angle"] + w_j * row["standard_angle"]
        comparison_pred = w_i * row["comparison_angle"] + w_j * row["comparison_angle"]
        delta_pred = standard_pred - comparison_pred
        return delta_pred


def get_pred_IC(row):
    if row["cue_combination"] != "combined":
        sigma_i = row["sigma_" + str(row["standard_cue"])]
        sigma_j = row["sigma_" + str(row["comparison_cue"])]
        sin_i = (1 / np.sqrt(1 + (sigma_i ** 2)))
        sin_j = (1 / np.sqrt(1 + (sigma_j ** 2)))
        direction = 1 if (sigma_i < sigma_j and row["cue_combination"] == "across") else -1
        delta_pred = sin_i / sin_j * (row["standard_angle"] - row["comparison_angle"]) * direction
        return delta_pred


df_final["pred_bayes"] = df_final.apply(lambda row: get_pred_bayes(row), axis=1)
df_final["pred_IC"] = df_final.apply(lambda row: get_pred_IC(row), axis=1)

df_final = df_final.merge(df_final.groupby(["subject_id", "stim_name", "standard_angle", "trial_type"],
                                           as_index=False)["pred_bayes", "pred_IC"].mean(),
                          on=["subject_id", "stim_name", "standard_angle", "trial_type"],
                          suffixes=("", "_mean"))

df_final["pred_IC_mean_mult"] = df_final["pred_IC_mean"] * 1.7

df_plot = df_final[(df_final["stim_name"] == "lp_2000")]


g = sns.FacetGrid(df_plot, hue="trial_type", col="cue_combination", row="standard_angle",
                  col_order=["within", "across"],
                  hue_order=["itd-->itd", "ild-->ild", "ild-->itd", "itd-->ild"],
                  height=1.5, aspect=2, sharex=True, sharey=True)
g.refline(y=0.5, c="grey", alpha=.3)
[ax.axvline(float(ax.title.get_text().split("standard_angle = ")[1].split("|")[0]), ls='--', c="black", alpha=.3) for ax in g.axes.flatten()]
g.set(xlim=(-50, 50))
g.map(sns.regplot, "comparison_angle", "score", logistic=True, scatter=False, ci=None)
def plot_PSE(x, **kwargs):
    c = kwargs.get("color", "g")
    x_mean = x.mean()
    plt.axvline(x_mean, ymax=0.5, c=c, alpha=.5)
    plt.plot(x_mean, 0.5, "o", c=c, markersize=5)
g.map(plot_PSE, "mu")
g.add_legend()
g.set_titles(template="standard stim at: {row_name}°")
g.figure.suptitle("PSE of 'comparison stimuli' in different cue combinations", size=16)
g.figure.subplots_adjust(top=.9)
g.set_xlabels(label="comparison stim azimuth")
# plt.savefig("Interaural Intrinsic Constraint pilot PSEs (1-16 deg).png", dpi=400)
# cursor(hover=True)

# df_itd_vs_ild = df_plot.groupby(["subject_id", "trial_type", "standard_angle"], as_index=False)["mu", "sigma"].mean().pivot(
#     index=["subject_id", "standard_angle"], columns=["trial_type"])
# df_itd_vs_ild.columns = ['_'.join(tup).rstrip('_') for tup in df_itd_vs_ild.columns.values]
# df_itd_vs_ild = df_itd_vs_ild.reset_index()
#
# df_itd_vs_ild = df_itd_vs_ild.merge(df_itd_vs_ild.groupby(["subject_id"],
#                                           as_index=False)["sigma_itd-->ild", "sigma_ild-->itd"].mean(),
#                     on=["subject_id"], suffixes=("", "_mean"))
#
# df_itd_vs_ild["mu_ratio"] = abs(df_itd_vs_ild["mu_itd-->ild"]) / abs(df_itd_vs_ild["mu_ild-->itd"])
# df_itd_vs_ild["mu_pred"] = df_itd_vs_ild["mu_ratio"] * df_itd_vs_ild["standard_angle"]
# df_itd_vs_ild["sigma_ratio"] = abs(df_itd_vs_ild["sigma_itd-->ild_mean"]) / abs(df_itd_vs_ild["sigma_ild-->itd_mean"])
# df_itd_vs_ild["sigma_pred"] = df_itd_vs_ild["sigma_ratio"] * df_itd_vs_ild["standard_angle"]

g_2 = sns.FacetGrid(df_plot,
                    hue="trial_type",
                    hue_order=["itd-->itd", "ild-->ild", "ild-->itd", "itd-->ild"],
                    col="cue_combination",
                    col_order=["within", "across"],
                    # row="stim_name",
                    # row_order=["lp_500", "lp_2000", "lp_4000"],
                    height=4)
g_2.refline(x=0, c="grey", alpha=.2)
g_2.refline(y=0, c="grey", alpha=.2)
g_2.map(sns.lineplot, "standard_angle", "pred_bayes_mean", ls="dotted")
g_2.map(sns.scatterplot, "standard_angle", "pred_IC_mean", marker=(4, 2, 0))
# g_2.map(sns.lineplot, "standard_angle", "pred_IC_mean_mult", ls="dashed", alpha=.2)
g_2.map(sns.lineplot, "standard_angle", "delta_mu")
g_2.map(sns.scatterplot, "standard_angle", "delta_mu", label=False)
# g_2.map(sns.lineplot, "standard_angle", "delta_mu_min", alpha=.01)
# g_2.map(sns.lineplot, "standard_angle", "delta_mu_max", alpha=.01)
# g_2.set(ylim=(-15, 40))
g_2.add_legend()
# [ax.legend() for ax in g_2.axes.flatten()]
# plt.savefig("Interaural Intrinsic Constraint pilot PSE diff plots.png", dpi=400)


# ======== PSE error plot

g_pse_error = sns.FacetGrid(df_plot,
                            hue="trial_type",
                            hue_order=["itd-->itd", "ild-->ild", "ild-->itd", "itd-->ild", "ild-->both",  "itd-->both"],
                            col="cue_combination",
                            # row="stim_name",
                            height=3)
g_pse_error.map(sns.lineplot, "standard_angle", "sigma")
g_pse_error.map(sns.scatterplot, "standard_angle", "sigma")
g_pse_error.add_legend()


# g_3 = sns.FacetGrid(
#     df_itd_vs_ild,
#     # col="subject_id",
#     # col_order=['jakab', 'jakab_lp_200', 'jakab_lp_500', 'jakab_lp_1000', 'jakab_lp_1500']
# )
# g_3.refline(x=0, c="grey", alpha=.2)
# g_3.refline(y=0, c="grey", alpha=.2)
# g_3.map(sns.regplot, "standard_angle", "mu_pred", color="black", ci=None)
# # g_3.map(sns.regplot, "standard_angle", "sigma_pred", color="grey", scatter=False)
# g_3.add_legend()

df_count_check = df_final.groupby(["subject_id", "stim_name", "standard_angle", "trial_type"]).count()
print(df_count_check[df_count_check["Unnamed: 0"] < 300])


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

