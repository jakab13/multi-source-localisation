import numpy as np
import os
from labplatform.config import get_config
from analysis.utils.misc import load_dataframe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")
plt.rcParams['text.usetex'] = True  # TeX rendering


# permutation test
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")

n_permutations = 10000

# divide reversed speech blocks from clear speech
filledh = dfh.reversed_speech.ffill()
revspeechh = dfh[np.where(filledh == True, True, False)]  # True where reversed_speech is True
clearspeechh = dfh[np.where(filledh == False, True, False)]  # True where reversed_speech is False

filledv = dfv.reversed_speech.ffill()
revspeechv = dfv[np.where(filledv == True, True, False)]  # True where reversed_speech is True
clearspeechv = dfv[np.where(filledv == False, True, False)]  # True where reversed_speech is False

# compare group means
group1h = revspeechh.response.fillna(0)
group2h = clearspeechh.response.fillna(0)

group1v = revspeechv.response.fillna(0)
group2v = clearspeechv.response.fillna(0)

resultsh = list()

# Concatenate the two groups to create a pooled dataset
pooled = pd.concat([group1h, group2h])

# Get the length of the first group (used for sampling)
len_group = len(group1h)

# Calculate the observed difference in means between the two groups
observed_diff = group1h.mean() - group2h.mean()

# Perform n_permutations permutations
for _ in range(n_permutations):
    # Randomly permute the pooled data
    permuted = np.random.permutation(pooled)

    # Calculate the mean for each permuted group
    assigned1 = permuted[:len_group].mean()
    assigned2 = permuted[len_group:].mean()

    # Calculate the difference in means for this permutation
    resultsh.append(assigned1 - assigned2)

# Convert results to a numpy array and take absolute values
results = np.abs(np.array(resultsh))

# Count how many permutations have a difference as extreme as or more extreme than observed_diff
values_as_or_more_extreme = sum(results >= observed_diff)

# Calculate the p-value
num_simulations = results.shape[0]
p_value = values_as_or_more_extreme / num_simulations

# Plot the permutation distribution and observed difference
density_plot = sns.kdeplot(results, fill=True)
density_plot.set(
    xlabel='Absolute Mean Difference Between Groups',
    ylabel='Proportion of Permutations'
)
# Add a line to show the actual difference observed in the data
density_plot.axvline(
    x=observed_diff,
    linestyle='--',
    color="orange"
)


resultsv = list()

# Concatenate the two groups to create a pooled dataset
pooled = pd.concat([group1v, group2v])

# Get the length of the first group (used for sampling)
len_group = len(group1v)

# Calculate the observed difference in means between the two groups
observed_diff = group1v.mean() - group2v.mean()

# Perform n_permutations permutations
for _ in range(n_permutations):
    # Randomly permute the pooled data
    permuted = np.random.permutation(pooled)

    # Calculate the mean for each permuted group
    assigned1 = permuted[:len_group].mean()
    assigned2 = permuted[len_group:].mean()

    # Calculate the difference in means for this permutation
    resultsv.append(assigned1 - assigned2)

# Convert results to a numpy array and take absolute values
results = np.abs(np.array(resultsv))

# Count how many permutations have a difference as extreme as or more extreme than observed_diff
values_as_or_more_extreme = sum(results >= observed_diff)

# Calculate the p-value
num_simulations = results.shape[0]
p_value = values_as_or_more_extreme / num_simulations

# Plot the permutation distribution and observed difference
density_plot = sns.kdeplot(resultsv, fill=True)
density_plot.set(
    xlabel='Absolute Mean Difference Between Groups',
    ylabel='Proportion of Permutations'
)
# Add a line to show the actual difference observed in the data
density_plot.axvline(
    x=observed_diff,
    linestyle='--',
    color="blue"
)

