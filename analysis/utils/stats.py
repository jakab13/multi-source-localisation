from statsmodels.regression.linear_model import OLS
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def mallows_ck(k, X, y):
    """
    Perform Mallows' C_p criterion for feature selection.

    Parameters:
    - k (int): The number of predictors to consider.
    - X (pandas DataFrame): The feature matrix.
    - y (pandas Series): The target variable.

    Returns:
    - best_model (pandas Series): The best model selected based on Mallows' C_p criterion.

    This function performs Mallows' C_p criterion for feature selection to identify the best model
    with a given number of predictors. It fits multiple linear regression models for all possible
    combinations of k predictors and returns the model with the lowest residual sum of squares (RSS).

    """
    results = []

    def process_subset(feature_set, X, y):
        # Fit a linear regression model and calculate RSS
        model = OLS(y, X[list(feature_set)])
        regr = model.fit()
        RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
        return {"model": regr, "RSS": RSS}

    for combo in itertools.combinations(X.columns, k):
        results.append(process_subset(combo, X, y))

    # Create a DataFrame to store model results
    models = pd.DataFrame(results)

    # Select the model with the lowest RSS
    best_model = models.loc[models['RSS'].argmin()]

    # Print information about the modeling process
    print("Processed", models.shape[0], "models on", k, "predictors")

    # Return the best model and its associated information
    return best_model


def permutation_test(group1, group2, n_permutations=10000, plot=True, **kwargs):
    """
    Perform a permutation test to compare two groups.

    Parameters:
    - group1 (pandas Series or numpy array): Data for the first group.
    - group2 (pandas Series or numpy array): Data for the second group.
    - n_permutations (int, optional): Number of permutations to perform (default is 10000).
    - plot (bool, optional): Whether to plot the permutation distribution (default is True).

    Returns:
    - p_value (float): The p-value for the permutation test.

    The permutation test assesses whether the difference between the means of two groups is
    statistically significant by randomly permuting the data and computing the p-value.

    """

    # Initialize a list to save the results of each permutation
    results = list()

    # Concatenate the two groups to create a pooled dataset
    pooled = pd.concat([group1, group2])

    # Get the length of the first group (used for sampling)
    len_group = len(group1)

    # Calculate the observed difference in means between the two groups
    observed_diff = group1.mean() - group2.mean()

    # Perform n_permutations permutations
    for _ in range(n_permutations):
        # Randomly permute the pooled data
        permuted = np.random.permutation(pooled)

        # Calculate the mean for each permuted group
        assigned1 = permuted[:len_group].mean()
        assigned2 = permuted[len_group:].mean()

        # Calculate the difference in means for this permutation
        results.append(assigned1 - assigned2)

    # Convert results to a numpy array and take absolute values
    results = np.abs(np.array(results))

    # Count how many permutations have a difference as extreme as or more extreme than observed_diff
    values_as_or_more_extreme = sum(results >= observed_diff)

    # Calculate the p-value
    num_simulations = results.shape[0]
    p_value = values_as_or_more_extreme / num_simulations

    if plot:
        # Plot the permutation distribution and observed difference
        density_plot = sns.kdeplot(results, fill=True, **kwargs)
        density_plot.set(
            xlabel='Absolute Mean Difference Between Groups',
            ylabel='Proportion of Permutations'
        )

        # Add a line to show the actual difference observed in the data
        density_plot.axvline(
            x=observed_diff,
            color='red',
            linestyle='--'
        )

        # Add a legend to the plot
        plt.legend(
            labels=['Permutation Distribution', f'Observed Difference: {round(observed_diff, 2)}'],
            loc='upper right'
        )

        # Display the plot
        plt.show()

    return p_value


if __name__ == "__main__":
    pass
