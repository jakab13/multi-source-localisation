from statsmodels.regression.linear_model import OLS
import time
import itertools
import pandas as pd


def process_subset(feature_set, X, y):
    # Fit model on feature_set and calculate RSS
    model = OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model": regr, "RSS": RSS}


def mallows_ck(k, X, y):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(process_subset(combo, X, y))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc - tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model


if __name__ == "__main__":
    pass
