from statsmodels.regression.linear_model import OLS
import itertools
import pandas as pd


def mallows_ck(k, X, y):
    results = []

    def process_subset(feature_set, X, y):
        # Fit model on feature_set and calculate RSS
        model = OLS(y, X[list(feature_set)])
        regr = model.fit()
        RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
        return {"model": regr, "RSS": RSS}

    for combo in itertools.combinations(X.columns, k):
        results.append(process_subset(combo, X, y))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    print("Processed", models.shape[0], "models on", k, "predictors")
    # Return the best model, along with some other useful information about the model
    return best_model


if __name__ == "__main__":
    fromfp = "/home/max/labplatform/data/csv/final_df_general.csv"
    df = pd.read_csv(fromfp, index_col=1)
    df.pop("Sub_ID")
    y = df.performance
    df.pop("performance")
    x = df
    # Could take quite awhile to complete...
    models_best = pd.DataFrame(columns=["RSS", "model"])
    for k in range(1, (len(x.columns))):
        models_best.loc[k] = mallows_ck(k, x, y)
    print(models_best.loc[3, "model"].summary())
    print(mallows_ck(3, x, y)["model"].summary())
