import math
import sklearn
from sklearn.linear_model import LinearRegression


def gain(x, y):
    return LinearRegression().fit(x, y)


def possible_combinations(n, k):
    return math.comb(n, k)


if __name__ == "__main__":
    from analysis.utils.misc import load_dataframe
    from labplatform.config import get_config
    import os
    import pandas as pd
    import seaborn as sn
    dd = os.path.join(get_config("DATA_ROOT"), "MSL")
    df = load_dataframe(data_dir=dd, exp_name="NumJudge", plane="h")
    cm = pd.crosstab(df['response'], df['solution'], rownames=['response'], colnames=['solution'], margins=True,
                     normalize=True)
    cm = cm.drop(index=8)
    sn.heatmap(cm, annot=True)
