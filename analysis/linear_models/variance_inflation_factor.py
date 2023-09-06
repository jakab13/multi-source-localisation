from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd


fromfp = "/home/max/labplatform/data/csv/final_df_general.csv"
df = pd.read_csv(fromfp, index_col=1)
df.pop("Sub_ID")

X = add_constant(df)  # add constant of 1

# Variance inflation factor
ds = pd.Series([variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])], index=X.columns)

print(ds)
