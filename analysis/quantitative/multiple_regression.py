from analysis.utils.plotting import *
import seaborn as sns
sns.set_theme()
from statsmodels.formula.api import ols


fromfp = "/home/max/labplatform/data/csv/final_df_general.csv"
df = pd.read_csv(fromfp, index_col=1)
df.pop("Sub_ID")

# scatterplot matrix
sns.pairplot(df)

# multiple regression
formula = "performance ~ locaaccu_babble + locaaccu_noise + spatmask + coverage"
model = ols(formula, data=df).fit()
model.summary()
