from analysis.utils.plotting import *
import seaborn as sns
import statsmodels.api as sm
sns.set_theme()


fromfp = "/home/max/labplatform/data/linear_model/final_df_revspeech_v.csv"
df = pd.read_csv(fromfp, index_col=0)

X = df[["coverage", "numjudge", "spatmask", "lababble", "lanoise"]]
X = sm.add_constant(X)
y = df.response
est = sm.OLS(y, X).fit()
est.summary()
