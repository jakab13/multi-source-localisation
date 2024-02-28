from analysis.utils.plotting import *
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import researchpy as rp
import scipy.stats as stats
sns.set_theme(style="white")
plt.rcParams['text.usetex'] = True  # TeX rendering


fromfp = "/home/max/labplatform/data/linear_model/final_df_mean_clearspeech_h.csv"
df = pd.read_csv(fromfp, index_col=0)

# look at the performance based on the other variables
rp.summary_cont(df)

# multiple regression
# random intercept and slope
model = smf.mixedlm("response ~ coverage + numjudge + lababble + lanoise + spatmask", data=df,
                    groups=df["subID"])
result = model.fit()
table = result.summary()

# scatterplot matrix
sns.pairplot(df)

# error distribution plot
ax = sns.distplot(result.resid, hist=False, kde_kws={"shade": True, "lw": 1}, fit=stats.norm)
ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

# Q-QPLot
sm.qqplot(result.resid, dist=stats.norm, line='s')

# shapiro wilk test
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(result.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
