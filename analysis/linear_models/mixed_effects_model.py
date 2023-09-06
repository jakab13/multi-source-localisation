from analysis.utils.plotting import *
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import researchpy as rp
import scipy.stats as stats
sns.set_theme()


fromfp = "/home/max/labplatform/data/linear_model/final_df_clearspeech_h.csv"
df = pd.read_csv(fromfp, index_col=0)

# look at the performance based on the other variables
rp.summary_cont(df)

# multiple regression
# random intercept and slope
vc_formula={"coverage": "0 + coverage",
            "spatmask": "0 + spatmask",
            "lababble": "0 + lababble",
            "lanoise": "0 + lanoise",
            "numjudge": "0 + numjudge"}
model = smf.mixedlm("performance ~ coverage + spatmask + lababble + lanoise + numjudge", data=df,
                    groups=df["subID"], vc_formula=vc_formula)
result = model.fit()
result.summary()

# scatterplot matrix
sns.pairplot(df)

# boxplot
boxplot = df.boxplot(["performance"], by=["coverage"], showmeans=True, notch=True)

# error distribution plot
ax = sns.distplot(result.resid, hist=False, kde_kws={"shade": True, "lw": 1}, fit=stats.norm)
ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")

# Q-QPLot
sm.qqplot(result.resid, dist=stats.norm, line='s')
plt.set_title("Q-Q Plot")

# shapiro wilk test
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(result.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)



















"""
oldenburg
berlin stipendium
imprss
school of cognition
aachen institute for hearing acoustics
gesa hartwigsen
stephan ebener
"""
