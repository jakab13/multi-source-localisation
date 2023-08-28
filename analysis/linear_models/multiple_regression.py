from analysis.utils.plotting import *
import seaborn as sns
from statsmodels.regression.linear_model import GLS
import statsmodels.api as sm
sns.set_theme()


fromfp = "/home/max/labplatform/data/linear_model/final_df_revspeech_v.csv"
df = pd.read_csv(fromfp, index_col=0)

# multiple regression
x = df.drop(["response"], axis=1)
y = df.response
x_added_constant = sm.add_constant(x)
model = GLS(y, x_added_constant).fit()
model.summary()

# scatterplot matrix
sns.pairplot(df)

"""
Model Summary:

coef: The measurement of how much a variable changes in relation to the dependent variable. A negative value indicates an inverse relationship.
std err: The standard error of the coefficient.
t: The t-statistic, which is the coefficient divided by the standard error.
P>|t|: The probability that the coefficient is measured through the model by chance.

Omnibus: The normalcy of the distribution of residuals using skewness and kurtosis as measurements. A value of 0 indicates perfect normalcy.
Prob(Omnibus): A statistical test measuring the probability the residuals are normally distributed. A value of 1 indicates perfectly normal distribution.
Skew: The measurement of symmetry in the data, with 0 indicating perfect symmetry.
Kurtosis: The measure of peakiness of the data, or its concentration around 0 in a normal curve. Higher kurtosis implies fewer outliers.
Durbin-Watson: The measurement of homoscedasticity, or an even distribution of errors throughout the data. Heteroscedasticity would imply an uneven distribution, for example as the data point grows higher the relative error grows higher. Ideal homoscedasticity will lie between 1 and 2.
Jarque-Bera (JB) and Prob(JB): Alternate methods of measuring the same value as Omnibus and Prob(Omnibus) using skewness and kurtosis.
"""

