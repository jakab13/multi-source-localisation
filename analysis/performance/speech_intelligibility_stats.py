import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as st

df = pd.read_csv("Dataframes/numerosity_judgement_filtered_excl.csv")

# Aggregate to stabilise random-effects estimation:
# mean response per participant × plane × stim_type × stim_number
agg = (df.dropna(subset=["resp_number","stim_number","plane","stim_type","subject_id"])
         .groupby(["subject_id","plane","stim_type","stim_number"], observed=True)
         .agg(resp_number_mean=("resp_number","mean"))
         .reset_index())

agg["subject_id"] = agg["subject_id"].astype("category")
agg["plane"] = agg["plane"].astype("category")
agg["stim_type"] = agg["stim_type"].astype("category")

formula = ("resp_number_mean ~ stim_number + stim_type + plane "
           "+ stim_number:plane + stim_number:stim_type + stim_type:plane")

# Fixed-effects-only baseline
m_fixed = smf.ols(formula, data=agg).fit()

# Mixed model with participant random intercept + random slope (gain)
m_mixed = smf.mixedlm(formula, data=agg, groups=agg["subject_id"],
                      re_formula="~stim_number").fit(reml=False, method="lbfgs")

# Likelihood-ratio test for participant random effects
lr = 2 * (m_mixed.llf - m_fixed.llf)
df_lr = 3  # Var(intercept), Var(slope), Cov(intercept,slope)
p = st.chi2.sf(lr, df=df_lr)

print("LRT chi2 =", lr, "df =", df_lr, "p =", p)
print(m_mixed.summary())
