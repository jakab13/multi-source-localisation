"""
Spectral coverage → numerosity judgement
Mixed-effects analysis with:
- subject_id random intercept
- fixed effects: plane × stim_number × stim_type
- coverage effect varying by plane × stim_number (collapsed across stim_type)
- within-cell centring of coverage within subject×plane×stim_number×stim_type
- clear omnibus Wald tests + simple-slope table with BH–FDR correction (15 tests)

Requirements:
  pip install pandas numpy statsmodels patsy scipy

Adjust PATH below to your project "Dataframes" folder file.
"""

import warnings
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -----------------------------
# 0) Load
# -----------------------------
PATH = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"  # <-- change to your Dataframes folder path
df = pd.read_csv(PATH)

# basic hygiene
df["subject_id"] = df["subject_id"].astype(str)
df["plane"] = df["plane"].astype(str).str.lower()
df["stim_type"] = df["stim_type"].astype(str).str.lower()
df["stim_number"] = df["stim_number"].astype(int)

# keep only the two stim types of interest
df = df.loc[df["stim_type"].isin(["forward", "reversed"])].copy()

# drop NAs in required columns
req = ["subject_id", "plane", "stim_type", "stim_number", "resp_number",
       "spectral_coverage", "spectral_coverage_2"]
df = df.dropna(subset=req).copy()

# ----------------------------------------
# 1) Define coverage_used (matches earlier)
# ----------------------------------------
# horizontal & vertical: use spectral_coverage_2
# distance: use spectral_coverage
df["coverage_used"] = np.where(
    df["plane"].isin(["horizontal", "vertical"]),
    df["spectral_coverage_2"],
    df["spectral_coverage"],
)

# ----------------------------------------
# 2) Within-cell centring of coverage
#    within subject×plane×stim_number×stim_type
# ----------------------------------------
cell = ["subject_id", "plane", "stim_number", "stim_type"]
df["coverage_c"] = df["coverage_used"] - df.groupby(cell)["coverage_used"].transform("mean")

# interpretability: per +0.10 coverage
df["coverage_c_0p10"] = df["coverage_c"] / 0.10

# ----------------------------------------
# 3) Set categorical reference levels
#    (distance, 2, forward as baselines)
# ----------------------------------------
df["plane"] = pd.Categorical(df["plane"], categories=["distance", "horizontal", "vertical"], ordered=True)
df["stim_number"] = pd.Categorical(df["stim_number"], categories=[2, 3, 4, 5, 6], ordered=True)
df["stim_type"] = pd.Categorical(df["stim_type"], categories=["forward", "reversed"], ordered=True)

print("\n=== Data summary ===")
print(f"File: {PATH}")
print(f"N rows: {len(df)}")
print(f"N subjects: {df['subject_id'].nunique()}")
print("Trials per plane:", df["plane"].value_counts().to_dict())
print("Trials per stim_type:", df["stim_type"].value_counts().to_dict())
print("Coverage_used summary:\n", df["coverage_used"].describe())


# ----------------------------------------
# 4) Fit MixedLM (collapsed coverage across stim_type)
# ----------------------------------------
# - stim_type is included (and all interactions with plane×stim_number)
# - coverage slope varies with plane×stim_number
# - coverage does NOT interact with stim_type => collapsed effect by design
formula = (
    "resp_number ~ C(plane)*C(stim_number)*C(stim_type)"
    " + coverage_c_0p10*C(plane)*C(stim_number)"
)

print("\n=== Fitting MixedLM ===")
print("Formula:\n", formula)

# These warnings are common with MixedLM; we’ll show them but still proceed
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = smf.mixedlm(formula, df, groups=df["subject_id"]).fit(
        reml=False, method="lbfgs", maxiter=200
    )
    if len(w) > 0:
        print("\n--- Warnings during fit (informative, not necessarily fatal) ---")
        for ww in w[:10]:
            print(f"{type(ww.message).__name__}: {ww.message}")
        if len(w) > 10:
            print(f"... ({len(w) - 10} more warnings)")

print("\n=== Model fit summary (fixed effects excerpt) ===")
print(m.summary())


# ----------------------------------------
# 5) Omnibus Wald tests for model terms
# ----------------------------------------
print("\n=== Omnibus Wald tests for model terms (MixedLM) ===")
# statsmodels provides a convenient term-wise omnibus test
wt = m.wald_test_terms()
# wt is a DataFrame-like; printing it directly is usually readable
print(wt)


# ----------------------------------------
# 6) Simple slopes of coverage for each plane×stim_number
#    and collapsed across stim_type by averaging forward/reversed contrasts.
#
# Key idea:
#   Build a 1-row design matrix using patsy + design_info,
#   compute L = X(coverage=1) - X(coverage=0),
#   then estimate slope = L@beta with correct SE from fixed-effect covariance.
# ----------------------------------------

# fixed-effect names + covariance restricted to fixed effects
fe_names = list(m.fe_params.index)
beta = m.fe_params.to_numpy()

cov_all = m.cov_params()
if isinstance(cov_all, pd.DataFrame):
    cov_fe = cov_all.loc[fe_names, fe_names]
    V = cov_fe.to_numpy()
else:
    k = len(fe_names)
    V = cov_all[:k, :k]

design_info = m.model.data.design_info  # Patsy DesignInfo for fixed effects

def design_row(plane, stim_number, stim_type, coverage_val):
    """Return fixed-effect design row aligned to fe_names."""
    new = pd.DataFrame({
        "plane": [plane],
        "stim_number": [stim_number],
        "stim_type": [stim_type],
        "coverage_c_0p10": [coverage_val],
    })
    # match categories (important)
    new["plane"] = pd.Categorical(new["plane"], categories=df["plane"].cat.categories, ordered=True)
    new["stim_number"] = pd.Categorical(new["stim_number"], categories=df["stim_number"].cat.categories, ordered=True)
    new["stim_type"] = pd.Categorical(new["stim_type"], categories=df["stim_type"].cat.categories, ordered=True)

    X = patsy.dmatrix(design_info, new, return_type="dataframe")
    X = X.reindex(columns=fe_names)
    return X.to_numpy().reshape(-1)

def slope_from_contrast(L):
    est = float(L @ beta)
    se = float(np.sqrt(L @ V @ L))
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p

planes = ["horizontal", "vertical", "distance"]
nums = [2, 3, 4, 5, 6]
types = ["forward", "reversed"]

rows = []
for pl in planes:
    for sn in nums:
        # type-specific slopes (diagnostic)
        L_f = design_row(pl, sn, "forward", 1.0) - design_row(pl, sn, "forward", 0.0)
        L_r = design_row(pl, sn, "reversed", 1.0) - design_row(pl, sn, "reversed", 0.0)

        est_f, se_f, z_f, p_f = slope_from_contrast(L_f)
        est_r, se_r, z_r, p_r = slope_from_contrast(L_r)

        # collapsed across stim_type (equal-weight average)
        L_coll = 0.5 * (L_f + L_r)
        est_c, se_c, z_c, p_c = slope_from_contrast(L_coll)

        rows.append({
            "plane": pl,
            "stim_number": sn,
            "slope_forward": est_f,
            "SE_forward": se_f,
            "p_forward": p_f,
            "slope_reversed": est_r,
            "SE_reversed": se_r,
            "p_reversed": p_r,
            "slope_collapsed": est_c,
            "SE_collapsed": se_c,
            "CI95_lo": est_c - 1.96 * se_c,
            "CI95_hi": est_c + 1.96 * se_c,
            "p_collapsed": p_c,
        })

res = pd.DataFrame(rows)

# BH–FDR across the 15 collapsed tests (3 planes × 5 numerosities)
res["q_fdr_bh"] = multipletests(res["p_collapsed"], method="fdr_bh")[1]
res["sig_q<0.05"] = res["q_fdr_bh"] < 0.05

# nice formatting for printout
def fmt_p(x):
    if pd.isna(x):
        return "nan"
    if x < 1e-4:
        return "<1e-4"
    return f"{x:.4f}"

print("\n=== Simple slopes: effect of coverage (per +0.10) ===")
print("Interpretation: change in resp_number per +10 percentage point increase in within-cell centred coverage.")
out_cols = [
    "plane", "stim_number",
    "slope_collapsed", "SE_collapsed", "CI95_lo", "CI95_hi",
    "p_collapsed", "q_fdr_bh", "sig_q<0.05",
]
tmp = res[out_cols].copy()
tmp["p_collapsed"] = tmp["p_collapsed"].map(fmt_p)
tmp["q_fdr_bh"] = tmp["q_fdr_bh"].map(fmt_p)
print(tmp.sort_values(["plane", "stim_number"]).to_string(index=False))

print("\n=== (Diagnostic) Type-specific slopes (not FDR-corrected here) ===")
diag_cols = ["plane","stim_number","slope_forward","p_forward","slope_reversed","p_reversed"]
tmp2 = res[diag_cols].copy()
tmp2["p_forward"] = tmp2["p_forward"].map(fmt_p)
tmp2["p_reversed"] = tmp2["p_reversed"].map(fmt_p)
print(tmp2.sort_values(["plane","stim_number"]).to_string(index=False))


# ----------------------------------------
# 7) Optional: save results table
# ----------------------------------------
OUT_CSV = "Dataframes/coverage_effects_collapsed_by_plane_stimnumber.csv"
res.to_csv(OUT_CSV, index=False)
print(f"\nSaved results table to: {OUT_CSV}")
