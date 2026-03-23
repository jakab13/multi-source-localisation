from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
import patsy
from statsmodels.stats.multitest import multipletests

PATH = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(PATH)

# -----------------------------
# 1) Basic cleaning / filtering
# -----------------------------
df["stim_type"] = df["stim_type"].astype(str).str.lower()
df["plane"] = df["plane"].astype(str).str.lower()

# Keep both types (forward + reversed) so we can test the reversed condition too
df = df.loc[df["stim_type"].isin(["forward", "reversed"])].copy()

required = ["subject_id", "plane", "stim_number", "stim_type",
            "spectral_coverage", "spectral_coverage_2", "resp_number"]
df = df.dropna(subset=required)

# -----------------------------
# 2) Create plane-specific coverage predictor
#    - horizontal & vertical: spectral_coverage_2
#    - distance: spectral_coverage
# -----------------------------
df["coverage_used"] = np.where(
    df["plane"].isin(["horizontal", "vertical"]),
    df["spectral_coverage_2"],
    df["spectral_coverage"],
)

# -----------------------------
# 1) Within-cell centring of coverage (per subject × plane × stim_number × stim_type)
# -----------------------------
cell = ["subject_id", "plane", "stim_number", "stim_type"]
df = df.copy()

df["coverage_c"] = df["coverage_used"] - df.groupby(cell)["coverage_used"].transform("mean")

# scale to "per +0.10" to make slopes directly interpretable
df["coverage_c_0p10"] = df["coverage_c"] / 0.10

# Ensure categorical coding is stable
df["plane"] = pd.Categorical(df["plane"], categories=["distance", "horizontal", "vertical"])
df["stim_type"] = pd.Categorical(df["stim_type"], categories=["forward", "reversed"])
df["stim_number"] = pd.Categorical(df["stim_number"].astype(int), categories=[2,3,4,5,6])

# -----------------------------
# 2) Fit the simplified MixedLM:
#    - stim_type is included (nuisance / control)
#    - coverage effect varies by plane × stim_number
#    - coverage does NOT vary by stim_type (collapsed effect)
# -----------------------------
# -----------------------------
# Fit the model (simplified, collapsed coverage across stim_type)
# -----------------------------
formula = (
    "resp_number ~ C(plane)*C(stim_number)*C(stim_type)"
    " + coverage_c_0p10*C(plane)*C(stim_number)"
)

m = smf.mixedlm(formula, df, groups=df["subject_id"]).fit(reml=False, method="lbfgs")

fe_names = list(m.fe_params.index)
beta = m.fe_params.to_numpy()

cov_all = m.cov_params()
cov_fe = cov_all.loc[fe_names, fe_names] if isinstance(cov_all, pd.DataFrame) else cov_all[:len(fe_names), :len(fe_names)]
V = cov_fe.to_numpy() if isinstance(cov_fe, pd.DataFrame) else cov_fe

# Grab the design_info used to create the original exog
design_info = m.model.data.design_info  # Patsy DesignInfo

def design_row(plane, stim_number, stim_type, coverage_val):
    new = pd.DataFrame({
        "plane": [plane],
        "stim_number": [stim_number],
        "stim_type": [stim_type],
        "coverage_c_0p10": [coverage_val],
    })

    # Make categories consistent with training data (important)
    if hasattr(df["plane"], "cat"):
        new["plane"] = pd.Categorical(new["plane"], categories=df["plane"].cat.categories)
    if hasattr(df["stim_number"], "cat"):
        new["stim_number"] = pd.Categorical(new["stim_number"], categories=df["stim_number"].cat.categories)
    if hasattr(df["stim_type"], "cat"):
        new["stim_type"] = pd.Categorical(new["stim_type"], categories=df["stim_type"].cat.categories)

    X = patsy.dmatrix(design_info, new, return_type="dataframe")
    # Ensure column order matches fe_params exactly
    X = X.reindex(columns=fe_names)
    return X.to_numpy().reshape(-1)

def slope_for_condition(plane, stim_number, stim_type):
    x0 = design_row(plane, stim_number, stim_type, coverage_val=0.0)
    x1 = design_row(plane, stim_number, stim_type, coverage_val=1.0)  # +0.10 coverage
    L = x1 - x0

    est = float(L @ beta)
    se = float(np.sqrt(L @ V @ L))
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return est, se, p, L

# -----------------------------
# Collapsed slopes across stim_type (average forward & reversed)
# -----------------------------
planes = ["horizontal", "vertical", "distance"]
nums = [2, 3, 4, 5, 6]
types = ["forward", "reversed"]

rows = []
for pl in planes:
    for sn in nums:
        est_f, se_f, p_f, L_f = slope_for_condition(pl, sn, "forward")
        est_r, se_r, p_r, L_r = slope_for_condition(pl, sn, "reversed")

        # Equal-weight collapsed slope (properly, by averaging contrasts)
        L_coll = 0.5 * (L_f + L_r)
        est_c = float(L_coll @ beta)
        se_c = float(np.sqrt(L_coll @ V @ L_coll))
        z_c = est_c / se_c if se_c > 0 else np.nan
        p_c = 2 * (1 - norm.cdf(abs(z_c)))

        rows.append({
            "plane": pl,
            "stim_number": sn,
            "slope_forward": est_f,
            "slope_reversed": est_r,
            "slope_collapsed": est_c,
            "SE_collapsed": se_c,
            "CI95_lo": est_c - 1.96 * se_c,
            "CI95_hi": est_c + 1.96 * se_c,
            "p_value": p_c,
        })


res = pd.DataFrame(rows)
res["q_fdr_bh"] = multipletests(res["p_value"], method="fdr_bh")[1]
res["sig_q<0.05"] = res["q_fdr_bh"] < 0.05

print(res.sort_values(["plane", "stim_number"]))

import matplotlib.pyplot as plt

bar_w = 0.22
type_gap = 0.02

fig, axes = plt.subplots(1, len(planes), figsize=(14, 5), sharey=True)

for i, plane in enumerate(planes):
    ax = axes[i]
    base_x = np.arange(len(nums))  # one group per stim_number

    subset = (
        res[(res["plane"] == plane)]
        .sort_values("stim_number")
        .set_index("stim_number")
        .reset_index()
    )

    # x positions (side-by-side bars)
    x = base_x + (bar_w + type_gap)
    y = subset["slope_collapsed"].to_numpy()

    # CI-based error bars
    yerr = np.vstack([
        y - subset["CI95_lo"].to_numpy(),
        subset["CI95_hi"].to_numpy() - y
    ])

    ax.bar(x, y, width=bar_w, yerr=yerr, capsize=2)

    # Mark FDR-significant bars with star above CI
    sig = subset[subset["sig_q<0.05"] == True]
    if not sig.empty:
        idx = [nums.index(int(s)) for s in sig["stim_number"].to_list()]
        sig_x = base_x[idx] + (bar_w + type_gap)
        sig_y = sig["CI95_hi"].to_numpy()
        ax.scatter(sig_x, sig_y, marker="*", s=120)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(plane.capitalize())
    ax.set_xlabel("n presented")
    ax.set_xticks(base_x)
    ax.set_xticklabels(nums)

axes[0].set_ylabel("Effect of spectro-temporal coverage (per +10%)")
axes[-1].legend(title="Stimulus type", loc="best")

plt.tight_layout()
plt.savefig("analysis/spectral_coverage/spectral_coverage_collapsed_slopes_faceted_bars.png", dpi=300, bbox_inches="tight")
plt.savefig("analysis/spectral_coverage/spectral_coverage_collapsed_slopes_faceted_bars.svg", format="svg", bbox_inches="tight")
plt.show()
