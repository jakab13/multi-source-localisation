from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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

# If your plane labels are different, adjust above. For safety:
df = df.dropna(subset=["coverage_used"])

# -----------------------------
# 3) Set categorical types
# -----------------------------
df["subject_id"] = df["subject_id"].astype("category")
df["plane"] = df["plane"].astype("category")
df["stim_type"] = df["stim_type"].astype("category")
df["stim_number"] = df["stim_number"].astype(int).astype("category")

# -----------------------------
# 4) Within-cell centring (recommended)
# Centre within subject × plane × stim_number × stim_type
# -----------------------------
cell_cols = ["subject_id", "plane", "stim_number", "stim_type"]
df["coverage_c"] = df["coverage_used"] - df.groupby(cell_cols)["coverage_used"].transform("mean")

# Scale to “per +0.10” units if coverage is 0..1
df["coverage_c_0p10"] = df["coverage_c"] / 0.10

# -----------------------------
# 5) Fit combined MixedLM
# -----------------------------
formula = "resp_number ~ plane * stim_number * stim_type * coverage_c_0p10"

md = smf.mixedlm(
    formula=formula,
    data=df,
    groups=df["subject_id"],
    re_formula="1",   # random intercept
)

m = md.fit(method="lbfgs", reml=False)

print("\n=== MixedLM summary ===")
print(m.summary())

print("\n=== Omnibus Wald tests for model terms ===")
print(m.wald_test_terms(skip_single=False))

# -----------------------------
# 6) Simple slopes: coverage effect per stim_type × plane × stim_number
# (MixedLM-safe contrasts: fixed effects only + 2D constraint matrix)
# -----------------------------
fe_names = list(m.fe_params.index)

plane_levels = list(df["plane"].cat.categories)
stim_levels = list(df["stim_number"].cat.categories)
type_levels = list(df["stim_type"].cat.categories)

def make_slope_contrast(fe_names, plane_level: str, stim_level: str, type_level: str) -> np.ndarray:
    L = pd.Series(0.0, index=fe_names)

    base = "coverage_c_0p10"
    if base not in L.index:
        raise RuntimeError(f"Expected fixed-effect '{base}' not found.\nFE terms: {fe_names}")
    L[base] = 1.0

    candidates = [
        f"stim_type[T.{type_level}]:coverage_c_0p10",
        f"plane[T.{plane_level}]:coverage_c_0p10",
        f"stim_number[T.{stim_level}]:coverage_c_0p10",

        f"plane[T.{plane_level}]:stim_type[T.{type_level}]:coverage_c_0p10",
        f"plane[T.{plane_level}]:stim_number[T.{stim_level}]:coverage_c_0p10",
        f"stim_number[T.{stim_level}]:stim_type[T.{type_level}]:coverage_c_0p10",

        f"plane[T.{plane_level}]:stim_number[T.{stim_level}]:stim_type[T.{type_level}]:coverage_c_0p10",
    ]

    for term in candidates:
        if term in L.index:
            L[term] = 1.0

    return np.atleast_2d(L.values)

rows = []
for stype in type_levels:
    for pl in plane_levels:
        for sn in stim_levels:
            L = make_slope_contrast(fe_names, plane_level=str(pl), stim_level=str(sn), type_level=str(stype))
            test = m.t_test(L)

            slope = float(test.effect)
            se = float(test.sd)
            pval = float(test.pvalue)

            ci_lo = slope - 1.96 * se
            ci_hi = slope + 1.96 * se

            rows.append({
                "stim_type": str(stype),
                "plane": str(pl),
                "stim_number": int(sn),
                "slope_per_+0.10_cov": slope,
                "SE": se,
                "CI95_lo": ci_lo,
                "CI95_hi": ci_hi,
                "p_value": pval,
            })

slopes = pd.DataFrame(rows).sort_values(["stim_type", "plane", "stim_number"]).reset_index(drop=True)
print("\n=== Simple slopes (Δresp per +0.10 coverage_used) ===")
print(slopes.to_string(index=False))

# FDR across 30 tests
rej, qvals, _, _ = multipletests(slopes["p_value"].values, method="fdr_bh")
slopes["q_fdr_bh"] = qvals
slopes["sig_q<0.05"] = rej

print("\n=== Simple slopes with FDR correction (BH) ===")
print(slopes.to_string(index=False))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


columns = [
    "stim_type", "plane", "stim_number",
    "slope_per_+0.10_cov", "SE", "CI95_lo", "CI95_hi",
    "p_value", "q_fdr_bh", "sig_q<0.05"
]
df_slopes = pd.DataFrame(slopes, columns=columns)

# -----------------------------
# Plot settings
# -----------------------------
planes = ["horizontal", "vertical", "distance"]  # requested order
stim_types = ["forward", "reversed"]
stim_numbers = [2, 3, 4, 5, 6]

bar_w = 0.22
type_gap = 0.02

color = {"forward": "#1f77b4", "reversed": "#ff7f0e"}

fig, axes = plt.subplots(1, len(planes), figsize=(14, 5), sharey=True)

for i, plane in enumerate(planes):
    ax = axes[i]
    base_x = np.arange(len(stim_numbers))  # one group per stim_number

    for j, stim_type in enumerate(stim_types):
        subset = (
            df_slopes[(df_slopes["plane"] == plane) & (df_slopes["stim_type"] == stim_type)]
            .sort_values("stim_number")
            .set_index("stim_number")
            .reindex(stim_numbers)
            .reset_index()
        )

        # x positions (side-by-side bars)
        x = base_x + (j - (len(stim_types) - 1) / 2) * (bar_w + type_gap)
        y = subset["slope_per_+0.10_cov"].to_numpy()

        # CI-based error bars
        yerr = np.vstack([
            y - subset["CI95_lo"].to_numpy(),
            subset["CI95_hi"].to_numpy() - y
        ])

        ax.bar(x, y, width=bar_w, label=stim_type, yerr=yerr, capsize=2, color=color[stim_type])

        # Mark FDR-significant bars with star above CI
        sig = subset[subset["sig_q<0.05"] == True]
        if not sig.empty:
            idx = [stim_numbers.index(int(s)) for s in sig["stim_number"].to_list()]
            sig_x = base_x[idx] + (j - (len(stim_types) - 1) / 2) * (bar_w + type_gap)
            sig_y = sig["CI95_hi"].to_numpy()
            ax.scatter(sig_x, sig_y, marker="*", s=120)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(plane.capitalize())
    ax.set_xlabel("n presented")
    ax.set_xticks(base_x)
    ax.set_xticklabels(stim_numbers)

axes[0].set_ylabel("Effect of spectro-temporal coverage (per +10%)")
axes[-1].legend(title="Stimulus type", loc="best")

plt.tight_layout()
plt.savefig("analysis/spectral_coverage/spectral_coverage_simple_slopes_faceted_bars.png", dpi=300, bbox_inches="tight")
plt.savefig("analysis/spectral_coverage/spectral_coverage_simple_slopes_faceted_bars.svg", format="svg", bbox_inches="tight")
plt.show()
