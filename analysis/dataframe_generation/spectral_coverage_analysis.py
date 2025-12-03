import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- 1. Load data ---
df = pd.read_csv("Dataframes/numerosity_judgement_spectral_coverage_otsu.csv")

df = df[~(df.resp_number == 1)]

excluded_subs = ["sub_116", "sub_119", "sub_120", "sub_106"]
df = df[~df.subject_id.isin(excluded_subs)]

# df = df[df.stim_type == "forward"]

# --- 2. Keep only the planes of interest ---
planes_of_interest = ["horizontal", "vertical", "distance"]
df = df[df["plane"].isin(planes_of_interest)]

# --- 3. Compute subject means *plus* trial counts per condition ---

subj_means = (
    df.groupby(["plane", "stim_number",
                "stim_type",
                "resp_number"])
    .agg(
        spectral_coverage=("spectral_coverage", "mean"),
        n_trials=("spectral_coverage", "size")  # how many rows went into that mean
    )
    .reset_index()
)

# stim_number as category (for nicer facet labelling if needed)
subj_means["stim_number"] = subj_means["stim_number"].astype(int).astype("category")
subj_means["resp_number"] = subj_means["resp_number"].astype(int).astype("category")

group_means = (
    df.groupby(["plane", "stim_number", "resp_number"], as_index=False)
    ["spectral_coverage"]
    .mean()
)

# =========================================
# Scatter plot
# =========================================

# assume subj_means is already computed and contains:
# plane, stim_number (category or int), resp_number, spectral_coverage, n_trials

# Make sure stim_number is categorical (for ordering), then get numeric codes
subj_means["stim_number"] = subj_means["stim_number"].astype(int).astype("category")

# 1) Base x-position: numeric code for stim_number (0, 1, 2, …)
stim_codes = subj_means["stim_number"].cat.codes.astype(float)
subj_means["stim_code"] = stim_codes  # this is definitely float now

# 2) Offset per resp_number
resp_levels = sorted(subj_means["resp_number"].unique())
offsets = np.linspace(-0.3, 0.3, len(resp_levels))  # tweak range if needed
offset_map = dict(zip(resp_levels, offsets))

# Make sure resp_number used for mapping has no weird dtype issues
# (int is fine, category is also fine, mapping returns float)
offset_series = subj_means["resp_number"].map(offset_map).astype(float)

# 3) Jittered x = base code + small resp_number-specific offset
subj_means["x_jittered"] = subj_means["stim_code"].astype(float) + offset_series

orig = plt.cm.get_cmap("Blues")
new = orig(np.linspace(0.2, 0.9, 256))  # shift bright end darker
blues_darker = ListedColormap(new)

hue_levels = sorted(subj_means["resp_number"].unique())
n_hues = len(hue_levels)

# Sample equally spaced colors from your custom Blues colormap
palette_list = blues_darker(np.linspace(0, 1, n_hues))

g = sns.relplot(
    data=subj_means,
    col="plane",
    col_order=planes_of_interest,
    x="x_jittered",
    y="spectral_coverage",
    hue="resp_number",
    style="stim_type",
    size="n_trials",
    sizes=(10, 300),
    alpha=0.5,
    kind="scatter",
    height=4,
    aspect=1,
    facet_kws={"sharey": True, "sharex": True},
    palette=palette_list,
    edgecolor="none"
)

# 4) Fix x-ticks to show stim_number labels instead of numeric codes
for ax in g.axes.flat:
    # get the stim_number categories in order
    cats = subj_means["stim_number"].cat.categories
    # their numeric positions (0,1,2,…) = same as stim_code
    positions = np.arange(len(cats))
    ax.set_xticks(positions)
    ax.set_xticklabels(cats)

g.set_axis_labels("Stimulus number (stim_number)", "Spectral coverage (subject mean)")
g.set_titles("Plane: {col_name}")
g._legend.set_title("resp_number / n_trials")

# g.fig.savefig("numerosity_spectral_coverage.svg", format="svg", bbox_inches="tight")

# plt.tight_layout()
plt.show()

# LMM for spectral coverage as a predictor for resp_number

import statsmodels.formula.api as smf

coef_rows = []

for plane in planes_of_interest:
    for stim in sorted(df["stim_number"].unique()):
    # for stim_type in df["stim_type"].unique():
        subset = df[(df["plane"] == plane) & (df["stim_number"] == stim)]
        # Minimum data requirements
        if subset["subject_id"].nunique() < 2 or len(subset) < 10:
            continue

        # Mixed model
        md = smf.mixedlm(
            "resp_number ~ spectral_coverage + stim_type",
            subset,
            groups=subset["subject_id"]
        )

        try:
            mdf = md.fit(reml=False)
        except Exception as e:
            print(f"Model failed for plane={plane}, stim={stim}: {e}")
            continue

        intercept = mdf.fe_params["Intercept"]
        slope = mdf.fe_params["spectral_coverage"]

        # Collect SE, t-value, p-value
        se = mdf.bse["spectral_coverage"]
        tval = mdf.tvalues["spectral_coverage"]
        pval = mdf.pvalues["spectral_coverage"]

        # Optional: AIC/BIC
        aic = mdf.aic
        bic = mdf.bic

        coef_rows.append({
            "plane": plane,
            "stim_number": stim,
            # "stim_type": stim_type,
            "intercept": intercept,
            "slope": slope,
            "se": se,
            "tval": tval,
            "pval": pval,
            "aic": aic,
            "bic": bic
        })

coefs_df = pd.DataFrame(coef_rows)
print(coefs_df)

# 1) Ensure types are nice
coefs_df["stim_number"] = coefs_df["stim_number"].astype(int).astype("category")


# 2) Map p-values to significance stars
def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


coefs_df["sig"] = coefs_df["pval"].apply(p_to_stars)

# 3) Faceted barplot of slopes
g = sns.catplot(
    data=coefs_df,
    col="plane",
    col_order=planes_of_interest,  # ["horizontal", "vertical", "distance"]
    # row="stim_type",
    # hue="stim_type",
    x="stim_number",
    y="slope",
    kind="bar",
    height=4,
    aspect=1,
    color="tab:blue"
)

g.set_axis_labels("Stimulus number (stim_number)",
                  "Effect of spectral coverage on resp_number (LMM slope)")
g.set_titles("Plane: {col_name}")


# 4) Add stars above each bar according to p-value
for ax, plane in zip(g.axes.flat, planes_of_interest):
    # subset coefs for this plane, in same order as bars on x-axis
    sub = coefs_df[coefs_df["plane"] == plane].sort_values("stim_number")
    # x positions are 0, 1, 2, ... in that order
    for i, (_, row) in enumerate(sub.iterrows()):
        y = row["slope"]
        star = row["sig"]
        if star == "":
            star = "n.s."
            # continue  # no stars for non-significant

        # small offset above the bar (takes sign into account)
        offset = 0.05 * (1 if y >= 0 else -1)

        y = y + offset if y >= 0 else 0

        ax.text(
            i,                  # x position (bar center index)
            y,         # y position just above (or below) the bar
            star,
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=11,
            fontweight="bold"
        )

# g.fig.savefig("numerosity_spectral_coverage_stats.svg", format="svg", bbox_inches="tight")
plt.show()

