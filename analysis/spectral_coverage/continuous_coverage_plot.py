from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
PATH = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"

OUT_DIR = "analysis/spectral_coverage"
OUT_PNG = os.path.join(OUT_DIR, "coverage_continuous_binned_subjectavg_lines.png")
OUT_SVG = os.path.join(OUT_DIR, "coverage_continuous_binned_subjectavg_lines.svg")

N_BINS = 20
PLANES = ["horizontal", "vertical", "distance"]
STIM_NUMS = [2, 3, 4, 5, 6]
STIM_TYPES = ["forward", "reversed"]

# =========================================================
# LOAD + PREPARE DATA
# =========================================================
df = pd.read_csv(PATH)

df["stim_type"] = df["stim_type"].astype(str).str.lower()
df["plane"] = df["plane"].astype(str).str.lower()
df["subject_id"] = df["subject_id"].astype(str)
df["stim_number"] = df["stim_number"].astype(int)

# keep both forward and reversed for consistency with your earlier analysis
df = df.loc[df["stim_type"].isin(STIM_TYPES)].copy()

required = [
    "subject_id", "plane", "stim_number", "stim_type",
    "spectral_coverage", "spectral_coverage_2", "resp_number"
]
df = df.dropna(subset=required).copy()

# Plane-specific coverage variable
df["coverage_used"] = np.where(
    df["plane"].isin(["horizontal", "vertical"]),
    df["spectral_coverage_2"],
    df["spectral_coverage"],
)

# =========================================================
# WITHIN-CELL Z-SCORING
# =========================================================
# Match your modelling logic:
# per subject × plane × stim_number × stim_type
cell = ["subject_id", "plane", "stim_number", "stim_type"]

df["coverage_mean_cell"] = df.groupby(cell)["coverage_used"].transform("mean")
df["coverage_sd_cell"] = df.groupby(cell)["coverage_used"].transform("std")

# remove cells with undefined/zero SD
df = df.loc[df["coverage_sd_cell"].notna() & (df["coverage_sd_cell"] > 0)].copy()

df["coverage_z"] = (
    (df["coverage_used"] - df["coverage_mean_cell"]) / df["coverage_sd_cell"]
)

# =========================================================
# BINNED OBSERVED DATA
# =========================================================
df_plot = df.copy()
# df_plot["coverage_bin"] = pd.cut(df_plot["coverage_z"], bins=N_BINS)


def make_quantile_bins(x, n_bins):
    ranks = x.rank(method="first")
    return pd.qcut(ranks, q=n_bins, duplicates="drop")


df_plot["coverage_bin"] = (
    df_plot.groupby(["plane", "stim_number"], observed=True)["coverage_z"]
    .transform(lambda x: make_quantile_bins(x, N_BINS))
)
df_binned = (
    df_plot.groupby(["plane", "stim_number", "coverage_bin"], observed=True)
    .agg(
        coverage_mean=("coverage_z", "mean"),
        resp_mean=("resp_number", "mean"),
        resp_sem=("resp_number", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n=("resp_number", "size"),
    )
    .reset_index()
)

df_binned = df_binned.dropna(subset=["coverage_mean", "resp_mean"]).copy()

y_min = df_binned["resp_mean"].min()
y_max = df_binned["resp_mean"].max()

y_pad = 0.18 * (y_max - y_min)
y_lim_low = y_min - y_pad
y_lim_high = y_max + y_pad

# =========================================================
# SUBJECT-WISE REGRESSIONS
# =========================================================
# We first fit subject-wise lines within:
# subject × plane × stim_number
# pooling over forward/reversed so the final line reflects
# the average relationship across subject for that condition.

rows = []

for (sub, plane, stim_number), g in df.groupby(
    ["subject_id", "plane", "stim_number"], observed=True
):
    g = g.dropna(subset=["coverage_z", "resp_number"]).copy()

    # need at least 2 unique x values
    if len(g) < 3 or g["coverage_z"].nunique() < 2:
        continue

    x = g["coverage_z"].to_numpy()
    y = g["resp_number"].to_numpy()

    # simple OLS line: y = b0 + b1*x
    slope, intercept = np.polyfit(x, y, deg=1)

    rows.append({
        "subject_id": sub,
        "plane": plane,
        "stim_number": int(stim_number),
        "slope": slope,
        "intercept": intercept,
        "n_trials": len(g),
    })

df_subject_lines = pd.DataFrame(rows)

# Average slopes/intercepts across subjects within plane × stim_number
df_mean_lines = (
    df_subject_lines.groupby(["plane", "stim_number"], observed=True)
    .agg(
        mean_slope=("slope", "mean"),
        mean_intercept=("intercept", "mean"),
        sem_slope=("slope", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        sem_intercept=("intercept", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_subjects=("subject_id", "nunique"),
    )
    .reset_index()
)

print("\nSubject-averaged line parameters:")
print(df_mean_lines.sort_values(["plane", "stim_number"]))

# =========================================================
# PLOT
# =========================================================
sns.set(style="white", context="talk")

palette = sns.color_palette("viridis", n_colors=len(STIM_NUMS))
color_map = dict(zip(STIM_NUMS, palette))

plane_titles = {
    "horizontal": "Azimuth",
    "vertical": "Elevation",
    "distance": "Distance",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

# robust x-range based on observed z values
x_min = df_plot["coverage_z"].quantile(0.02)
x_max = df_plot["coverage_z"].quantile(0.98)
xgrid = np.linspace(x_min, x_max, 200)

for ax, plane in zip(axes, PLANES):
    sub_b = df_binned[df_binned["plane"] == plane]
    sub_l = df_mean_lines[df_mean_lines["plane"] == plane]

    for sn in STIM_NUMS:
        # -------------------------
        # binned observed points
        # -------------------------
        d = (
            sub_b[sub_b["stim_number"].astype(int) == sn]
            .sort_values("coverage_mean")
            .copy()
        )

        # ax.errorbar(
        #     d["coverage_mean"],
        #     d["resp_mean"],
        #     yerr=d["resp_sem"],
        #     fmt="none",
        #     ecolor=color_map[sn],
        #     alpha=0.25,
        #     elinewidth=1.0,
        #     capsize=0,
        #     zorder=2,
        # )

        ax.scatter(
            d["coverage_mean"],
            d["resp_mean"],
            s=48,
            marker="o",
            alpha=0.25,
            edgecolors="none",
            color=color_map[sn],
            label=str(sn) if plane == PLANES[0] else None,
            zorder=3,
        )

        # -------------------------
        # subject-averaged line
        # -------------------------
        line_row = sub_l[sub_l["stim_number"].astype(int) == sn]
        if len(line_row) == 0 or len(d) == 0:
            continue

        line_row = line_row.iloc[0]

        x_line_min = d["coverage_mean"].min()
        x_line_max = d["coverage_mean"].max()
        xgrid = np.linspace(x_line_min, x_line_max, 100)

        ygrid = line_row["mean_intercept"] + line_row["mean_slope"] * xgrid

        ax.plot(
            xgrid,
            ygrid,
            color=color_map[sn],
            linewidth=3,
            zorder=4,
        )

    ax.set_title(plane_titles[plane])
    ax.set_xlabel("Coverage (z-scored)")
    x_min = df_plot["coverage_z"].quantile(0.02)
    x_max = df_plot["coverage_z"].quantile(0.98)
    x_pad = 0.10 * (x_max - x_min)
    x_lim_low = x_min - x_pad
    x_lim_high = x_max + x_pad
    ax.set_xlim(x_lim_low, x_lim_high)
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.axhline(0, linestyle="--", linewidth=1, color="0.6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Perceived numerosity")
# axes[0].legend(title="Presented numerosity", frameon=False)

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_SVG, format="svg", bbox_inches="tight")
plt.show()