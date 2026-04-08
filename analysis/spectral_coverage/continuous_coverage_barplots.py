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
OUT_PNG = os.path.join(OUT_DIR, "coverage_effect_subjectavg_barplots.png")
OUT_SVG = os.path.join(OUT_DIR, "coverage_effect_subjectavg_barplots.svg")

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
# Same cell logic as before:
# subject × plane × stim_number × stim_type
cell = ["subject_id", "plane", "stim_number", "stim_type"]

df["coverage_mean_cell"] = df.groupby(cell)["coverage_used"].transform("mean")
df["coverage_sd_cell"] = df.groupby(cell)["coverage_used"].transform("std")

# Remove undefined / zero-SD cells
df = df.loc[df["coverage_sd_cell"].notna() & (df["coverage_sd_cell"] > 0)].copy()

df["coverage_z"] = (
    (df["coverage_used"] - df["coverage_mean_cell"]) / df["coverage_sd_cell"]
)

# =========================================================
# SUBJECT-WISE SLOPES
# =========================================================
# Pool across forward/reversed within each subject × plane × stim_number
rows = []

for (sub, plane, stim_number), g in df.groupby(
    ["subject_id", "plane", "stim_number"], observed=True
):
    g = g.dropna(subset=["coverage_z", "resp_number"]).copy()

    if len(g) < 3 or g["coverage_z"].nunique() < 2:
        continue

    x = g["coverage_z"].to_numpy()
    y = g["resp_number"].to_numpy()

    slope, intercept = np.polyfit(x, y, deg=1)

    rows.append({
        "subject_id": sub,
        "plane": plane,
        "stim_number": int(stim_number),
        "slope": slope,
        "intercept": intercept,
        "n_trials": len(g),
    })

df_subject_slopes = pd.DataFrame(rows)

# =========================================================
# AGGREGATE FOR BARPLOT
# =========================================================
df_bar = (
    df_subject_slopes.groupby(["plane", "stim_number"], observed=True)
    .agg(
        slope_mean=("slope", "mean"),
        slope_sem=("slope", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_subjects=("subject_id", "nunique"),
    )
    .reset_index()
)

# 95% CI (optional; use instead of SEM if you prefer)
df_bar["ci95_lo"] = df_bar["slope_mean"] - 1.96 * df_bar["slope_sem"]
df_bar["ci95_hi"] = df_bar["slope_mean"] + 1.96 * df_bar["slope_sem"]

print(df_bar.sort_values(["plane", "stim_number"]))

# =========================================================
# PLOT
# =========================================================
sns.set(style="white", context="talk")

plane_titles = {
    "horizontal": "Azimuth",
    "vertical": "Elevation",
    "distance": "Distance",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)

bar_color = sns.color_palette("viridis", n_colors=5)

# y-limits with padding
y_min = df_bar["ci95_lo"].min()
y_max = df_bar["ci95_hi"].max()
y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 0.1
y_lim_low = min(0, y_min - y_pad)
y_lim_high = y_max + y_pad

for ax, plane in zip(axes, PLANES):
    sub = (
        df_bar[df_bar["plane"] == plane]
        .sort_values("stim_number")
        .copy()
    )

    x = np.arange(len(sub))
    y = sub["slope_mean"].to_numpy()
    yerr = np.vstack([
        y - sub["ci95_lo"].to_numpy(),
        sub["ci95_hi"].to_numpy() - y
    ])

    ax.bar(
        x,
        y,
        color=bar_color,
        width=0.72,
        edgecolor="none",
        zorder=2,
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="0.2",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )

    ax.axhline(0, linestyle="--", linewidth=1, color="0.6", zorder=1)
    ax.set_title(plane_titles[plane])
    ax.set_xlabel("Presented numerosity")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["stim_number"].astype(int).tolist())
    ax.set_ylim(y_lim_low, y_lim_high)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Effect of spectrotemporal coverage (slope per z-unit)")

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_SVG, format="svg", bbox_inches="tight")
plt.show()