from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --------------------------------------------------
# Paths
# --------------------------------------------------
base = Path("Dataframes/performance")
# If you are running outside that folder, use absolute paths instead:
# base = Path("/mnt/data")

loc_path = base / "metrics_localisation.csv"
num_path = base / "metrics_numerosity_linear.csv"
unm_path = base / "metrics_unmasking_summary.csv"

OUT_PNG = os.path.join(base, "spatial_hearing_vs_numerosity_performance.png")
OUT_SVG = os.path.join(base, "spatial_hearing_vs_numerosity_performance.svg")

# --------------------------------------------------
# Load data
# --------------------------------------------------
loc = pd.read_csv(loc_path)
num = pd.read_csv(num_path)
unm = pd.read_csv(unm_path)

# --------------------------------------------------
# Inspect / harmonise
# --------------------------------------------------
# Expected columns:
# loc: subject_id, plane, stim_type, sens, sens_kind, ...
# num: subject_id, plane, stim_type, num_slope, ...
# unm: subject_id, plane, unmask_best_delta_tmr

# Optional: rename plane labels for prettier plotting
plane_order = ["horizontal", "vertical", "distance"]
plane_labels = {
    "horizontal": "Azimuth",
    "vertical": "Elevation",
    "distance": "Distance",
}

# --------------------------------------------------
# Summarise participant-level metrics
# --------------------------------------------------

# Localisation:
# Average across noise/babble within subject x plane
loc_sum = (
    loc.groupby(["subject_id", "plane"], as_index=False)
       .agg(
           localisation_metric=("sens", "mean"),
           localisation_r2=("r2", "mean")
       )
)

# Numerosity:
# Option A: average across forward and reversed
# num_sum = (
#     num.groupby(["subject_id", "plane"], as_index=False)
#        .agg(
#            numerosity_slope=("num_slope", "mean"),
#            numerosity_mae=("num_mae", "mean")
#        )
# )

# If instead you want only forward, use this:
num_sum = (
    num[num["stim_type"] == "forward"]
    .groupby(["subject_id", "plane"], as_index=False)
    .agg(
        numerosity_slope=("num_slope", "mean"),
        numerosity_mae=("num_mae", "mean")
    )
)

# Unmasking:
# The values are negative deltas in your file.
# Flip sign so that larger positive values = larger masking release
unm_sum = unm.copy()
unm_sum["spatial_release"] = -unm_sum["unmask_best_delta_tmr"]
unm_sum = unm_sum[["subject_id", "plane", "spatial_release"]]

# --------------------------------------------------
# Merge for plotting
# --------------------------------------------------
df_loc = pd.merge(
    num_sum,
    loc_sum,
    on=["subject_id", "plane"],
    how="inner"
)

df_unm = pd.merge(
    num_sum,
    unm_sum,
    on=["subject_id", "plane"],
    how="inner"
)

# --------------------------------------------------
# Plot: 2 x 3 panel
# x-axis = numerosity slope
# top row = localisation metric
# bottom row = spatial release
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

for col, plane in enumerate(plane_order):
    # -------------------------
    # Top row: localisation
    # -------------------------
    ax = axes[0, col]
    sub = df_loc[df_loc["plane"] == plane].copy()

    ax.scatter(
        sub["numerosity_slope"],
        sub["localisation_metric"],
        s=40,
        alpha=0.8
    )

    # Fit regression line manually
    if len(sub) >= 2:
        x = sub["numerosity_slope"]
        y = sub["localisation_metric"]
        m, b = pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var(), y.mean() - (pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var()) * x.mean()
        xfit = pd.Series([x.min(), x.max()])
        yfit = m * xfit + b
        ax.plot(xfit, yfit, linewidth=1.5)

        if len(sub) >= 3:
            r, p = pearsonr(x, y)
            ax.text(
                0.05, 0.95,
                f"r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes,
                ha="left", va="top"
            )

    ax.set_title(plane_labels.get(plane, plane))
    if col == 0:
        ax.set_ylabel("Localisation accuracy")
    else:
        ax.set_ylabel("")

    # Optional fixed y-limits
    ax.set_ylim(0, 1.15)

    # -------------------------
    # Bottom row: unmasking
    # -------------------------
    ax = axes[1, col]
    sub = df_unm[df_unm["plane"] == plane].copy()

    ax.scatter(
        sub["numerosity_slope"],
        sub["spatial_release"],
        s=40,
        alpha=0.8
    )

    if len(sub) >= 2:
        x = sub["numerosity_slope"]
        y = sub["spatial_release"]
        m, b = pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var(), y.mean() - (pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var()) * x.mean()
        xfit = pd.Series([x.min(), x.max()])
        yfit = m * xfit + b
        ax.plot(xfit, yfit, linewidth=1.5)

        if len(sub) >= 3:
            r, p = pearsonr(x, y)
            ax.text(
                0.05, 0.95,
                f"r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes,
                ha="left", va="top"
            )

    ax.set_xlabel("Numerosity slope")
    if col == 0:
        ax.set_ylabel("Spatial release (dB)")
    else:
        ax.set_ylabel("")

    # Optional fixed y-limits
    ax.set_ylim(-1, 20)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_SVG, format="svg", bbox_inches="tight")
plt.show()