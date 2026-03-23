import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv("Dataframes/spatial_unmasking_filtered_excl.csv")

# Keep only rows with the required fields
df = df.dropna(subset=["plane", "subject_id", "masker_speaker_loc", "threshold"]).copy()

# Choose a consistent facet order (if present)
plane_order = ["horizontal", "vertical", "distance"]
planes = [p for p in plane_order if p in df["plane"].unique()] + [
    p for p in df["plane"].unique() if p not in plane_order
]

# Create faceted figure
fig, axes = plt.subplots(1, len(planes), figsize=(14, 4), sharey=True)
if len(planes) == 1:
    axes = [axes]

for ax, plane in zip(axes, planes):
    sub = df[df["plane"] == plane].copy()

    # Subject-level trajectories (thin lines)
    for sid, ssub in sub.groupby("subject_id"):
        ssub = ssub.sort_values("masker_speaker_loc")
        ax.plot(
            ssub["masker_speaker_loc"].values,
            ssub["threshold"].values,
            linewidth=0.8,
            color="gray",
            alpha=0.3,
        )

    # Group mean trajectory (thick line)
    mean_df = (
        sub.groupby("masker_speaker_loc", as_index=False)["threshold"]
        .mean()
        .sort_values("masker_speaker_loc")
    )
    ax.plot(
        mean_df["masker_speaker_loc"].values,
        mean_df["threshold"].values,
        linewidth=5.0,
        color="black"
    )

    ax.set_title(f"{plane.capitalize()} plane")
    ax.set_xlabel("Masker location (°)" if plane in ["horizontal", "vertical"] else "Masker distance (m)")

axes[0].set_ylabel("Threshold (dB TMR)")
fig.suptitle("Spatial unmasking thresholds by plane (subject trajectories and group mean)", y=1.03)
fig.tight_layout()

# Save
out_name = "figures/spatial_unmasking_faceted_by_plane"
fig.savefig(out_name + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out_name + ".svg", format="svg", bbox_inches="tight")

