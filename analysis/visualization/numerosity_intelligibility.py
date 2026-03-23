import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.metrics import mean_squared_error

# -----------------------
# Load data
# -----------------------
path = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(path)

exclude_subjects = []  # e.g., ["sub_106", "sub_116", "sub_119", "sub_120"]
if exclude_subjects:
    df = df[~df["subject_id"].isin(exclude_subjects)].copy()

cols = ["subject_id", "plane", "stim_type", "stim_number", "resp_number"]
df = df[cols].dropna()
df["stim_number"] = pd.to_numeric(df["stim_number"])
df["resp_number"] = pd.to_numeric(df["resp_number"])

plane_order = ["horizontal", "vertical", "distance"]
stim_order = ["forward", "reversed"]

# -----------------------
# Base colors (matplotlib defaults)
# -----------------------
base_hex = {
    "forward":  "#1f77b4",  # tab:blue
    "reversed": "#ff7f0e",  # tab:orange
}

def hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def ramp_from_base_hls(base_hex_color: str,
                       t: float,
                       L_dark: float = 0.22,
                       L_light: float = 0.95,
                       S_dark: float = 0.95,
                       S_light: float = 0.55) -> tuple:
    """
    HLS ramp anchored to a base color.
    t=0 -> darkest (best), t=1 -> lightest (worst)
    """
    r, g, b = hex_to_rgb01(base_hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    L = (1 - t) * L_dark + t * L_light
    S = (1 - t) * S_dark + t * S_light
    return colorsys.hls_to_rgb(h, L, S)

# -----------------------
# Rank subjects by overall RMSE (best -> worst)
# -----------------------
rmse_by_subject = (
    df.groupby("subject_id", observed=True)
      .apply(lambda g: mean_squared_error(g["stim_number"], g["resp_number"], squared=False))
      .rename("rmse")
      .reset_index()
      .sort_values("rmse")
)

subjects_sorted = rmse_by_subject["subject_id"].tolist()
n_sub = len(subjects_sorted)

# Evenly spaced color positions by rank (NOT by RMSE magnitude)
if n_sub == 1:
    sub2t = {subjects_sorted[0]: 0.0}
else:
    sub2t = {sid: i / (n_sub - 1) for i, sid in enumerate(subjects_sorted)}

# -----------------------
# Layout: 2 rows x (3 planes + 1 narrow key column)
# -----------------------
fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(
    nrows=len(stim_order),
    ncols=len(plane_order) + 1,
    width_ratios=[1, 1, 1, 0.18],  # narrow last column
    wspace=0.15,
    hspace=0.15
)

xmin = float(df["stim_number"].min())
xmax = float(df["stim_number"].max())

# -----------------------
# Plot panels
# -----------------------
for r, stim_type in enumerate(stim_order):
    base = base_hex[stim_type]

    # --- main three panels ---
    for c, plane in enumerate(plane_order):
        ax = fig.add_subplot(gs[r, c])
        subdf = df[(df["stim_type"] == stim_type) & (df["plane"] == plane)].copy()

        # subject curves (one per subject)
        for sid in subjects_sorted:
            g = subdf[subdf["subject_id"] == sid]
            if g.empty:
                continue

            # average within subject x stim_number (so each subject has a clean curve)
            g2 = (g.groupby("stim_number", as_index=False, observed=True)["resp_number"]
                    .mean()
                    .sort_values("stim_number"))

            t = float(sub2t[sid])  # rank-based, evenly spaced
            color = ramp_from_base_hls(base, t)

            ax.plot(
                g2["stim_number"].to_numpy(),
                g2["resp_number"].to_numpy(),
                linewidth=2.0,
                # alpha=0.85,
                color=color,
                zorder=1
            )

        # group mean overlay
        # mean_curve = (subdf.groupby("stim_number", as_index=False, observed=True)["resp_number"]
        #                   .mean()
        #                   .sort_values("stim_number"))
        #
        # ax.plot(
        #     mean_curve["stim_number"].to_numpy(),
        #     mean_curve["resp_number"].to_numpy(),
        #     linewidth=3.0,
        #     marker="o",
        #     color="black",
        #     zorder=3
        # )

        # equality line
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.0, color="gray", zorder=0)

        if r == 0:
            ax.set_title(plane)
        if c == 0:
            ax.set_ylabel(f"{stim_type}\n\nresp_number")
        if r == len(stim_order) - 1:
            ax.set_xlabel("stim_number")
        ax.set_xlim(xmin - 0.1, xmax + 0.1)
        ax.set_ylim(xmin - 0.1, xmax + 0.6)

    # --- narrow right-hand column: color key ---
    ax_key = fig.add_subplot(gs[r, len(plane_order)])
    if r == 0:
        ax_key.set_title("RMSE rank")

    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(-0.5, n_sub - 0.5)

    # Best on top, worst on bottom
    y = np.arange(n_sub)[::-1]
    x = np.full(n_sub, 0.5)

    colors = [ramp_from_base_hls(base, sub2t[sid]) for sid in subjects_sorted]  # best->worst

    ax_key.scatter(x, y, s=60, c=colors, edgecolor="none")
    ax_key.text(0.5, y[0] + 0.4, "best", ha="center", va="bottom", fontsize=9)
    ax_key.text(0.5, y[-1] - 0.4, "worst", ha="center", va="top", fontsize=9)

    ax_key.set_xticks([])
    ax_key.set_yticks([])
    for sp in ax_key.spines.values():
        sp.set_visible(False)

fig.suptitle(
    "Numerosity judgement: rows=stim_type, cols=plane\n"
    "Subject colors evenly spaced by RMSE rank (best darker → worst lighter); mean in black",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_name = "figures/numerosity_intelligibility"
plt.savefig(out_name + ".png", dpi=300, bbox_inches="tight")
plt.savefig(out_name + ".svg", format="svg", bbox_inches="tight")
plt.show()
