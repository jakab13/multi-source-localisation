import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
path = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(path)

use = df[['subject_id', 'plane', 'stim_type', 'stim_number', 'resp_number']].dropna().copy()
use['stim_number'] = pd.to_numeric(use['stim_number'])
use['resp_number'] = pd.to_numeric(use['resp_number'])

plane_order = ["horizontal", "vertical", "distance"]
stim_order = ["forward", "reversed"]
c_forward, c_reversed = "#1f77b4", "#ff7f0e"  # matplotlib defaults

# -----------------------
# Helper: OLS slope of resp ~ stim
# -----------------------
def slope_ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(np.var(x), 0):
        return np.nan
    return np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)

# -----------------------
# Compute slope per subject × plane × stim_type
# -----------------------
rows = []
for (sid, plane, st), g in use.groupby(['subject_id', 'plane', 'stim_type'], observed=True):
    sl = slope_ols(g['stim_number'], g['resp_number'])
    rows.append({'subject_id': sid, 'plane': plane, 'stim_type': st, 'slope': sl})

slope_df = pd.DataFrame(rows)

# -----------------------
# Compute overall slope per subject across ALL planes & stim_types
# (used ONLY for ranking subjects)
# -----------------------
overall = []
for sid, g in use.groupby('subject_id', observed=True):
    sl = slope_ols(g['stim_number'], g['resp_number'])
    overall.append({'subject_id': sid, 'slope_overall': sl})

overall_df = pd.DataFrame(overall).dropna()

# Rank by "overall performance": closeness of slope to 1 (veridical)
# If you'd rather rank by largest slope, swap to: sort_values('slope_overall', ascending=False)
overall_df['score'] = (overall_df['slope_overall'] - 1.0).abs()
overall_df = overall_df.sort_values('score', ascending=True)

subjects_sorted = overall_df['subject_id'].tolist()
n_sub = len(subjects_sorted)

# y positions: best at top
ypos = np.arange(n_sub)[::-1]
sub2y = dict(zip(subjects_sorted, ypos))

# -----------------------
# Prepare wide format for plotting forward vs reversed in each plane
# -----------------------
wide = (slope_df.pivot_table(index=['subject_id', 'plane'], columns='stim_type', values='slope', observed=True)
                .reset_index())

# Anchor within plane: mean slope across stim_type (forward+reversed)
anchor = (slope_df.groupby(['subject_id', 'plane'], observed=True)['slope']
                  .mean()
                  .rename('slope_plane')
                  .reset_index())

wide = wide.merge(anchor, on=['subject_id', 'plane'], how='left')

# -----------------------
# Plot: caterpillar-style, columns = plane
# Points = forward/reversed, black anchor = within-plane mean
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

for i, plane in enumerate(plane_order):
    ax = axes[i]
    p = wide[wide['plane'] == plane].copy()
    p['y'] = p['subject_id'].map(sub2y)

    # Paired forward vs reversed per subject
    for _, row in p.iterrows():
        y = row['y']
        if pd.isna(y):
            continue

        fwd = row.get('forward', np.nan)
        rev = row.get('reversed', np.nan)

        if pd.notna(fwd) and pd.notna(rev):
            ax.plot([fwd, rev], [y, y], color='lightgray', linewidth=1, zorder=1)

        if pd.notna(fwd):
            ax.scatter(fwd, y, s=35, color=c_forward, alpha=0.95, zorder=3, edgecolor="none")
        if pd.notna(rev):
            ax.scatter(rev, y, s=35, color=c_reversed, alpha=0.95, zorder=3, edgecolor="none")

    # Anchor line (within-plane mean across stim_type) + connecting line as visual guide
    p_anchor = p.dropna(subset=['y', 'slope_plane']).sort_values('y')
    ax.plot(p_anchor['slope_plane'], p_anchor['y'], color='black', linewidth=2, zorder=2)
    ax.scatter(p_anchor['slope_plane'], p_anchor['y'], color='black', s=18, zorder=4)

    # Reference: ideal slope = 1
    # ax.axvline(1.0, linestyle=':', linewidth=1.2, color='gray', zorder=0)

    # Plane mean slopes (separate forward/reversed), optional
    plane_sub = slope_df[slope_df['plane'] == plane]
    mu_fwd = plane_sub.loc[plane_sub['stim_type'] == 'forward', 'slope'].mean()
    mu_rev = plane_sub.loc[plane_sub['stim_type'] == 'reversed', 'slope'].mean()
    ax.axvline(mu_fwd, linestyle='--', linewidth=1.4, color=c_forward, alpha=0.9, zorder=0)
    ax.axvline(mu_rev, linestyle='--', linewidth=1.4, color=c_reversed, alpha=0.9, zorder=0)

    ax.set_title(plane)
    ax.set_xlabel("Slope of resp_number ~ stim_number")
    # ax.grid(True, axis='x')

# Y labels on left
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(subjects_sorted)
axes[0].set_ylabel("Subjects (ranked by overall |slope−1| across all data)")

# Legend
handles = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c_forward, markersize=7, label='forward'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c_reversed, markersize=7, label='reversed'),
    plt.Line2D([0], [0], color='lightgray', linewidth=2, label='paired difference'),
    plt.Line2D([0], [0], color='black', linewidth=2, label='within-plane mean (forward+reversed)'),
    plt.Line2D([0], [0], color='gray', linewidth=1.2, linestyle=':', label='ideal slope = 1'),
    plt.Line2D([0], [0], color=c_forward, linewidth=2, linestyle='--', label='plane mean (forward)'),
    plt.Line2D([0], [0], color=c_reversed, linewidth=2, linestyle='--', label='plane mean (reversed)'),
]
# axes[2].legend(handles=handles, frameon=True, loc='upper right')

fig.suptitle(
    "Caterpillar plot of subject-wise numerosity slopes by plane\n"
    "Subjects ranked by overall performance (closeness of slope to 1) across all trials",
    y=0.98
)

plt.tight_layout()
out_name = "figures/numerosity_caterpillar_slope"
plt.savefig(out_name + ".png", dpi=300, bbox_inches="tight")
plt.savefig(out_name + ".svg", format="svg", bbox_inches="tight")
plt.show()
