import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(path)

use = df[['subject_id', 'plane', 'stim_type', 'stim_number', 'resp_number']].dropna().copy()
use['stim_number'] = pd.to_numeric(use['stim_number'])
use['resp_number'] = pd.to_numeric(use['resp_number'])

plane_order = ["horizontal", "vertical", "distance"]
stim_order = ["forward", "reversed"]
colors = {"forward": "#1f77b4", "reversed": "#ff7f0e"}  # matplotlib defaults

# One curve per subject: average within subject x stim_number, per plane & stim_type
sub_curve = (
    use.groupby(['plane','stim_type','subject_id','stim_number'], observed=True)['resp_number']
       .mean()
       .reset_index()
)

# Group summary curve across subjects (mean or median)
mean_curve = (
    sub_curve.groupby(['plane','stim_type','stim_number'], observed=True)['resp_number']
             .mean()
             .reset_index()
)
median_curve = (
    sub_curve.groupby(['plane','stim_type','stim_number'], observed=True)['resp_number']
             .median()
             .reset_index()
)

summary_kind = "mean"  # change to "median" if you prefer
summary_df = mean_curve if summary_kind == "mean" else median_curve

# Plot: facets by plane (columns), overlay stim_type hues in each
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

xmin = float(use['stim_number'].min())
xmax = float(use['stim_number'].max())

for i, plane in enumerate(plane_order):
    ax = axes[i]
    ax.set_title(plane)

    for stim_type in stim_order:
        c = colors[stim_type]

        # Subject-level thin lines
        p = sub_curve[(sub_curve['plane'] == plane) & (sub_curve['stim_type'] == stim_type)]
        for sid, g in p.groupby('subject_id', observed=True):
            g = g.sort_values('stim_number')
            ax.plot(g['stim_number'], g['resp_number'], color=c, alpha=0.25, linewidth=1)

        # Group summary thick line
        s = summary_df[(summary_df['plane'] == plane) & (summary_df['stim_type'] == stim_type)].sort_values('stim_number')
        ax.plot(s['stim_number'], s['resp_number'], color=c, linewidth=3, label=f"{stim_type} ({summary_kind})")

    # Line of equality
    ax.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='gray', linewidth=1)

    ax.set_xlabel("stim_number")
    ax.grid(True)

axes[0].set_ylabel("resp_number")
axes[2].legend(frameon=True, loc='upper left')

fig.suptitle(f"Numerosity judgement by plane with subject curves and {summary_kind} response", y=1.02)
plt.tight_layout()
plt.show()
