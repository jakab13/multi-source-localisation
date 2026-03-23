import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["svg.image_inline"] = True
matplotlib.rcParams["svg.hashsalt"] = "fixed"

# -----------------------------
# Load data
# -----------------------------
# Update this path to your "Dataframes" folder file name if needed
PATH = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(PATH)

# df["stim_type"] = df["stim_type"].astype(str).str.lower()
df["plane"] = df["plane"].astype(str).str.lower()
df["subject_id"] = df["subject_id"].astype(str)
df["stim_number"] = df["stim_number"].astype(int)

df = df.copy()

df = df.dropna(
    subset=[
        "subject_id", "plane", "stim_number",
        "resp_number", "spectral_coverage", "spectral_coverage_2"
    ]
)

# -----------------------------
# Define coverage_used (matches your modelling choice)
# -----------------------------
df["coverage_used"] = np.where(
    df["plane"].isin(["horizontal", "vertical"]),
    df["spectral_coverage_2"],
    df["spectral_coverage"],
)

# -----------------------------
# Quartile split within each subject×plane×stim_type×stim_number
# -----------------------------
group_cols = ["subject_id", "plane", "stim_type", "stim_number"]
# group_cols = ["subject_id", "plane", "stim_number"]

def label_quartiles(g: pd.DataFrame, min_trials=8) -> pd.DataFrame:
    x = g["coverage_used"].to_numpy()
    if len(x) < min_trials:
        g["cov_group"] = np.nan
        return g

    q1 = np.quantile(x, 0.10)
    q3 = np.quantile(x, 0.90)

    lab = np.full(len(g), np.nan, dtype=object)
    lab[x <= q1] = "low (Q1)"
    lab[x >= q3] = "high (Q4)"
    g["cov_group"] = lab
    return g

df = df.groupby(group_cols, group_keys=False).apply(label_quartiles)
df_q = df.dropna(subset=["cov_group"]).copy()

# -----------------------------
# Aggregate for plotting:
# (1) per-subject means per cell to avoid trial-count weighting
# (2) group mean + SEM across subjects
# -----------------------------

group_cols = ["subject_id", "plane", "stim_number"]
subj_means = (
    df_q.groupby(group_cols + ["cov_group"], as_index=False)
        .agg(resp_mean=("resp_number", "mean"),
             n_trials=("resp_number", "size"))
)

agg = (
    subj_means.groupby(["plane", "stim_number", "cov_group"], as_index=False)
             .agg(mean_resp=("resp_mean", "mean"),
                  sd_resp=("resp_mean", "std"),
                  n_subj=("resp_mean", "size"))
)
agg["sem_resp"] = agg["sd_resp"] / np.sqrt(agg["n_subj"].clip(lower=1))
# --- colour map you requested ---
COL = {
    # ("forward",  "low (Q1)"):  "#4FA3FF",  # brighter blue
    # ("forward",  "high (Q4)"): "#0B4FA8",  # darker blue
    # ("reversed", "low (Q1)"):  "#FFB15C",  # brighter orange
    # ("reversed", "high (Q4)"): "#CC6D00",  # darker orange
    ("low (Q1)"):  "#4FA3FF",  # brighter blue
    ("high (Q4)"): "#0B4FA8",  # darker blue
}

# Optional: keep markers/linestyles consistent across stim_type
STYLE = {
    "low (Q1)":  dict(marker="o", linestyle="-"),
    "high (Q4)": dict(marker="o", linestyle="-"),
}

plane_order = ["horizontal", "vertical", "distance"]
# stim_type_order = ["forward", "reversed"]
cov_order = ["low (Q1)", "high (Q4)"]

stim_numbers = sorted(df["stim_number"].unique())
# nrows, ncols = len(stim_type_order), len(plane_order)
ncols = len(plane_order)

fig, axes = plt.subplots(1, ncols, figsize=(18, 8), sharey=True, sharex=True)

for c, plane in enumerate(plane_order):
    ax = axes[c]
    a = agg[agg["plane"] == plane].copy()

    for cov_group in cov_order:
        sub = a[a["cov_group"] == cov_group].copy()
        if sub.empty:
            continue

        sub = sub.set_index("stim_number").reindex(stim_numbers).reset_index()

        x = sub["stim_number"].to_numpy(dtype=float)
        y = sub["mean_resp"].to_numpy(dtype=float)
        m = sub["mean_resp"].to_numpy(dtype=float)
        se = sub["sem_resp"].to_numpy(dtype=float)
        yerr = sub["sem_resp"].to_numpy(dtype=float)

        # ax.errorbar(
        #     x, y, yerr=yerr,
        #     capsize=3,
        #     linewidth=1.8,
        #     color=COL[(stype, cov_group)],
        #     label=f"{stype} | {cov_group}",
        #     **STYLE[cov_group],
        # )

        ax.plot(
            x, m,
            linewidth=2,
            color=COL[cov_group],
            label=f"{cov_group}",
        )

        # SEM band (ribbon)
        ax.fill_between(
            x, m - se, m + se,
            color=COL[cov_group],
            alpha=0.20,
            linewidth=0,
        )

    # veridical reference
    ax.plot(stim_numbers, stim_numbers, linestyle=":", linewidth=1, color="0.4")

    ax.set_xlabel("True numerosity (stim_number)")

    ax.set_xticks(stim_numbers)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_name = "figures/numerosity_quartiles_collapsed"
# plt.savefig(out_name + ".png", dpi=300, bbox_inches="tight")
# plt.savefig(out_name + ".svg", format="svg", bbox_inches="tight")
# plt.savefig(out_name + ".pdf", format="pdf", bbox_inches="tight")
plt.show()

print("Saved:", out_name)
print("Trials used after quartile filtering:", len(df_q))
print("Subjects contributing:", df_q["subject_id"].nunique())
