import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

PATH = "Dataframes/numerosity_judgement_spectral_coverage_otsu_2.csv"
df = pd.read_csv(PATH)

# -----------------------------
# 1) Clean + construct coverage_used
# -----------------------------
df["stim_type"] = df["stim_type"].astype(str).str.lower()
df["plane"] = df["plane"].astype(str).str.lower()
df = df.loc[df["stim_type"].isin(["forward", "reversed"])].copy()

df = df.dropna(
    subset=[
        "subject_id",
        "plane",
        "stim_number",
        "stim_type",
        "resp_number",
        "spectral_coverage",
        "spectral_coverage_2",
    ]
)

# Use spectral_coverage_2 for horizontal/vertical, spectral_coverage for distance
df["coverage_used"] = np.where(
    df["plane"].isin(["horizontal", "vertical"]),
    df["spectral_coverage_2"],
    df["spectral_coverage"],
)

df["stim_number_num"] = df["stim_number"].astype(int)
df["error"] = df["resp_number"] - df["stim_number_num"]

# Within-cell centring within subject × plane × stim_number × stim_type
cell_cols = ["subject_id", "plane", "stim_number_num", "stim_type"]
df["coverage_c"] = df["coverage_used"] - df.groupby(cell_cols)["coverage_used"].transform("mean")
df["coverage_c_0p10"] = df["coverage_c"] / 0.10  # per +10 percentage points

planes = sorted(df["plane"].unique())
types = sorted(df["stim_type"].unique())

# -----------------------------
# 2) Compute per-subject × plane × stim_type descriptors + sensitivity
# -----------------------------
min_trials_subject_plane_type = 60  # per subject per plane per stim_type
min_cell_trials = 8                 # per (subj, plane, stim_type, stim_number)

records = []

for pl in planes:
    for stype in types:
        dps = df.loc[(df["plane"] == pl) & (df["stim_type"] == stype)].copy()

        # ---- within-condition slopes per subject for this plane×type ----
        cell_slopes = []
        for (sid, sn), dc in dps.groupby(["subject_id", "stim_number_num"]):
            if len(dc) < min_cell_trials:
                continue
            m_cell = smf.ols("error ~ coverage_c_0p10", data=dc).fit()
            cell_slopes.append({"subject_id": sid, "stim_number": sn, "slope": float(m_cell.params["coverage_c_0p10"])})
        cell_slopes = pd.DataFrame(cell_slopes)

        sens = (
            cell_slopes.groupby("subject_id")["slope"]
            .median()
            .rename("sens")
            if len(cell_slopes) else pd.Series(dtype=float, name="sens")
        )

        # ---- per-subject descriptors for this plane×type ----
        for sid, ds in dps.groupby("subject_id"):
            if len(ds) < min_trials_subject_plane_type:
                continue

            curve = smf.ols("resp_number ~ stim_number_num", data=ds).fit()
            gain = float(curve.params["stim_number_num"])
            r2 = float(curve.rsquared)

            mae = float(np.mean(np.abs(ds["error"])))
            rmse = float(np.sqrt(np.mean(ds["error"] ** 2)))

            records.append({
                "subject_id": sid,
                "plane": pl,
                "stim_type": stype,
                "n_trials": len(ds),
                "gain": gain,
                "r2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "sens_median": float(sens.get(sid, np.nan)),
            })

subj = pd.DataFrame(records)
print("\n=== Long table (subject × plane × stim_type) ===")
print(subj.head())

# -----------------------------
# 3) Correlate within each plane × stim_type
# -----------------------------
def corr_report(d: pd.DataFrame, x: str, y: str, label: str):
    d = d[[x, y]].dropna()
    if len(d) < 8:
        print(f"{label}: n={len(d)} (skipping; too few subjects)")
        return
    rp, pp = pearsonr(d[x], d[y])
    rs, ps = spearmanr(d[x], d[y])
    print(f"{label} (n={len(d)}): Pearson r={rp:.3f}, p={pp:.3g}; Spearman rho={rs:.3f}, p={ps:.3g}")

for pl in planes:
    for stype in types:
        d = subj.loc[(subj["plane"] == pl) & (subj["stim_type"] == stype)].copy()
        print(f"\n--- plane={pl}, stim_type={stype} ---")
        corr_report(d, "gain", "sens_median", "gain vs sensitivity")
        corr_report(d, "MAE", "sens_median", "MAE vs sensitivity")
        corr_report(d, "RMSE", "sens_median", "RMSE vs sensitivity")
        corr_report(d, "r2", "sens_median", "R2 vs sensitivity")
