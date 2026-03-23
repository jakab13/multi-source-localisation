"""
Localization accuracy metrics (per subject × plane × stimulus type)

Computes:
- MAE (mean absolute error) using abs_error_loc if present, otherwise |resp_loc - stim_loc|
- bias (mean signed error)
- linear regression: slope, intercept, r, R², p, stderr  (resp_loc ~ stim_loc)
- z-scores across participants *within each plane × stim_type* for MAE, slope, R²

Input:
    localisation_accuracy_filtered_excl.csv

Output:
    localisation_accuracy_metrics_by_sub_plane_stim.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress


# -----------------------
# CONFIG
# -----------------------
INPUT_CSV = Path("Dataframes/localisation_accuracy_filtered_excl.csv")  # <-- change to your path
OUTPUT_DIR = INPUT_CSV.parent  # <-- change if you want
OUT_CSV = OUTPUT_DIR / "localisation_accuracy_performance_by_sub_plane_stim.csv"


# -----------------------
# HELPERS
# -----------------------
def _safe_linregress(x: np.ndarray, y: np.ndarray) -> dict:
    """Run linregress robustly; return NaNs if not enough information."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Need at least 2 points and at least 2 unique x values to estimate slope.
    if x.size < 2 or np.unique(x).size < 2:
        return dict(slope=np.nan, intercept=np.nan, r=np.nan, r2=np.nan, p=np.nan, stderr=np.nan)

    res = linregress(x, y)
    return dict(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r=float(res.rvalue),
        r2=float(res.rvalue**2),
        p=float(res.pvalue),
        stderr=float(res.stderr),
    )


def _zscore_within_group(df: pd.DataFrame, group_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    """z-score value_col within each group in group_cols."""
    def z(g: pd.Series) -> pd.Series:
        mu = g.mean(skipna=True)
        sd = g.std(skipna=True, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series([np.nan] * len(g), index=g.index)
        return (g - mu) / sd

    df[out_col] = df.groupby(group_cols, dropna=False)[value_col].transform(z)
    return df


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    required = {"subject_id", "plane", "stim_type", "stim_loc", "resp_loc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Use provided error columns if present; otherwise compute them
    if "error_loc" not in df.columns:
        df["error_loc"] = df["resp_loc"] - df["stim_loc"]
    if "abs_error_loc" not in df.columns:
        df["abs_error_loc"] = df["error_loc"].abs()

    group_cols = ["subject_id", "plane", "stim_type"]

    rows = []
    for (sub, plane, stim), g in df.groupby(group_cols, dropna=False):
        x = g["stim_loc"].to_numpy()
        y = g["resp_loc"].to_numpy()

        reg = _safe_linregress(x, y)

        row = dict(
            subject_id=sub,
            plane=plane,
            stim_type=stim,
            n_trials=int(len(g)),
            n_unique_stim=int(pd.Series(x).nunique(dropna=True)),
            mae=float(np.nanmean(g["abs_error_loc"].to_numpy())),
            bias=float(np.nanmean(g["error_loc"].to_numpy())),  # signed error
            slope=reg["slope"],
            intercept=reg["intercept"],
            r=reg["r"],
            r2=reg["r2"],
            p=reg["p"],
            stderr=reg["stderr"],
        )
        rows.append(row)

    out = pd.DataFrame(rows)

    # z-scores across subjects within each plane × stim_type
    z_group = ["plane", "stim_type"]
    out = _zscore_within_group(out, z_group, "mae", "z_mae")
    out = _zscore_within_group(out, z_group, "slope", "z_slope")
    out = _zscore_within_group(out, z_group, "r2", "z_r2")

    # Nice ordering
    col_order = [
        "subject_id", "plane", "stim_type",
        "n_trials", "n_unique_stim",
        "mae", "bias",
        "slope", "intercept", "r", "r2", "p", "stderr",
        "z_mae", "z_slope", "z_r2",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
