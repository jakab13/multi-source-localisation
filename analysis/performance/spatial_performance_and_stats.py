"""
UPDATED: Numerosity judgement uses LINEAR sensitivity (slope) instead of saturating asymptote.

What this script does
---------------------
1) Localisation accuracy (subject × plane × stim_type):
   - azimuth/elevation: linear slope + R² from resp_loc ~ stim_loc
   - distance: power-law exponent (b) + R² from resp_loc = a * stim_loc^b (fit in log-log space)

2) Spatial unmasking:
   - ΔTMR relative to collocated reference (per subject × plane × masker location)
   - summary per subject × plane: best benefit (min ΔTMR)

3) Numerosity judgement (subject × plane × stim_type):
   - linear slope + R² from resp_number ~ stim_number
   - (optional) MAE for numerosity (still computed; not used as primary metric)

Statistical tests (hierarchical / mixed models; subject random intercept):
A) Plane differences:
   - Localisation sensitivity (sens) ~ plane * stim_type + (1|subject)
   - Spatial unmasking ΔTMR ~ plane * masker_loc + (1|subject)
   - Numerosity sensitivity (num_slope) ~ plane * stim_type + (1|subject)

B) Do localisation/unmasking predict numerosity?
   - num_slope ~ plane*stim_type + z_loc_sens + z_unmask_benefit + (1|subject)

Outputs:
- metrics_localisation.csv
- metrics_unmasking_trial.csv
- metrics_unmasking_summary.csv
- metrics_numerosity_linear.csv
- model_summaries_linear_numerosity.txt
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import linregress
import statsmodels.formula.api as smf


# -------------------------
# CONFIG: input paths
# -------------------------
PATH_LOC = Path("Dataframes/localisation_accuracy_filtered_excl.csv")
PATH_NUM = Path("Dataframes/numerosity_judgement_filtered_excl.csv")
PATH_UNM = Path("Dataframes/spatial_unmasking_filtered_excl.csv")  # adjust if needed

OUT_DIR = PATH_LOC.parent / "performance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns found: {candidates}\nAvailable: {list(df.columns)}")
    return None


def ensure_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _lin_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, R². NaNs if insufficient."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or np.unique(x).size < 2:
        return np.nan, np.nan, np.nan
    lr = linregress(x, y)
    return float(lr.slope), float(lr.intercept), float(lr.rvalue**2)


def _powerlaw_exponent_r2(stim: np.ndarray, resp: np.ndarray) -> tuple[float, float]:
    """
    Fit resp = a * stim^b via log-log regression.
    Returns exponent b and R² in log space.
    """
    stim = np.asarray(stim, float)
    resp = np.asarray(resp, float)
    m = np.isfinite(stim) & np.isfinite(resp) & (stim > 0) & (resp > 0)
    if m.sum() < 2 or np.unique(stim[m]).size < 2:
        return np.nan, np.nan
    x = np.log(stim[m])
    y = np.log(resp[m])
    lr = linregress(x, y)
    return float(lr.slope), float(lr.rvalue**2)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (series - mu) / sd


# -------------------------
# 1) Localisation metrics (split by stim_type)
# -------------------------
def compute_localisation_metrics(df_loc: pd.DataFrame) -> pd.DataFrame:
    needed = {"subject_id", "plane", "stim_type", "stim_loc", "resp_loc"}
    missing = needed - set(df_loc.columns)
    if missing:
        raise ValueError(f"Localisation missing columns: {sorted(missing)}")

    rows = []
    for (sub, plane, stim_type), g in df_loc.groupby(["subject_id", "plane", "stim_type"], dropna=False):
        stim = g["stim_loc"].to_numpy()
        resp = g["resp_loc"].to_numpy()

        if str(plane).lower() == "distance":
            sens, r2 = _powerlaw_exponent_r2(stim, resp)   # exponent b
            sens_kind = "powerlaw_exponent"
            intercept = np.nan
        else:
            sens, intercept, r2 = _lin_slope_r2(stim, resp)
            sens_kind = "linear_slope"

        rows.append(
            dict(
                subject_id=sub,
                plane=plane,
                stim_type=stim_type,
                n_trials=int(len(g)),
                sens=sens,
                intercept=intercept,
                r2=r2,
                sens_kind=sens_kind,
            )
        )

    out = pd.DataFrame(rows)
    out = ensure_categorical(out, ["subject_id", "plane", "stim_type", "sens_kind"])
    return out


# -------------------------
# 2) Spatial unmasking metrics
# -------------------------
def compute_unmasking_metrics(df_unm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if "subject_id" not in df_unm.columns or "plane" not in df_unm.columns:
        raise ValueError("Unmasking file must contain at least: subject_id, plane")

    tmr_col = pick_col(df_unm, ["tmr", "TMR", "threshold", "tmr_threshold", "srt_tmr", "TMR_threshold"])
    masker_col = pick_col(df_unm, ["masker_speaker_loc_abs", "masker_location", "masker_pos", "masker_position", "masker"], required=True)

    d = df_unm.copy()

    def _is_collocated(x):
        if pd.isna(x):
            return False
        if isinstance(x, str):
            return x.strip().lower() in {"collocated", "co-located", "colocated", "co located", "0", "0.0", "7", "7.0"}
        try:
            return float(x) == 0.0
        except Exception:
            return False

    d["_is_ref"] = d[masker_col].apply(_is_collocated)

    ref = (
        d[d["_is_ref"]]
        .groupby(["subject_id", "plane"], dropna=False)[tmr_col]
        .mean()
        .rename("tmr_ref")
        .reset_index()
    )
    if ref.empty:
        raise ValueError(
            "No collocated reference rows found. Ensure masker location uses 0 or 'collocated', "
            "or modify _is_collocated()."
        )

    trial = d.merge(ref, on=["subject_id", "plane"], how="left")
    trial["delta_tmr"] = trial[tmr_col] - trial["tmr_ref"]

    summary = (
        trial.groupby(["subject_id", "plane"], dropna=False)["delta_tmr"]
        .min()
        .rename("unmask_best_delta_tmr")
        .reset_index()
    )

    trial = ensure_categorical(trial, ["subject_id", "plane"])
    summary = ensure_categorical(summary, ["subject_id", "plane"])
    return trial, summary, tmr_col, masker_col


# -------------------------
# 3) Numerosity linear metrics (slope)
# -------------------------
def compute_numerosity_linear_metrics(df_num: pd.DataFrame) -> pd.DataFrame:
    needed = {"subject_id", "plane", "stim_type", "stim_number", "resp_number"}
    missing = needed - set(df_num.columns)
    if missing:
        raise ValueError(f"Numerosity missing columns: {sorted(missing)}")

    rows = []
    for (sub, plane, stim_type), g in df_num.groupby(["subject_id", "plane", "stim_type"], dropna=False):
        x = g["stim_number"].to_numpy(float)
        y = g["resp_number"].to_numpy(float)

        slope, intercept, r2 = _lin_slope_r2(x, y)

        # Optional: MAE as a descriptive metric (still useful)
        mae = float(np.nanmean(np.abs(y - x))) if np.isfinite(x).any() and np.isfinite(y).any() else np.nan

        rows.append(
            dict(
                subject_id=sub,
                plane=plane,
                stim_type=stim_type,
                n_trials=int(len(g)),
                num_slope=slope,
                num_intercept=intercept,
                num_r2=r2,
                num_mae=mae,
            )
        )

    out = pd.DataFrame(rows)
    out = ensure_categorical(out, ["subject_id", "plane", "stim_type"])
    return out


# -------------------------
# Mixed model helper
# -------------------------
def fit_mixedlm(formula: str, data: pd.DataFrame, group_col: str = "subject_id"):
    # Statsmodels MixedLM requires complete cases for variables in formula
    # We'll let patsy handle parsing; just drop NA rows across all columns used.
    # Quick conservative approach: drop rows with ANY NA in data columns referenced.
    # (If you want surgical NA dropping, we can add a proper parser.)
    d = data.copy()
    res = smf.mixedlm(formula, d, groups=d[group_col], missing="drop").fit(reml=True, method="lbfgs")
    return res


# -------------------------
# Main
# -------------------------
def main():
    df_loc = pd.read_csv(PATH_LOC)
    df_num = pd.read_csv(PATH_NUM)
    df_unm = pd.read_csv(PATH_UNM)

    # --- Compute metrics
    loc_metrics = compute_localisation_metrics(df_loc)
    # loc_metrics.to_csv(OUT_DIR / "metrics_localisation.csv", index=False)

    unm_trial, unm_summary, tmr_col, masker_col = compute_unmasking_metrics(df_unm)
    # unm_trial.to_csv(OUT_DIR / "metrics_unmasking_trial.csv", index=False)
    # unm_summary.to_csv(OUT_DIR / "metrics_unmasking_summary.csv", index=False)

    num_metrics = compute_numerosity_linear_metrics(df_num)
    # num_metrics.to_csv(OUT_DIR / "metrics_numerosity_linear.csv", index=False)


    print("Done. Outputs:")
    print(f"  {OUT_DIR / 'metrics_localisation.csv'}")
    print(f"  {OUT_DIR / 'metrics_unmasking_trial.csv'}")
    print(f"  {OUT_DIR / 'metrics_unmasking_summary.csv'}")
    print(f"  {OUT_DIR / 'metrics_numerosity_linear.csv'}")

if __name__ == "__main__":
    main()
