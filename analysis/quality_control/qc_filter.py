#!/usr/bin/env python3
"""
Quality control / outlier detection for MSL project datasets:
- localisation_accuracy.csv
- spatial_unmasking.csv
- numerosity_judgement.csv

Outputs (by default into ./qc_out):
- qc_metrics_long.csv : long-format metrics per subject x plane x paradigm
- qc_summary_wide.csv : wide-format subject-level metrics
- qc_flags.csv        : outlier flags with reasons
- excluded_subjects.csv : suggested exclusions with paradigm-specific reasons

Outlier logic:
- Robust z-score using MAD (median absolute deviation). Default threshold: |z| >= 3.5
- IQR rule (optional) for additional flagging: outside [Q1 - k*IQR, Q3 + k*IQR], k=3.0
- For numerosity: dedicated criterion on stim_number==2 performance (configurable)

This script is intentionally conservative: it flags and explains; you decide what to exclude.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def robust_z(x: pd.Series) -> pd.Series:
    """Robust z-score using MAD. Returns NaN if MAD==0 or insufficient data."""
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series([np.nan] * len(x), index=x.index)
    return 0.6745 * (x - med) / mad


def iqr_bounds(x: pd.Series, k: float = 3.0) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 4:
        return (np.nan, np.nan)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def add_outlier_flags(df: pd.DataFrame, value_col: str, group_cols: list[str],
                      z_thresh: float = 3.5, use_iqr: bool = True, iqr_k: float = 3.0) -> pd.DataFrame:
    """
    Adds robust z and outlier flags within each group (e.g., plane).
    Expects df to be aggregated to one row per subject within group.
    """
    df = df.copy()
    df["robust_z"] = np.nan
    df["outlier_z"] = False
    df["outlier_iqr"] = False

    for _, g in df.groupby(group_cols, dropna=False):
        idx = g.index
        z = robust_z(g[value_col])
        df.loc[idx, "robust_z"] = z
        df.loc[idx, "outlier_z"] = z.abs() >= z_thresh

        if use_iqr:
            lo, hi = iqr_bounds(g[value_col], k=iqr_k)
            if np.isfinite(lo) and np.isfinite(hi):
                df.loc[idx, "outlier_iqr"] = (g[value_col] < lo) | (g[value_col] > hi)

    df["is_outlier"] = df["outlier_z"] | df["outlier_iqr"]
    return df


def qc_localisation(path: Path) -> pd.DataFrame:
    """Compute subject-level localisation metrics per plane."""
    df = pd.read_csv(path)
    # Required columns
    req = {"subject_id", "plane", "stim_type", "abs_error_loc", "error_loc"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Localisation file missing columns: {missing}")

    # Aggregate
    agg = (df.groupby(["subject_id", "plane"], as_index=False)
             .agg(
                 loc_mae=("abs_error_loc", "mean"),
                 loc_median_abs_err=("abs_error_loc", "median"),
                 loc_bias=("error_loc", "mean"),
                 loc_sd_abs_err=("abs_error_loc", "std"),
                 loc_n=("abs_error_loc", "count"),
             ))
    agg["paradigm"] = "localisation"
    return agg


def qc_spatial_unmasking(path: Path) -> pd.DataFrame:
    """Compute subject-level spatial unmasking metrics per plane."""
    df = pd.read_csv(path)
    req = {"subject_id", "plane", "threshold", "normed_threshold"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Spatial unmasking file missing columns: {missing}")

    agg = (df.groupby(["subject_id", "plane"], as_index=False)
             .agg(
                 su_threshold_mean=("threshold", "mean"),
                 su_threshold_median=("threshold", "median"),
                 su_threshold_sd=("threshold", "std"),
                 su_normed_mean=("normed_threshold", "mean"),
                 su_normed_median=("normed_threshold", "median"),
                 su_normed_sd=("normed_threshold", "std"),
                 su_n=("threshold", "count"),
             ))
    agg["paradigm"] = "spatial_unmasking"
    return agg


def qc_numerosity(path: Path, stim_check_n: int = 2,
                 mean_overest_thresh: float = 0.5,
                 overest_rate_thresh: float = 0.6) -> pd.DataFrame:
    """
    Compute subject-level numerosity metrics per plane.

    Dedicated check: for stim_number==stim_check_n, flag if
      mean(resp_number - stim_check_n) > mean_overest_thresh  (i.e., strong overestimation),
      OR overestimation rate (resp > stim_check_n) > overest_rate_thresh.
    """
    df = pd.read_csv(path)
    req = {"subject_id", "plane", "stim_number", "resp_number"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Numerosity file missing columns: {missing}")

    df["stim_number"] = pd.to_numeric(df["stim_number"], errors="coerce")
    df["resp_number"] = pd.to_numeric(df["resp_number"], errors="coerce")
    df["num_error"] = df["resp_number"] - df["stim_number"]

    # Overall metrics per plane
    overall = (df.groupby(["subject_id", "plane"], as_index=False)
                 .agg(
                     num_mae=("num_error", lambda x: np.nanmean(np.abs(x))),
                     num_bias=("num_error", "mean"),
                     num_sd=("num_error", "std"),
                     num_n=("num_error", "count"),
                 ))

    # Dedicated 2-talker check per plane
    d2 = df[df["stim_number"] == stim_check_n].copy()
    if len(d2) == 0:
        # Still return overall metrics
        overall["num_check_n_mean_overest"] = np.nan
        overall["num_check_n_overest_rate"] = np.nan
        overall["num_check_n_n"] = 0
        overall["num_check_n_flag"] = False
    else:
        d2["over"] = d2["resp_number"] > stim_check_n
        d2["overest"] = d2["resp_number"] - stim_check_n

        d2agg = (d2.groupby(["subject_id", "plane"], as_index=False)
                   .agg(
                       num_check_n_mean_overest=("overest", "mean"),
                       num_check_n_median_overest=("overest", "median"),
                       num_check_n_overest_rate=("over", "mean"),
                       num_check_n_n=("overest", "count"),
                   ))
        overall = overall.merge(d2agg, on=["subject_id", "plane"], how="left")
        overall["num_check_n_flag"] = (
            (overall["num_check_n_mean_overest"] > mean_overest_thresh) |
            (overall["num_check_n_overest_rate"] > overest_rate_thresh)
        ).fillna(False)

    overall["paradigm"] = "numerosity"
    return overall


def to_long_metrics(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Convert wide aggregated metrics to long format (metric, value)."""
    value_cols = [c for c in df.columns if c not in set(id_cols)]
    out = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
    return out


def main():
    # loc = pd.read_csv("DataFrames/localisation_accuracy.csv")
    # su = pd.read_csv("DataFrames/spatial_unmasking.csv")
    # num = pd.read_csv("DataFrames/numerosity_judgement.csv")

    loc = qc_localisation("DataFrames/localisation_accuracy.csv")
    su = qc_spatial_unmasking("DataFrames/spatial_unmasking.csv")
    num = qc_numerosity("DataFrames/numerosity_judgement.csv", stim_check_n=2,
                        mean_overest_thresh=0.5,
                        overest_rate_thresh=0.6)

    # Outlier flags per paradigm/metric: we mainly care about:
    # localisation: loc_mae
    # spatial unmasking: su_threshold_mean and su_normed_mean
    # numerosity: num_check_n_flag (rule-based) and optionally num_mae
    flags = []

    def flag_metric(subdf: pd.DataFrame, value_col: str, metric_name: str, paradigm: str):
        d = subdf[["subject_id", "plane", value_col]].dropna().rename(columns={value_col: "value"}).copy()
        d["metric"] = metric_name
        d["paradigm"] = paradigm
        d = add_outlier_flags(d, "value", group_cols=["plane"], z_thresh=3.5,
                              use_iqr=True, iqr_k=3)
        flags.append(d)

    # localisation
    flag_metric(loc, "loc_mae", "loc_mae", "localisation")

    # spatial unmasking: threshold and normed threshold
    flag_metric(su, "su_threshold_mean", "su_threshold_mean", "spatial_unmasking")
    flag_metric(su, "su_normed_mean", "su_normed_mean", "spatial_unmasking")

    # numerosity: optional generic metric flag on num_mae + rule-based 2-talker check
    flag_metric(num, "num_mae", "num_mae", "numerosity")
    # rule-based flag
    rule = num[["subject_id", "plane", "num_check_n_flag",
                "num_check_n_mean_overest", "num_check_n_overest_rate", "num_check_n_n"]].copy()
    rule["paradigm"] = "numerosity"
    rule["metric"] = f"stim{2}_overest_rule"
    rule["value"] = rule["num_check_n_flag"].astype(int)
    rule["robust_z"] = np.nan
    rule["outlier_z"] = False
    rule["outlier_iqr"] = False
    rule["is_outlier"] = rule["num_check_n_flag"].fillna(False)

    rule = rule.drop(columns=["num_check_n_flag"])
    flags.append(rule)

    flags_df = pd.concat(flags, ignore_index=True)
    flags_df.to_csv("analysis/quality_control/qc_flags.csv", index=False)

    # Suggested exclusions: any subject flagged on any plane/metric, with reasons aggregated
    flagged = flags_df[flags_df["is_outlier"] == True].copy()
    if len(flagged) == 0:
        excl = pd.DataFrame(columns=["subject_id", "plane", "reasons"])
    else:
        flagged["reason"] = flagged.apply(
            lambda r: f"{r['paradigm']}:{r['metric']} (value={r['value']:.3g})", axis=1
        )
        excl = (flagged.groupby(["subject_id", "plane"], as_index=False)
                      .agg(reasons=("reason", lambda x: "; ".join(sorted(set(x))))))
    excl.to_csv("analysis/quality_control/excluded_subjects.csv", index=False)

    print("QC finished.")
    print(f"Outputs written to: analysis/quality_control")


if __name__ == "__main__":
    main()
