import pandas as pd
from pathlib import Path

# --- Subjects to exclude ---
excluded = {"sub_116", "sub_119", "sub_120", "sub_106"}

# --- Input files (adjust these paths to your setup) ---
in_files = {
    "numerosity_judgement": Path("Dataframes/numjudge_post_processed.csv"),
    "localisation_accuracy": Path("Dataframes/locaaccu_post_processed.csv"),
    "spatial_unmasking": Path("Dataframes/spatmask_post_processed.csv"),
}

# --- Output directory ---
out_dir = Path("Dataframes")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Filter + write ---
summary_rows = []

for name, path in in_files.items():
    df = pd.read_csv(path)
    n0 = len(df)

    if "subject_id" not in df.columns:
        raise ValueError(
            f"{name} has no 'subject_id' column. Found columns: {df.columns.tolist()}"
        )

    df_f = df[~df["subject_id"].isin(excluded)].copy()
    df_f = df_f[df_f["round"] == 2]
    n1 = len(df_f)

    out_path = out_dir / f"{name}_filtered_excl.csv"
    df_f.to_csv(out_path, index=False)

    summary_rows.append(
        {
            "table": name,
            "input_rows": n0,
            "output_rows": n1,
            "rows_removed": n0 - n1,
            "unique_subjects_before": df["subject_id"].nunique(),
            "unique_subjects_after": df_f["subject_id"].nunique(),
            "subjects_removed_present_in_table": sorted(set(df["subject_id"].unique()) & excluded),
            "output_file": str(out_path),
        }
    )

# --- Print a quick summary to the console ---
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
