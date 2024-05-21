import pandas as pd
import slab
import pathlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np

pd.options.mode.chained_assignment = None
sns.set_theme()
results_folder = pathlib.Path(os.getcwd()) / "Results"

subjects_excl = [""]

subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
subjects = sorted([s for s in subjects if not any(s in excl for excl in subjects_excl)])

results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

columns_la = ["subject_id", "plane", "stim_type", "offset_azi", "offset_ele", "stim_azi", "stim_ele", "resp_azi",
              "resp_ele"]
columns_nj = ["subject_id", "plane", "stim_type", "stim_number", "stim_country_ids", "stim_talker_ids",
              "speaker_ids", "resp_number", "reaction_time"]

df_la = pd.DataFrame(columns=columns_la)
df_nj = pd.DataFrame(columns=columns_nj)

for subject_id, results_file_list in results_files.items():
    for results_file_name in results_file_list:
        path = results_folder / subject_id / results_file_name
        if "LocaAccu" in results_file_name:
            plane = slab.ResultsFile.read_file(path, tag="plane")
            stim_type = slab.ResultsFile.read_file(path, tag="mode")
            offset_azi, offset_ele = slab.ResultsFile.read_file(path, tag="offset")
            stim_loc = np.asarray(slab.ResultsFile.read_file(path, tag="actual"), dtype=object)
            stim_loc = np.asarray([e if e is not None else [0., 0.] for e in stim_loc], dtype=object)
            stim_azi = stim_loc[:, 0]
            stim_ele = stim_loc[:, 1]
            resp = np.asarray(slab.ResultsFile.read_file(path, tag="perceived"), dtype=object)
            resp = np.asarray([e if e is not None else [0., 0.] for e in resp], dtype=object)
            resp_azi = resp[:, 0]
            resp_ele = resp[:, 1]
            reaction_time = slab.ResultsFile.read_file(path, tag="rt")
            block_length = len(reaction_time)

            df_curr = pd.DataFrame(index=range(0, block_length))
            df_curr["subject_id"] = subject_id
            df_curr["plane"] = plane
            df_curr["stim_type"] = stim_type
            df_curr["offset_azi"] = offset_azi
            df_curr["offset_ele"] = offset_ele
            df_curr["stim_azi"] = stim_azi.astype(float)
            df_curr["stim_ele"] = stim_ele.astype(float)
            df_curr["resp_azi"] = resp_azi.astype(float)
            df_curr["resp_ele"] = resp_ele.astype(float)
            df_la = pd.concat([df_la, df_curr], ignore_index=True)
        elif "NumJudge" in results_file_name:
            plane = slab.ResultsFile.read_file(path, tag="plane")
            stim_number = slab.ResultsFile.read_file(path, tag="solution")
            stim_type = "reversed" if slab.ResultsFile.read_file(path, tag="reversed_speech") else "forward"
            stim_country_ids = slab.ResultsFile.read_file(path, tag="country_idxs")
            stim_talker_ids = slab.ResultsFile.read_file(path, tag="signals_sample")
            speaker_ids = slab.ResultsFile.read_file(path, tag="speakers_sample")
            resp_number = slab.ResultsFile.read_file(path, tag="response")
            reaction_time = slab.ResultsFile.read_file(path, tag="rt")
            block_length = len(reaction_time)

            df_curr = pd.DataFrame(index=range(0, block_length))
            df_curr["subject_id"] = subject_id
            df_curr["plane"] = plane
            df_curr["stim_number"] = np.asarray(stim_number, dtype=float)
            df_curr["stim_type"] = stim_type
            df_curr["stim_country_ids"] = stim_country_ids
            df_curr["stim_talker_ids"] = stim_talker_ids
            df_curr["speaker_ids"] = speaker_ids
            df_curr["resp_number"] = np.asarray(resp_number, dtype=float)
            df_curr["reaction_time"] = reaction_time
            df_nj = pd.concat([df_nj, df_curr], ignore_index=True)

columns_performance = ["subject_id", "plane", "stim_type", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
df_performance_nj = pd.DataFrame(columns=columns_performance)

for subject_id in df_nj.subject_id.unique():
    if subject_id != "sub_01" and subject_id != "sub_02" and subject_id != "sub_09":
        for plane in df_nj.plane.unique():
            for stim_type in df_nj.stim_type.unique():
                df_curr = df_nj[(df_nj.subject_id == subject_id) & (df_nj.plane == plane) & (df_nj.stim_type == stim_type)]
                if len(df_curr) > 0:
                    x = df_curr.stim_number.values.astype(float)
                    y = df_curr.resp_number.values.astype(float)
                    reg = scipy.stats.linregress(x, y)
                    df_performance_nj.loc[len(df_performance_nj)] = [subject_id, plane, stim_type, reg.slope, reg.intercept, reg.rvalue, reg.pvalue, reg.stderr, reg.intercept_stderr]

columns_performance = ["subject_id", "plane", "stim_type", "slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
df_performance_la = pd.DataFrame(columns=columns_performance)
for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        for stim_type in df_la.stim_type.unique():
            df_curr = df_la[(df_la.subject_id == subject_id) & (df_la.plane == plane) & (df_la.stim_type == stim_type)]
            if len(df_curr) > 0:
                if plane == "v":
                    x = df_curr.stim_ele.values.astype(float)
                    y = df_curr.resp_ele.values.astype(float)
                elif plane == "h":
                    x = df_curr.stim_azi.values.astype(float)
                    y = df_curr.resp_azi.values.astype(float)
                reg = scipy.stats.linregress(x, y)
                if reg.slope < 0 or reg.pvalue > 0.05:
                    continue
                else:
                    df_performance_la.loc[len(df_performance_la)] = [subject_id, plane, stim_type, reg.slope, reg.intercept, reg.rvalue, reg.pvalue, reg.stderr, reg.intercept_stderr]

columns_slopes = ["subject_id", "plane", "stim_type_la", "stim_type_nj", "la_slope", "nj_slope"]
df_slopes = pd.DataFrame(columns=columns_slopes)
for subject_id in df_la.subject_id.unique():
    for plane in df_la.plane.unique():
        for stim_type_la in df_la.stim_type.unique():
            df_curr_la = df_performance_la[(df_performance_la.subject_id == subject_id) & (df_performance_la.plane == plane) & (df_performance_la.stim_type == stim_type_la)]
            for stim_type_nj in df_nj.stim_type.unique():
                df_curr_nj = df_performance_nj[(df_performance_nj.subject_id == subject_id) & (df_performance_nj.plane == plane) & (df_performance_nj.stim_type == stim_type_nj)]
                if len(df_curr_la) > 0 and len(df_curr_nj) > 0:
                    la_slope = df_curr_la.slope.iloc[0]
                    nj_slope = df_curr_nj.slope.iloc[0]
                    df_slopes.loc[len(df_slopes)] = [subject_id, plane, stim_type_la, stim_type_nj, la_slope, nj_slope]
