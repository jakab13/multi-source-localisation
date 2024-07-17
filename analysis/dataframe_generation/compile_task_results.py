import pandas as pd
import slab
import pathlib
import os
import seaborn as sns
import datetime as dt
import numpy as np
from analysis.dataframe_generation.utils import creation_date

pd.options.mode.chained_assignment = None
sns.set_theme()
results_folder = pathlib.Path(os.getcwd()) / "Results"

subjects_excl = [
    # "sub_101",
]

subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
subjects = sorted([s for s in subjects if not any(s in excl for excl in subjects_excl)])

results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

columns_la = ["subject_id", "round", "datetime_c", "plane", "stim_type", "offset_azi", "offset_ele", "stim_loc", "resp_loc"]

columns_su = ["subject_id", "round", "datetime_c", "plane", "masker_speaker_id", "masker_speaker_loc"]

columns_nj = ["subject_id", "round", "datetime_c", "plane", "stim_type", "stim_number", "stim_country_ids", "stim_talker_ids",
              "speaker_ids", "resp_number", "reaction_time"]

df_la = pd.DataFrame(columns=columns_la)
df_su = pd.DataFrame(columns=columns_su)
df_nj = pd.DataFrame(columns=columns_nj)

country_converter = {
    0: "Belgium",
    1: "Britain",
    2: "Congo",
    3: "Cuba",
    4: "Japan",
    5: "Mali",
    6: "Oman",
    7: "Peru",
    8: "Sudan",
    9: "Syria",
    10: "Togo",
    11: "Tonga",
    12: "Yemen",
}

for subject_id, results_file_list in results_files.items():
    subject_idx = int(subject_id[4:])
    round = 1 if subject_idx < 100 else 2
    for results_file_name in results_file_list:
        path = results_folder / subject_id / results_file_name
        if results_file_name.endswith(".txt"):
            date_string = results_file_name[-23:-4]
            datetime = dt.datetime.strptime(date_string, "%Y-%m-%d-%H-%M-%S")
            datetime = pd.to_datetime(datetime)
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
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = plane
                df_curr["stim_type"] = stim_type
                df_curr["offset_azi"] = offset_azi
                df_curr["offset_ele"] = offset_ele
                if plane == "h":
                    df_curr["stim_loc"] = stim_azi.astype(float)
                    df_curr["resp_loc"] = resp_azi.astype(float)
                elif plane == "v":
                    df_curr["stim_loc"] = stim_ele.astype(float)
                    df_curr["resp_loc"] = resp_ele.astype(float)
                df_la = pd.concat([df_la, df_curr], ignore_index=True)
            elif "SpatMask" in results_file_name:
                plane = slab.ResultsFile.read_file(path, tag="plane")
                masker_speaker_ids = slab.ResultsFile.read_file(path, tag="masker_speaker_id")
                masker_speaker_ids = list(dict.fromkeys(masker_speaker_ids)) if type(masker_speaker_ids) is list else masker_speaker_ids
                stairs_json = slab.ResultsFile.read_file(path, tag="stairs")
                stairs = list()
                thresholds = list()
                if len(stairs_json) > 0 and type(stairs_json) is list:
                    for stair_json in stairs_json:
                        stair = slab.Staircase(1)
                        stair.__dict__ = stair_json
                        stairs.append(stair)
                    thresholds = [stair.threshold() for stair in stairs]
                block_length = len(thresholds)

                df_curr = pd.DataFrame(index=range(0, block_length))
                df_curr["subject_id"] = subject_id
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = plane
                df_curr["masker_speaker_id"] = masker_speaker_ids[:len(thresholds)]
                df_curr["threshold"] = thresholds
                df_su = pd.concat([df_su, df_curr], ignore_index=True)
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
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = plane
                df_curr["stim_number"] = np.asarray(stim_number, dtype=float)
                df_curr["stim_type"] = stim_type
                df_curr["stim_country_ids"] = stim_country_ids
                df_curr["stim_talker_ids"] = stim_talker_ids
                df_curr["speaker_ids"] = speaker_ids
                df_curr["resp_number"] = np.asarray(resp_number, dtype=float)
                df_curr["reaction_time"] = reaction_time
                df_nj = pd.concat([df_nj, df_curr], ignore_index=True)
        elif results_file_name.endswith(".csv"):
            datetime = creation_date(path)
            datetime = pd.to_datetime(datetime, unit="s")
            if "localisation_accuracy" in results_file_name:
                df = pd.read_csv(path)
                df_curr = pd.DataFrame(index=range(0, len(df)))
                df_curr["subject_id"] = df["subject_id"]
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = df["plane"]
                df_curr["stim_type"] = df["stim_type"]
                df_curr["stim_loc"] = df["stim_dist"]
                df_curr["resp_loc"] = df["resp_dist"]
                df_curr["stim_type"].replace("pinknoise", "noise", inplace=True)
                df_la = pd.concat([df_la, df_curr], ignore_index=True)
            elif "results_spacial_unmasking" in results_file_name:
                df = pd.read_csv(path)
                df.rename(columns={'subject': 'subject_id'}, inplace=True)
                df_curr = pd.DataFrame(index=range(0, len(df)))
                df_curr["subject_id"] = df["subject_id"]
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = "distance"
                df_curr["masker_speaker_loc"] = df["distance_masker"]
                df_curr["threshold"] = df["threshold"]
                df_su = pd.concat([df_su, df_curr], ignore_index=True)
            elif "numerosity_judgement" in results_file_name:
                df = pd.read_csv(path)
                df_curr = pd.DataFrame(index=range(0, len(df)))
                df_curr["subject_id"] = df["subject_id"]
                df_curr["round"] = round
                df_curr["datetime_c"] = datetime
                df_curr["plane"] = df["plane"]
                df_curr["stim_number"] = df["stim_number"]
                df_curr["stim_type"] = df["stim_type"]
                df_curr["stim_country_ids"] = df["stim_country_ids"]
                df_curr["stim_talker_ids"] = df["stim_talker_ids"]
                df_curr["speaker_ids"] = df["speaker_ids"]
                df_curr["resp_number"] = df["resp_number"]
                df_curr["reaction_time"] = df["reaction_time"]
                df_curr["stim_type"].replace("countries_forward", "forward", inplace=True)
                df_curr["stim_type"].replace("countries_reversed", "reversed", inplace=True)
                df_nj = pd.concat([df_nj, df_curr], ignore_index=True)

dfs = [df_la, df_su, df_nj]

for df in dfs:
    df.loc[(df.plane == "h"), "plane"] = "horizontal"
    df.loc[(df.plane == "v"), "plane"] = "vertical"
    df.loc[(df.plane == "d"), "plane"] = "distance"

