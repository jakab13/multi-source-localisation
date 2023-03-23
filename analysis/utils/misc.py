import pandas as pd
import os
import slab
import numpy as np


def load_dataframe(data_dir, exp_name="NumJudge", plane="h"):
    dfs = list()
    files = search_for_files(data_dir=data_dir, exp_name=exp_name, plane=plane)
    for file in files:
        data_dict = dict()
        data = slab.ResultsFile.read_file(file)
        for item in data:
            for k, v in item.items():
                if k not in data_dict.keys():
                    data_dict[k] = list()
                data_dict[k].append(v)
        df = pd.DataFrame(columns=data_dict.keys())
        for key in data_dict.keys():
            df[key] = pd.Series(data_dict[key])
        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df = df.transpose()
        dfs.append(df)
    return pd.concat(dfs, keys=_extract_subject_ids_from_files(files),
                     names=["Sub_ID"])


def search_for_files(data_dir, exp_name, plane):
    filelist = list()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if exp_name in file and plane in file:
                filelist.append(os.path.join(root, file))
    return filelist


def _extract_subject_ids_from_files(files):
    subject_ids = []
    for file in files:
        for i in range(100):
            if i < 10:
                if f"sub_0{i}" in file:
                    subject_ids.append(f"sub_0{i}")
            if i >= 10:
                if f"sub_{i}" in file:
                    subject_ids.append(f"sub_{i}")
    return subject_ids


def extract_subject_ids_from_dataframe(dataframe):
    return np.unique([x[0] for x in dataframe.index.values]).tolist()


def get_azimuth_from_df(dataset):
    azimuth = list()
    for x in dataset:
        if x is None:
            azimuth.append(None)
        else:
            azimuth.append(x[0])
    return azimuth


def get_elevation_from_df(dataset):
    elevation = list()
    for x in dataset:
        if x is None:
            elevation.append(None)
        else:
            elevation.append(x[1])
    return elevation


def replace_in_array(x, to_replace_val=None, replace_with_val=0):
    for i, val in enumerate(x):
        if val == to_replace_val:
            x[i] = replace_with_val
    return x


def crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=True,
             normalize=True):
    cm = pd.crosstab(index, columns, values=values, rownames=rownames, colnames=colnames, margins=margins,
                     normalize=normalize, aggfunc=aggfunc)
    return cm


if __name__ == "__main__":
    from labplatform.config import get_config
    data_dir = os.path.join(get_config("DATA_ROOT"), "MSL")
    exp_name = "NumJudge"
    plane = "v"
    df = load_dataframe(data_dir, exp_name, plane)
