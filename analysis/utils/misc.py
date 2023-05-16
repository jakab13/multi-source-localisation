import pandas as pd
import os
import slab
import numpy as np


def load_dataframe(data_dir, exp_name="NumJudge", plane="h"):
    """
    Loads data from multiple files in a directory and returns a concatenated pandas DataFrame.

    Parameters:
    - data_dir (str): The directory path where the data files are located.
    - exp_name (str, optional): The name of the experiment. Default is "NumJudge".
    - plane (str, optional): The plane of the data. Default is "h".

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded data from the files.

    """

    dfs = list()  # List to store individual DataFrames from each file
    files = search_for_files(data_dir=data_dir, exp_name=exp_name, plane=plane)  # Search for files in the specified directory

    # Iterate over each file
    for file in files:
        data_dict = dict()  # Dictionary to store data items from the file
        data = slab.ResultsFile.read_file(file)  # Read data from the file using a slab library

        # Iterate over each item in the data
        for item in data:
            for k, v in item.items():
                if k not in data_dict.keys():
                    data_dict[k] = list()
                data_dict[k].append(v)

        # Create a DataFrame using the collected data
        df = pd.DataFrame(columns=data_dict.keys())  # Create an empty DataFrame with columns as data_dict keys
        for key in data_dict.keys():
            df[key] = pd.Series(data_dict[key])  # Add each key's data as a column in the DataFrame

        df = pd.DataFrame.from_dict(data_dict, orient="index")  # Transpose the DataFrame
        df = df.transpose()

        dfs.append(df)  # Append the DataFrame to the list

    # Concatenate all DataFrames into a single DataFrame
    return pd.concat(dfs, keys=_extract_subject_ids_from_files(files), names=["Sub_ID"])


def search_for_files(data_dir, exp_name, plane):
    """
    Searches for files in a directory that match the specified experiment name and plane.

    Parameters:
    - data_dir (str): The directory path where the files will be searched.
    - exp_name (str): The name of the experiment to match in the file names.
    - plane (str): The plane to match in the file names.

    Returns:
    - list: A list of file paths that match the given experiment name and plane.

    """

    filelist = list()  # List to store the file paths

    # Walk through the directory tree
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if the file name contains the experiment name and plane
            if exp_name in file and plane in file:
                filelist.append(os.path.join(root, file))  # Add the file path to the list

    return filelist


def _extract_subject_ids_from_files(files):
    """
    Extracts subject IDs from a list of file names.

    Parameters:
    - files (list): A list of file names from which to extract the subject IDs.

    Returns:
    - list: A list of subject IDs extracted from the file names.

    """

    subject_ids = []  # List to store the extracted subject IDs

    # Iterate over each file name
    for file in files:
        # Iterate from 0 to 99 (assuming subject IDs are in the range of sub_00 to sub_99)
        for i in range(100):
            if i < 10:
                # Check if the file name contains sub_0i (where i is a single-digit number)
                if f"sub_0{i}" in file:
                    subject_ids.append(f"sub_0{i}")  # Add the subject ID to the list
            if i >= 10:
                # Check if the file name contains sub_i (where i is a double-digit number)
                if f"sub_{i}" in file:
                    subject_ids.append(f"sub_{i}")  # Add the subject ID to the list

    return subject_ids


def extract_subject_ids_from_dataframe(dataframe):
    """
    Extracts unique subject IDs from the index values of a pandas DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The pandas DataFrame from which to extract the subject IDs.

    Returns:
    - list: A list of unique subject IDs extracted from the index values of the DataFrame.

    """

    subject_ids = np.unique([x[0] for x in dataframe.index.values]).tolist()
    # Extracts the subject IDs from the index values of the DataFrame

    return subject_ids


def get_azimuth_from_df(dataset):
    """
    Extracts azimuth values from a dataset and returns them as a list.

    Parameters:
    - dataset (iterable): The dataset from which to extract azimuth values.

    Returns:
    - list: A list of azimuth values extracted from the dataset.

    """

    azimuth = list()  # List to store the extracted azimuth values

    # Iterate over each element in the dataset
    for x in dataset:
        if x is None:
            azimuth.append(None)  # Append None if the element is None
        else:
            azimuth.append(x[0])  # Append the first element of x to the azimuth list

    return azimuth


def get_elevation_from_df(dataset):
    """
    Extracts elevation values from a dataset and returns them as a list.

    Parameters:
    - dataset (iterable): The dataset from which to extract elevation values.

    Returns:
    - list: A list of elevation values extracted from the dataset.

    """

    elevation = list()  # List to store the extracted elevation values

    # Iterate over each element in the dataset
    for x in dataset:
        if x is None:
            elevation.append(None)  # Append None if the element is None
        else:
            elevation.append(x[1])  # Append the second element of x to the elevation list

    return elevation


def replace_in_array(array, to_replace_val=None, replace_with_val=0):
    """
    Replaces values in an array with a specified replacement value.

    Parameters:
    - array (list or ndarray): The array in which values will be replaced.
    - to_replace_val (object, optional): The value to be replaced. Default is None.
    - replace_with_val (object, optional): The replacement value. Default is 0.

    Returns:
    - list or ndarray: An array with the specified values replaced.

    """

    for i, val in enumerate(array):
        if val == to_replace_val:
            array[i] = replace_with_val  # Replace the value at index i with replace_with_val

    return array


def crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=True,
             normalize=True):
    """
    Compute a cross-tabulation of two (or more) factors.

    Parameters:
    - index: array-like or list of arrays
        The values to group by on the rows.
    - columns: array-like or list of arrays
        The values to group by on the columns.
    - values: array-like, optional
        Array of values to aggregate according to the factors. If not specified, counts the occurrences of each factor combination.
    - rownames: sequence, default None
        The names to use for the row levels in the resulting DataFrame.
    - colnames: sequence, default None
        The names to use for the column levels in the resulting DataFrame.
    - aggfunc: function, optional
        If specified, aggregates the values according to the provided function.
    - margins: bool, default True
        Add row/column margins (subtotals).
    - normalize: bool, default True
        Normalize by dividing all values by the sum of values.

    Returns:
    - DataFrame
        A cross-tabulation DataFrame.

    """

    cm = pd.crosstab(index, columns, values=values, rownames=rownames, colnames=colnames, margins=margins,
                     normalize=normalize, aggfunc=aggfunc)
    # Compute a cross-tabulation using the provided parameters

    return cm



if __name__ == "__main__":
    from labplatform.config import get_config
    data_dir = os.path.join(get_config("DATA_ROOT"), "MSL")
    exp_name = "NumJudge"
    plane = "v"
    df = load_dataframe(data_dir, exp_name, plane)
