import os
import seaborn as sns
import numpy as np
import pandas as pd
import slab
sns.set_theme()


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


if __name__ == "__main__":
    """
    Entry point of the script for data processing and visualization.

    The script performs the following steps:
    1. Load a dataframe from a specified file path.
    2. Search for files in a directory that match the specified experiment name and plane.
    3. Read data from each file using the slab library and store the relevant information in dictionaries.
    4. Convert the dictionaries into a Pandas DataFrame.
    5. Extract specific data from the DataFrame for further analysis and visualization.
    6. Plot a line graph using seaborn to visualize the relationship between response and correctness.

    Note:
    - This code assumes the availability of the slab library, numpy, pandas, and seaborn.
    - Modify the values of fp, exp_name, and plane to suit your specific data and file paths.

    """

    # load dataframe
    fp = "/home/max/labplatform/data/distance_oskar"
    exp_name = ""
    plane = ""

    # search for files
    files = search_for_files(data_dir=fp, exp_name=exp_name, plane=plane)

    subdata = dict()
    dfs = list()
    sub_ids = list()

    for file in files:
        resfile = slab.ResultsFile.read_file(file)  # Read data from the file using slab library
        date = list(resfile[0].keys())[0]
        sub = resfile[0][date]
        sub_ids.append(sub)
        distances = resfile[1]["distances"]
        nametag = list(resfile[2].keys())[0]
        datapool = resfile[2][nametag]
        trials = datapool["trials"]
        data = datapool["data"]
        subdata[sub] = dict()
        # put retrieved data into dictionary
        subdata[sub]["distances"] = distances
        subdata[sub]["trials"] = trials
        subdata[sub]["data"] = list()
        for trialdata in data:
            subdata[sub]["data"].append(trialdata[0])

    sub_ids = np.unique(sub_ids)
    df = pd.DataFrame.from_dict(data=subdata, orient="index")

    responses = list()
    iscorrect = list()
    diff_dist = list()
    dist1 = list()
    dist2 = list()
    for sub_id in sub_ids:
        for data in df.data[sub_id]:
            responses.append(data["response"])
            iscorrect.append(data["correct"])
            dist1.append(data["distance_1"])
            dist2.append(data["distance_2"])
            diff_dist.append(data["distance_2"] - data["distance_1"])

    sns.lineplot(x=iscorrect, y=responses)

    """
    Here's a breakdown of what the code does:

    1. It serves as the entry point of the script for data processing and visualization.
    2. It loads a dataframe from a specified file path.
    3. It searches for files in a directory that match the specified experiment name and plane.
    4. For each file found:
        It reads data from the file using the slab library.
        It extracts relevant information such as date, sub, distances, trials, and data from the file.
        It stores the extracted data into a dictionary called subdata.
    5. It gets unique sub_ids from the stored data.
    6. It converts the subdata dictionary into a Pandas DataFrame called df.
    7. It initializes empty lists for storing responses, iscorrect, diff_dist, dist1, and dist2.
    8. For each unique sub_id, it iterates over the data in df.data[sub_id] and extracts specific values such as "response", "correct", "distance_1", "distance_2", and the difference between distance_2 and distance_1. These values are appended to the respective lists.
    9. Finally, it uses seaborn (sns.lineplot) to plot a line graph, where iscorrect is plotted on the x-axis and responses on the y-axis.
    """
