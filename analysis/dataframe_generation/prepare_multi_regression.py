import pickle as pkl
from analysis.utils.plotting import *
import os
import seaborn as sns
sns.set_theme()


root = "/home/max/labplatform/data/csv"  # root

# data paths
labfp = os.path.join(root, "mad_babble_v.csv")  # locaaccu babble
lanfp = os.path.join(root, "mad_noise_v.csv")  # locaaccu noise
sufp = os.path.join(root, "spatmask_h_linear_slopes.csv")  # spatmask
njshfp = os.path.join(root, "numjudge_revspeech_v_gain.csv")  # percentage correct numjudge
njrfp = os.path.join(root, "numjudge_response_revspeech_v.csv")  # numjudge clearspeech

coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# pandas dataframes for all paradigms
labh = pd.read_csv(labfp)
lanh = pd.read_csv(lanfp)
suh = pd.read_csv(sufp, index_col=0)
njsh = pd.read_csv(njshfp, index_col=0)
njrh = pd.read_csv(njrfp, index_col=1)
njrh.pop("Sub_ID")

# make above dataframes length
dflen = njrh.__len__()
replication_factor = dflen/len(labh)
# Convert DataFrame to a numpy array and replicate each element
labh_array = np.repeat(labh.values, replication_factor, axis=0)
lanh_array = np.repeat(lanh.values, replication_factor, axis=0)
suh_array = np.repeat(suh.values, replication_factor, axis=0)
njsh_array = np.repeat(njsh.values, replication_factor, axis=0)

# Create a new DataFrame with the replicated elements
new_labh = pd.DataFrame(labh_array, columns=labh.columns)
new_lanh = pd.DataFrame(lanh_array, columns=lanh.columns)
new_suh = pd.DataFrame(suh_array, columns=suh.columns)
new_njsh = pd.DataFrame(njsh_array, columns=njsh.columns)


# finalize dataframe
finaldf = pd.DataFrame(columns=["response", "locaaccu_babble", "locaaccu_noise", "spatmask", "coverage", "numjudge"])
finaldf.response = njrh
finaldf.locaaccu_babble = [x[0] for x in new_labh.values.tolist()]
finaldf.locaaccu_noise = [x[0] for x in new_lanh.values.tolist()]
finaldf.coverage = coverage.loc["revspeech_v"]["coverage"]
finaldf.spatmask = [x[0] for x in new_suh.values.tolist()]
finaldf.numjudge = [x[0] for x in new_njsh.values.tolist()]
finaldf = finaldf.fillna(0)

tofp = "/home/max/labplatform/data/linear_model/final_df_revspeech_v.csv"
finaldf.to_csv(tofp)
