import pickle as pkl
from analysis.utils.plotting import *
import os


root = "/home/max/labplatform/data/csv"  # root

# data paths
labfp = os.path.join(root, "mad_babble_h.csv")  # locaaccu babble
lanfp = os.path.join(root, "mad_noise_h.csv")  # locaaccu noise
sufp = os.path.join(root, "spatmask_h_linear_slopes.csv")  # spatmask
njshfp = os.path.join(root, "numjudge_clearspeech_h_gain.csv")  # percentage correct numjudge
njrfp = os.path.join(root, "numjudge_response_clearspeech_h.csv")  # numjudge clearspeech responses
njpfp = os.path.join(root, "numjudge_performance_clearspeech_h.csv")  # numjudge solution - response

coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# pandas dataframes for all paradigms
labh = pd.read_csv(labfp)
lanh = pd.read_csv(lanfp)
suh = pd.read_csv(sufp, index_col=0)
njsh = pd.read_csv(njshfp, index_col=0)
njrh = pd.read_csv(njrfp, index_col=1)
njph = pd.read_csv(njpfp, index_col=1)
njrh.pop("Sub_ID")
njph.pop("Sub_ID")
sub_ids = np.array([range(1, 14)])

# make above dataframes length
dflen = njrh.__len__()
replication_factor = dflen/len(labh)
# Convert DataFrame to a numpy array and replicate each element
labh_array = np.repeat(labh.values, replication_factor, axis=0)
lanh_array = np.repeat(lanh.values, replication_factor, axis=0)
suh_array = np.repeat(suh.values, replication_factor, axis=0)
njsh_array = np.repeat(njsh.values, replication_factor, axis=0)
sub_ids = np.repeat(sub_ids, replication_factor, axis=1)
sub_ids = sub_ids.reshape((1950, 1))


# Create a new DataFrame with the replicated elements
new_labh = pd.DataFrame(labh_array, columns=labh.columns)
new_lanh = pd.DataFrame(lanh_array, columns=lanh.columns)
new_suh = pd.DataFrame(suh_array, columns=suh.columns)
new_njsh = pd.DataFrame(njsh_array, columns=njsh.columns)
new_subs = pd.DataFrame(sub_ids, columns=["subID"])


# finalize dataframe
finaldf = pd.DataFrame(columns=["response", "lababble", "lanoise", "spatmask", "coverage", "numjudge",
                                "performance", "subID"])
finaldf.response = njrh
finaldf.lababble = [x[0] for x in new_labh.values.tolist()]
finaldf.lanoise = [x[0] for x in new_lanh.values.tolist()]
finaldf.coverage = coverage.loc["clearspeech_h"]["coverage"]
finaldf.spatmask = [x[0] for x in new_suh.values.tolist()]
finaldf.numjudge = [x[0] for x in new_njsh.values.tolist()]
finaldf.performance = njph
finaldf = finaldf.reset_index()
finaldf.subID = new_subs
finaldf.pop("index")
finaldf = finaldf.fillna(0)

tofp = "/home/max/labplatform/data/linear_model/final_df_clearspeech_h.csv"
finaldf.to_csv(tofp)
