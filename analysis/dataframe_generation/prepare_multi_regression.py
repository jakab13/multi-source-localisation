import pickle as pkl
from analysis.utils.plotting import *
import os
import seaborn as sns
from labplatform.config import get_config
sns.set_theme()


# load data from all subjects
fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

filled_h = dfh.reversed_speech.ffill()
revspeech_h = dfh[np.where(filled_h==True, True, False)]  # True where reversed_speech is True
revspeech_h = revspeech_h.sort_index()
clearspeech_h = dfh[np.where(filled_h==False, True, False)]  # True where reversed_speech is False
clearspeech_h = clearspeech_h.sort_index()

# vertical
filled_v = dfv.reversed_speech.ffill()
revspeech_v = dfv[np.where(filled_v==True, True, False)]  # True where reversed_speech is True
revspeech_v = revspeech_v.sort_index()
clearspeech_v = dfv[np.where(filled_v==False, True, False)]  # True where reversed_speech is False
clearspeech_v = clearspeech_v.sort_index()

# general performance
# prepare horizontal clearspeech
# filepaths
root = "/home/max/labplatform/data/csv"  # root

# data paths
labfp = os.path.join(root, "locaaccu_babble_h_gain.csv")  # locaaccu babble
lanfp = os.path.join(root, "locaaccu_noise_h_gain.csv")  # locaaccu noise
sufp = os.path.join(root, "spatmask_threshold_h.csv")  # spatmask
njpchfp = os.path.join(root, "numjudge_percentage_correct_clear_h.csv")  # percentage correct numjudge
ccsfp = os.path.join(root, "coverage_clearspeech_h.csv")  # spectral coverage clearspeech
njfp = os.path.join(root, "numjudge_performance_clearspeech_h.csv")  # numjudge clearspeech

coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))
# some processing
performance = pd.read_csv("/home/max/labplatform/data/csv/numjudge_performance_clearspeech_h.csv", index_col=1)
performance.pop("Sub_ID")
performance = [x[0] for x in performance.values.tolist()]

# pandas dataframes for all paradigms
labh = pd.read_csv(labfp, index_col=0)
lanh = pd.read_csv(lanfp, index_col=0)
suh = pd.read_csv(sufp, index_col=0)
njpch = pd.read_csv(njpchfp, index_col=0)

# make above dataframes length 2100
dflen = 2100
replication_factor = dflen/len(labh)
# Convert DataFrame to a numpy array and replicate each element
labh_array = np.repeat(labh.values, replication_factor, axis=0)
lanh_array = np.repeat(lanh.values, replication_factor, axis=0)
suh_array = np.repeat(suh.values, replication_factor, axis=0)
njpch_array = np.repeat(njpch.values, replication_factor, axis=0)

# Create a new DataFrame with the replicated elements
new_labh = pd.DataFrame(labh_array, columns=labh.columns)
new_lanh = pd.DataFrame(lanh_array, columns=lanh.columns)
new_suh = pd.DataFrame(suh_array, columns=suh.columns)
new_njpch = pd.DataFrame(njpch_array, columns=njpch.columns)


# finalize dataframe
finaldf = pd.DataFrame(columns=["response", "solution", "performance", "locaaccu_babble", "locaaccu_noise", "spatmask",
                                "coverage", "percentage_correct"])
finaldf.response = clearspeech_h.response
finaldf.solution = clearspeech_h.solution
finaldf.performance = performance
finaldf.locaaccu_babble = [x[0] for x in new_labh.values.tolist()]
finaldf.locaaccu_noise = [x[0] for x in new_lanh.values.tolist()]
finaldf.percentage_correct = [x[0] for x in new_njpch.values.tolist()]
finaldf.coverage = coverage.loc["clearspeech_h"]["coverage"]
finaldf.spatmask = [x[0] for x in new_suh.values.tolist()]
finaldf = finaldf.fillna(0)

tofp = "/home/max/labplatform/data/csv/final_df_general.csv"
finaldf.to_csv(tofp)
