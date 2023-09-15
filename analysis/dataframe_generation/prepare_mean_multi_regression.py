import pickle as pkl
from analysis.utils.plotting import *
import os


root = "/home/max/labplatform/data/csv"  # root

# data paths
labfp = os.path.join(root, "mad_babble_h.csv")  # locaaccu babble
lanfp = os.path.join(root, "mad_noise_h.csv")  # locaaccu noise
sufp = os.path.join(root, "spatmask_h_linear_slopes.csv")  # spatmask
njshfp = os.path.join(root, "numjudge_revspeech_h_gain.csv")  # percentage correct numjudge
njrfp = os.path.join(root, "numjudge_response_revspeech_h.csv")  # numjudge clearspeech responses
njpfp = os.path.join(root, "numjudge_performance_revspeech_h.csv")  # numjudge solution - resprev
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
sub_ids = np.array(range(1, 14))
mean_cov = [np.mean(coverage.loc["revspeech_h"]["coverage"][(x-1)*150:(x-1)*150+150]) for x in sub_ids]
mean_response = [njrh[(x-1)*150:(x-1)*150+150].mean().response for x in sub_ids]
mean_performance = [njph[(x-1)*150:(x-1)*150+150].mean().performance_reversed_h for x in sub_ids]

# finalize dataframe
finaldf = pd.DataFrame(columns=["performance", "response", "lababble",
                                "lanoise", "spatmask", "coverage", "numjudge", "subID"])
finaldf.performance = mean_performance
finaldf.response = mean_response
finaldf.lababble = [x[0] for x in labh.values.tolist()]
finaldf.lanoise = [x[0] for x in lanh.values.tolist()]
finaldf.coverage = mean_cov
finaldf.spatmask = [x[0] for x in suh.values.tolist()]
finaldf.numjudge = [x[0] for x in njsh.values.tolist()]
finaldf = finaldf.reset_index()
finaldf.subID = sub_ids
finaldf.pop("index")
finaldf = finaldf.fillna(0)

tofp = "/home/max/labplatform/data/linear_model/final_df_mean_revspeech_h.csv"
finaldf.to_csv(tofp)
