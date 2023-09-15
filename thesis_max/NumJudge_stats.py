import scipy.stats as stats
from analysis.utils.misc import *
from labplatform.config import get_config
import pickle as pkl
from analysis.utils.stats import permutation_test
import statsmodels.formula.api as smf
import statsmodels.api as sm


fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
dfv = load_dataframe(fp, exp_name=exp_name, plane="v")
dfh = load_dataframe(fp, exp_name=exp_name, plane="h")

sub_ids = extract_subject_ids_from_dataframe(dfh)

# divide reversed speech blocks from clear speech
filledv = dfv.reversed_speech.ffill()
revspeechv = dfv[np.where(filledv==True, True, False)]  # True where reversed_speech is True

clearspeechv = dfv[np.where(filledv==False, True, False)]  # True where reversed_speech is False

filledh = dfh.reversed_speech.ffill()
revspeechh = dfh[np.where(filledh==True, True, False)]  # True where reversed_speech is True
clearspeechh = dfh[np.where(filledh==False, True, False)]  # True where reversed_speech is False
revspeechh = revspeechh.sort_index()
clearspeechh = clearspeechh.sort_index()
revspeechv = revspeechv.sort_index()
clearspeechv = clearspeechv.sort_index()

# reaction time
rtclearh = np.mean(clearspeechh.rt)
rtrevh = np.mean(revspeechh.rt)
rtclearv = np.mean(clearspeechv.rt)
rtrevv = np.mean(revspeechv.rt)

print(f"MEAN REACTION TIMES \n"
      f"AZIMUTH: \n"
      f"forward: {rtclearh} ms\n"
      f"reversed: {rtrevh} ms\n"
      f"ELEVATION: \n"
      f"forward: {rtclearv} ms\n"
      f"reversed: {rtrevv} ms\n")

print(f"WILCOXON SIGNED RANK TEST REACTION TIMES ELEVATION VS. AZIMUTH\n"
      f"{stats.wilcoxon(replace_in_array(np.append(clearspeechh.rt, revspeechh.rt)), replace_in_array(np.append(clearspeechv.rt, revspeechv.rt)))}")

# mean responses
means = dict(revmeanv=dict(), clearmeanv=dict(),
             revmeanh=dict(), clearmeanh=dict())

for val in np.unique(revspeechh.solution.values).tolist():
    means["revmeanv"][val] = revspeechv.response[revspeechv.solution == val].mean()
    means["clearmeanv"][val] = clearspeechv.response[clearspeechv.solution == val].mean()
    means["revmeanh"][val] = revspeechh.response[revspeechh.solution == val].mean()
    means["clearmeanh"][val] = clearspeechh.response[clearspeechh.solution == val].mean()


stderrs = dict(revmeanv=dict(), clearmeanv=dict(),
               revmeanh=dict(), clearmeanh=dict())

for val in np.unique(revspeechh.solution.values).tolist():
    stderrs["revmeanv"][val] = stats.sem(revspeechv.response[revspeechv.solution == val].fillna(0))
    stderrs["clearmeanv"][val] = stats.sem(clearspeechv.response[clearspeechv.solution == val].fillna(0))
    stderrs["revmeanh"][val] = stats.sem(revspeechh.response[revspeechh.solution == val].fillna(0))
    stderrs["clearmeanh"][val] = stats.sem(clearspeechh.response[clearspeechh.solution == val].fillna(0))

# spectro-temporal coverage
coverage = pkl.load(open("Results/coverage_dataframe.pkl", "rb"))

# get mean coverage scores
covmeans = dict(revmeanv=dict(), clearmeanv=dict(),
                revmeanh=dict(), clearmeanh=dict())

for val in np.unique(revspeechh.solution.values).tolist():
    covmeans["revmeanv"][val] = pd.Series(coverage.loc["revspeech_v"]["coverage"], index=revspeechv.solution.index)[revspeechv.solution == val].mean()
    covmeans["clearmeanv"][val] = pd.Series(coverage.loc["clearspeech_v"]["coverage"], index=clearspeechh.solution.index)[clearspeechv.solution == val].mean()
    covmeans["revmeanh"][val] = pd.Series(coverage.loc["revspeech_h"]["coverage"], index=revspeechh.solution.index)[revspeechh.solution == val].mean()
    covmeans["clearmeanh"][val] = pd.Series(coverage.loc["clearspeech_h"]["coverage"], index=clearspeechh.solution.index)[clearspeechh.solution == val].mean()

covstderrs = dict(revmeanv=dict(), clearmeanv=dict(),
                  revmeanh=dict(), clearmeanh=dict())

for val in np.unique(revspeechh.solution.values).tolist():
    covstderrs["revmeanv"][val] = stats.sem(revspeechv.response[revspeechv.solution == val].fillna(0))
    covstderrs["clearmeanv"][val] = stats.sem(clearspeechv.response[clearspeechv.solution == val].fillna(0))
    covstderrs["revmeanh"][val] = stats.sem(revspeechh.response[revspeechh.solution == val].fillna(0))
    covstderrs["clearmeanh"][val] = stats.sem(clearspeechh.response[clearspeechh.solution == val].fillna(0))

# permutation test on spetcral coverage
permutation_test(pd.Series(coverage.loc["revspeech_v"]["coverage"]), pd.Series(coverage.loc["clearspeech_v"]["coverage"]))

# linear model
formula = "response ~ coverage + spatmask + lababble + lanoise + numjudge"

# random intercept, random slope
vc_formula = {"spatmask": "0 + spatmask",
              "lababble": "0 + lababble",
              "lanoise": "0 + lanoise",
              "numjudge": "0 + numjudge"}

# clearspeech azimuth
clearspeechh_model_fp = "/home/max/labplatform/data/linear_model/final_df_clearspeech_h.csv"
clearspeechh_model_df = pd.read_csv(clearspeechh_model_fp, index_col=0)
modelch = smf.mixedlm(formula=formula, data=clearspeechh_model_df, groups=clearspeechh_model_df["subID"])
resultch = modelch.fit()
resultch.summary()

# clearspeech elevation
clearspeechv_model_fp = "/home/max/labplatform/data/linear_model/final_df_clearspeech_v.csv"
clearspeechv_model_df = pd.read_csv(clearspeechv_model_fp, index_col=0)
modelcv = smf.mixedlm(formula, data=clearspeechv_model_df, groups=clearspeechv_model_df["subID"])
resultcv = modelcv.fit()
resultcv.summary()

# reversed speech azimuth
revspeechh_model_fp = "/home/max/labplatform/data/linear_model/final_df_revspeech_h.csv"
revspeechh_model_df = pd.read_csv(revspeechh_model_fp, index_col=0)
modelrh = smf.mixedlm(formula, data=revspeechh_model_df, groups=revspeechh_model_df["subID"])
resultrh = modelrh.fit()
resultrh.summary()

# reversed speech elevation
revspeechv_model_fp = "/home/max/labplatform/data/linear_model/final_df_revspeech_v.csv"
revspeechv_model_df = pd.read_csv(revspeechv_model_fp, index_col=0)
modelrv = smf.mixedlm(formula, data=revspeechv_model_df, groups=revspeechv_model_df["subID"])
resultrv = modelrv.fit()
resultrv.summary()


