import os
from glob import glob
import pandas as pd
import slab

# get absolute filepath to cohort files
cwd = os.getcwd()
fp = os.path.join(cwd, "Results", "test")
files = glob(os.path.join(fp, '*'))
file = files[0]
data = slab.ResultsFile.read_file(file)
res = pd.Series(slab.ResultsFile.read_file(file, "response"))
sol = pd.Series(slab.ResultsFile.read_file(file, "solution"))
d = dict(response=res, solution=sol)
df = pd.DataFrame(d, columns=["response", "solution"])


