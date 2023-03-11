import os
import pandas as pd
import slab
from labplatform.config import get_config
import seaborn as sns
import matplotlib.pyplot as plt

fp = os.path.join(get_config("DATA_ROOT"), "MSL")
exp_name = "NumJudge"
plane = "v"
columns = ["response", "solution", "rt", "is_correct"]
responses = list()
solutions = list()
rts = list()
is_corrects = list()

for root, dirs, files in os.walk(fp):
    for file in files:
        if exp_name in file and plane in file:
            data_path = os.path.join(root, file)
            responses.extend(slab.ResultsFile.read_file(data_path, "response"))
            solutions.extend(slab.ResultsFile.read_file(data_path, "solution"))
            rts.extend(slab.ResultsFile.read_file(data_path, "rt"))
            is_corrects.extend(slab.ResultsFile.read_file(data_path, "is_correct"))
        else:
            continue

df = pd.DataFrame(columns=["response", "solution", "rt", "is_correct"])
df["response"] = responses
df["solution"] = solutions
df["rt"] = rts
df["is_correct"] = is_corrects

plot = sns.lineplot(data=df, x=df.solution.dropna(), y=df.response.dropna(), err_style="bars", errorbar=("se", 2))
plot.set_xticks(range(2, 6))
plot.set_yticks(range(2, 6))
plot.set_title(plane)
x0, x1 = plot.get_xlim()
y0, y1 = plot.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
plot.plot(lims, lims, color='grey', linestyle="dashed")
plt.show()
