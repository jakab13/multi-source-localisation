import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np

root = "/home/max/labplatform/data/DataFrames/"
su_fph = os.path.join(root, "spatmask_abs_distance_h")
su_fpv = os.path.join(root, "spatmask_abs_distance_v")
la_fph = os.path.join(root, "locaaccu_dfh")
la_fpv = os.path.join(root, "locaaccu_dfv")

suh = pickle.load(open(su_fph, "rb"))
suv = pickle.load(open(su_fpv, "rb"))

lah = pickle.load(open(la_fph, "rb"))
lav = pickle.load(open(la_fpv, "rb"))

model = LinearRegression(fit_intercept=True)  # linear regression model

single_subs = dict()
for col in suh.columns:
    single_subs[col] = list()
    for thresh1, thresh2 in zip(suh[col][:14], suh[col][14:]):
        single_subs[col].append((thresh1 + thresh2) / 2)

slopes = list()
X = np.array(list((single_subs.keys()))).reshape(-1, 1)
Y = [x for x in zip(single_subs[17.5], single_subs[35.0], single_subs[52.5])]

for y in Y:
    model.fit(X, y)
    slopes.append(model.coef_)

