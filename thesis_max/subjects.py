import numpy as np

n_subs = 13
n_males = 7
n_females = 6

ages = np.array([28, 29, 22, 21, 20, 22, 23, 18, 22, 23, 28, 22, 23])

print(f"Amount of subjects: {n_subs} \n"
      f"Amount of females: {n_females} \n"
      f"Age mean: {ages.mean()} \n"
      f"Age SD: {ages.std()}")

