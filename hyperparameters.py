import numpy as np

n_jobs = -1
n_bins = 2
n_folds = 2
max_depth = 1
bootstrap = True
random_state = 42
n_estimators = 16
max_features = "sqrt"
orthogonalities = np.linspace(0,1,11).round(2)
methods = ["local_sub", "kamiran_sub", "kamiran_div", "faht"]
datasets = ["bank_age", "adult_race", "adult_gender", "recidivism_race", "recidivism_gender"]