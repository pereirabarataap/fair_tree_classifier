import numpy as np

n_jobs = -1
n_bins = 3
n_folds = 3
max_depth = 2
bootstrap = True
random_state = 42
n_estimators = 2
max_features = "sqrt"
orthogonalities = np.linspace(0,1,3).round(2)
methods = ["local_sub", "kamiran_sub", "kamiran_div", "faht"]
datasets = [
    "bank_age",
    "adult_race", "adult_gender", "adult_multiple_1", "adult_multiple_2",
    "recidivism_age", "recidivism_race", "recidivism_gender", "recidivism_multiple_1", "recidivism_multiple_2"
]