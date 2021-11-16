import numpy as np

n_jobs = -1
n_bins = 10
n_folds = 10
max_depth = 4
bootstrap = True
random_state = 42
n_estimators = 500
max_features = "sqrt"
orthogonalities = np.linspace(0,1,11).round(2)
methods = ["local_sub", "kamiran_sub", "kamiran_div", "faht"]
datasets = [
    "bank_age",
    "adult_race", "adult_gender",
    "adult_multiple_1", "adult_multiple_2",
    "recidivism_race", "recidivism_gender",
]