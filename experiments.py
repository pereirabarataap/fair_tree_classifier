import os
import sys
import joblib
from datasets import *
from shutil import rmtree
from datetime import datetime
from itertools import product
from tqdm.notebook import tqdm as tqdm_n
from ftl import FairRandomForestClassifier as FRFC
from sklearn.model_selection import StratifiedKFold as SKF

def make_setup(
    n_bins, max_depth, random_state, n_folds,
    datasets_folder, experiments_folder, methods, datasets
):  
    try:
        os.mkdir(datasets_folder)
    except:
        rmtree(datasets_folder)
        os.mkdir(datasets_folder)
        
    for dataset in datasets:
        dataset_folder = datasets_folder + "/" + dataset
        try:
            os.mkdir(dataset_folder)
        except:
            pass
        X, y, s = globals()["get_"+dataset](show=False)
        joblib.dump(X, dataset_folder+"/X.pkl")
        joblib.dump(y, dataset_folder+"/y.pkl")
        joblib.dump(s, dataset_folder+"/s.pkl")
        splitter = SKF(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold = 0
        for train_idx, test_idx in splitter.split(X, y):
            joblib.dump(test_idx, dataset_folder+"/"+str(fold)+"_test_idx.pkl")
            joblib.dump(train_idx, dataset_folder+"/"+str(fold)+"_train_idx.pkl")
            fold += 1
    try:
        os.mkdir(experiments_folder)
    except:
        rmtree(experiments_folder)
        os.mkdir(experiments_folder)
        
    for method in methods:
        method_folder = experiments_folder + "/" + method
        try:
            os.mkdir(method_folder)
        except:
            rmtree(method_folder)
            os.mkdir(method_folder)
        
        for dataset in datasets:
            exp_dataset_folder = method_folder + "/" + dataset
            try:
                os.mkdir(exp_dataset_folder)
            except:
                rmtree(exp_dataset_folder)
                os.mkdir(exp_dataset_folder)
                    
def make_experiments_df(n_folds, orthogonalities, datasets_folder, experiments_folder, methods, datasets):
    data = []
    parameters = list(product(methods, datasets))
    for parameter in parameters:
        method, dataset = parameter
        dataset_folder = datasets_folder + "/" + dataset
        method_folder = experiments_folder + "/" + method
        exp_dataset_folder = method_folder + "/" + dataset

        X = joblib.load(dataset_folder+"/X.pkl")
        y = joblib.load(dataset_folder+"/y.pkl")
        s = joblib.load(dataset_folder+"/s.pkl")

        test_idxs = {
            fold: joblib.load(dataset_folder+"/"+str(fold)+"_test_idx.pkl") for fold in range(n_folds)
        }
        if method=="local_sub":
            for orthogonality in orthogonalities:
                for fold in range(n_folds):
                    file_name = str(orthogonality) + "_" + str(fold) + ".pkl"
                    file_path = exp_dataset_folder + "/" + file_name
                    file = joblib.load(file_path)
                    test_idx = test_idxs[fold]
                    y_test = y[test_idx]
                    s_test = s[test_idx]
                    p_test = file["p_test"]

                    row = [method, dataset, orthogonality, fold, y_test, s_test, p_test]
                    data.append(row)

                    print_line = str(datetime.now()) + " " + method + " " + dataset
                    print_line += " orthogonality=" + str(orthogonality) + " fold=" + str(fold)
                    sys.stdout.write("\r" + str(print_line)+"\t\t\t\t\t\t")
                    sys.stdout.flush()
        else:
            orthogonality = -1
            for fold in range(n_folds):
                file_name = str(fold) + ".pkl"
                file_path = exp_dataset_folder + "/" + file_name
                file = joblib.load(file_path)
                test_idx = test_idxs[fold]
                y_test = y[test_idx]
                s_test = s[test_idx]
                p_test = file["p_test"]

                row = [method, dataset, orthogonality, fold, y_test, s_test, p_test]
                data.append(row)

                print_line = str(datetime.now()) + " " + method +  " " + dataset
                print_line += " fold=" + str(fold)
                sys.stdout.write("\r" + str(print_line)+"\t\t\t\t\t\t")
                sys.stdout.flush()

    experiments_df = pd.DataFrame(
        data=data,
        columns=[
            "method", "dataset", "orthogonality", "fold", "y_test", "s_test", "p_test"
        ]
    )

    joblib.dump(experiments_df, "experiments_df.pkl")
    
def run_experiments(
    n_jobs, n_bins, n_folds, max_depth, bootstrap, random_state, n_estimators, max_features,
    orthogonalities, datasets_folder, experiments_folder, methods, datasets
): 
    parameters = list(product(methods, datasets))
    for parameter in tqdm_n(parameters):
        method, dataset = parameter
        dataset_folder = datasets_folder + "/" + dataset
        method_folder = experiments_folder + "/" + method
        exp_dataset_folder = method_folder + "/" + dataset

        X = joblib.load(dataset_folder+"/X.pkl")
        y = joblib.load(dataset_folder+"/y.pkl")
        s = joblib.load(dataset_folder+"/s.pkl")

        if "local_sub"==method:
            criterion = "auc_" + method.split("_")[-1]
            for orthogonality in orthogonalities:
                for fold in range(n_folds):
                    file_name = str(orthogonality) + "_" + str(fold) + ".pkl"
                    if file_name not in os.listdir(exp_dataset_folder):
                        file = {}
                        file_path = exp_dataset_folder + "/" + file_name
                        test_idx = joblib.load(dataset_folder+"/"+str(fold)+"_test_idx.pkl")
                        train_idx = joblib.load(dataset_folder+"/"+str(fold)+"_train_idx.pkl")
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        s_train, s_test = s[train_idx], s[test_idx]
                        clf = FRFC(
                            n_bins=n_bins,
                            n_jobs=n_jobs,
                            criterion=criterion,
                            bootstrap=bootstrap,
                            max_depth=max_depth,
                            random_state=random_state,
                            n_estimators=n_estimators,
                            max_features=max_features,
                            orthogonality=orthogonality,
                        )
                        file["start"] = datetime.now()
                        clf.fit(X_train, y_train, s_train)
                        p_test = clf.predict_proba(X_test)[:,1]
                        p_train = clf.predict_proba(X_train)[:,1]
                        file["stop"] = datetime.now()
                        file["p_train"] = p_train
                        file["p_test"] = p_test
                        joblib.dump(file, file_path)
                        print_line = str(datetime.now()) + " " + method + " " + dataset
                        print_line += " orthogonality=" + str(orthogonality) + " fold=" + str(fold) + " time: " + str(file["stop"] - file["start"])
                        sys.stdout.write("\r" + str(print_line)+"\t\t\t\t\t\t")
                        sys.stdout.flush()

        else:
            for fold in range(n_folds):
                file_name = str(fold) + ".pkl"
                if True:#file_name not in os.listdir(exp_dataset_folder):
                    file = {}
                    file_path = exp_dataset_folder + "/" + file_name
                    test_idx = joblib.load(dataset_folder+"/"+str(fold)+"_test_idx.pkl")
                    train_idx = joblib.load(dataset_folder+"/"+str(fold)+"_train_idx.pkl")
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    s_train, s_test = s[train_idx], s[test_idx]
                    clf = FRFC(
                        n_bins=n_bins,
                        n_jobs=n_jobs,
                        criterion=method,
                        bootstrap=bootstrap,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_estimators=n_estimators,
                        max_features=max_features,
                    )
                    file["start"] = datetime.now()
                    clf.fit(X_train, y_train, s_train)
                    p_test = clf.predict_proba(X_test)[:,1]
                    p_train = clf.predict_proba(X_train)[:,1]
                    file["stop"] = datetime.now()
                    file["p_train"] = p_train
                    file["p_test"] = p_test
                    joblib.dump(file, file_path)
                    print_line = str(datetime.now()) + " " + method + " " + dataset
                    print_line += " fold=" + str(fold) + " time: " + str(file["stop"] - file["start"])
                    sys.stdout.write("\r" + str(print_line)+"\t\t\t\t\t\t")
                    sys.stdout.flush()
                    
    make_experiments_df(n_folds, orthogonalities, datasets_folder, experiments_folder, methods, datasets)