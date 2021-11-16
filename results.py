import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr,spearmanr
from matplotlib import pyplot as plt, patheffects as pe
from sklearn.metrics import roc_auc_score, accuracy_score

def demographic_parity(s_test, y_pred):
    pos_y_cond_neg_s_prob = sum(y_pred[s_test==0]==1) / sum(s_test==0)
    pos_y_cond_pos_s_prob = sum(y_pred[s_test==1]==1) / sum(s_test==1)
    demographic_parity = abs(pos_y_cond_neg_s_prob - pos_y_cond_pos_s_prob)
    return demographic_parity

def equal_odds_tpr(s_test, y_pred, y_test):
    pos_s_loc = s_test==1
    neg_s_loc = s_test==0
    pos_pred_loc = y_pred==1
    pos_true_loc = y_test==1
    pos_s_tpr = sum(pos_pred_loc & pos_s_loc & pos_true_loc) / sum(pos_true_loc & pos_s_loc)
    neg_s_tpr = sum(pos_pred_loc & neg_s_loc & pos_true_loc) / sum(pos_true_loc & neg_s_loc)
    equal_odds_tpr = abs(pos_s_tpr - neg_s_tpr)
    return equal_odds_tpr

def equal_odds_fpr(s_test, y_pred, y_test):    
    pos_s_loc = s_test==1
    neg_s_loc = s_test==0
    neg_pred_loc = y_pred==0
    neg_true_loc = y_test==0
    pos_s_tnr = sum(neg_pred_loc & pos_s_loc & neg_true_loc) / sum(neg_true_loc & pos_s_loc)
    neg_s_tnr = sum(neg_pred_loc & neg_s_loc & neg_true_loc) / sum(neg_true_loc & neg_s_loc)
    pos_s_fpr = 1 - pos_s_tnr
    neg_s_fpr = 1 - neg_s_tnr
    equal_odds_fpr = abs(pos_s_fpr - neg_s_fpr)
    return equal_odds_fpr

def get_results(n_folds, orthogonalities, methods, datasets):
    results = {}
    experiments_df = joblib.load("experiments_df.pkl")
    for method in methods:
        results[method] = {}
        for dataset in datasets:
            results[method][dataset] = {}          

    parameters = list(product(methods, datasets))
    for parameter in parameters:
        method, dataset = parameter
        if "multiple" not in dataset:
            if method=="local_sub":
                y_test_auc_means = []
                s_test_auc_means = []
                y_test_auc_stdvs = []
                s_test_auc_stdvs = []
                for orthogonality in orthogonalities:
                    y_test_ortho_aucs = []
                    s_test_ortho_aucs = []
                    for fold in range(n_folds):
                        loc = \
                            (experiments_df["fold"]==fold ) & \
                            (experiments_df["method"]==method ) & \
                            (experiments_df["dataset"]==dataset ) & \
                            (experiments_df["orthogonality"]==orthogonality )
                        y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                        y_test_auc = roc_auc_score(y_test, p_test)
                        s_test_auc = roc_auc_score(s_test, p_test)
                        s_test_auc = max(1-s_test_auc, s_test_auc)
                        y_test_ortho_aucs.append(y_test_auc)
                        s_test_ortho_aucs.append(s_test_auc)
                    y_test_auc_means.append(np.mean(y_test_ortho_aucs))
                    s_test_auc_means.append(np.mean(s_test_ortho_aucs))
                    y_test_auc_stdvs.append(np.std(y_test_ortho_aucs, ddof=0))
                    s_test_auc_stdvs.append(np.std(s_test_ortho_aucs, ddof=0))
                results[method][dataset]["y_test_auc_means"] = np.array(y_test_auc_means)
                results[method][dataset]["s_test_auc_means"] = np.array(s_test_auc_means)
                results[method][dataset]["y_test_auc_stdvs"] = np.array(y_test_auc_stdvs)
                results[method][dataset]["s_test_auc_stdvs"] = np.array(s_test_auc_stdvs)
            else:
                y_test_aucs = []
                s_test_aucs = []
                orthogonality = -1
                for fold in range(n_folds):
                    loc = \
                        (experiments_df["fold"]==fold ) & \
                        (experiments_df["method"]==method ) & \
                        (experiments_df["dataset"]==dataset ) & \
                        (experiments_df["orthogonality"]==orthogonality )
                    y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                    y_test_auc = roc_auc_score(y_test, p_test)
                    s_test_auc = roc_auc_score(s_test, p_test)
                    s_test_auc = max(1-s_test_auc, s_test_auc)
                    y_test_aucs.append(y_test_auc)
                    s_test_aucs.append(s_test_auc)
                results[method][dataset]["y_test_auc_mean"] = np.mean(y_test_aucs)
                results[method][dataset]["s_test_auc_mean"] = np.mean(s_test_aucs)
                results[method][dataset]["y_test_auc_stdv"] = np.std(y_test_aucs, ddof=0)
                results[method][dataset]["s_test_auc_stdv"] = np.std(s_test_aucs, ddof=0)
        elif "adult_multiple_1" in dataset:
            if method=="local_sub":
                y_test_auc_means = []                
                y_test_auc_stdvs = []
                
                #race
                s_test_auc_means_0 = []
                s_test_auc_stdvs_0 = []
                
                #gender
                s_test_auc_means_1 = []
                s_test_auc_stdvs_1 = []
                
                for orthogonality in orthogonalities:
                    y_test_ortho_aucs = []
                    
                    s_test_ortho_aucs_0 = []
                    s_test_ortho_aucs_1 = []
                    for fold in range(n_folds):
                        loc = \
                            (experiments_df["fold"]==fold ) & \
                            (experiments_df["method"]==method ) & \
                            (experiments_df["dataset"]==dataset ) & \
                            (experiments_df["orthogonality"]==orthogonality )
                        y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                        y_test_auc = roc_auc_score(y_test, p_test)
                        y_test_ortho_aucs.append(y_test_auc)
                        
                        s_test_0 = s_test[:,0]
                        s_test_auc_0 = roc_auc_score(s_test_0, p_test)
                        s_test_auc_0 = max(1-s_test_auc_0, s_test_auc_0)
                        s_test_ortho_aucs_0.append(s_test_auc_0)
                        
                        s_test_1 = s_test[:,1]
                        s_test_auc_1 = roc_auc_score(s_test_1, p_test)
                        s_test_auc_1 = max(1-s_test_auc_1, s_test_auc_1)
                        s_test_ortho_aucs_1.append(s_test_auc_1)
                        
                    y_test_auc_means.append(np.mean(y_test_ortho_aucs))
                    y_test_auc_stdvs.append(np.std(y_test_ortho_aucs, ddof=0))
                    
                    s_test_auc_means_0.append(np.mean(s_test_ortho_aucs_0))
                    s_test_auc_stdvs_0.append(np.std(s_test_ortho_aucs_0, ddof=0))
                    
                    s_test_auc_means_1.append(np.mean(s_test_ortho_aucs_1))
                    s_test_auc_stdvs_1.append(np.std(s_test_ortho_aucs_1, ddof=0))
                    
                results[method][dataset]["y_test_auc_means"] = np.array(y_test_auc_means)
                results[method][dataset]["y_test_auc_stdvs"] = np.array(y_test_auc_stdvs)
                
                results[method][dataset]["s_test_auc_means_0"] = np.array(s_test_auc_means_0)
                results[method][dataset]["s_test_auc_stdvs_0"] = np.array(s_test_auc_stdvs_0)
                results[method][dataset]["s_test_auc_means_1"] = np.array(s_test_auc_means_1)
                results[method][dataset]["s_test_auc_stdvs_1"] = np.array(s_test_auc_stdvs_1)
        elif "adult_multiple_2" in dataset:
            if method=="local_sub":
                y_test_auc_means = []                
                y_test_auc_stdvs = []
                
                # 00, 01, 10, 11: BF, BM, WF, WM
                s_test_auc_means_00 = []
                s_test_auc_stdvs_00 = []
                s_test_auc_means_01 = []
                s_test_auc_stdvs_01 = []
                s_test_auc_means_10 = []
                s_test_auc_stdvs_10 = []
                s_test_auc_means_11 = []
                s_test_auc_stdvs_11 = []
                
            
                
                for orthogonality in orthogonalities:
                    y_test_ortho_aucs = []
                    
                    s_test_ortho_aucs_00 = []
                    s_test_ortho_aucs_01 = []
                    s_test_ortho_aucs_10 = []
                    s_test_ortho_aucs_11 = []
                    
                    for fold in range(n_folds):
                        loc = \
                            (experiments_df["fold"]==fold ) & \
                            (experiments_df["method"]==method ) & \
                            (experiments_df["dataset"]==dataset ) & \
                            (experiments_df["orthogonality"]==orthogonality )
                        y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                        y_test_auc = roc_auc_score(y_test, p_test)
                        y_test_ortho_aucs.append(y_test_auc)
                        
                        
                        s_test_00 = s_test[:,0]
                        s_test_auc_00 = roc_auc_score(s_test_00, p_test)
                        s_test_auc_00 = max(1-s_test_auc_00, s_test_auc_00)
                        s_test_ortho_aucs_00.append(s_test_auc_00)
                        
                        s_test_01 = s_test[:,1]
                        s_test_auc_01 = roc_auc_score(s_test_01, p_test)
                        s_test_auc_01 = max(1-s_test_auc_01, s_test_auc_01)
                        s_test_ortho_aucs_01.append(s_test_auc_01)
                        
                        s_test_10 = s_test[:,2]
                        s_test_auc_10 = roc_auc_score(s_test_10, p_test)
                        s_test_auc_10 = max(1-s_test_auc_10, s_test_auc_10)
                        s_test_ortho_aucs_10.append(s_test_auc_10)
                        
                        s_test_11 = s_test[:,3]
                        s_test_auc_11 = roc_auc_score(s_test_11, p_test)
                        s_test_auc_11 = max(1-s_test_auc_11, s_test_auc_11)
                        s_test_ortho_aucs_11.append(s_test_auc_11)
                        
                        
                    y_test_auc_means.append(np.mean(y_test_ortho_aucs))
                    y_test_auc_stdvs.append(np.std(y_test_ortho_aucs, ddof=0))
                    
                    s_test_auc_means_00.append(np.mean(s_test_ortho_aucs_00))                    
                    s_test_auc_means_01.append(np.mean(s_test_ortho_aucs_01))                    
                    s_test_auc_means_10.append(np.mean(s_test_ortho_aucs_10))
                    s_test_auc_means_11.append(np.mean(s_test_ortho_aucs_11))
                    
                    s_test_auc_stdvs_00.append(np.std(s_test_ortho_aucs_00, ddof=0))
                    s_test_auc_stdvs_01.append(np.std(s_test_ortho_aucs_01, ddof=0))                    
                    s_test_auc_stdvs_10.append(np.std(s_test_ortho_aucs_10, ddof=0))                    
                    s_test_auc_stdvs_11.append(np.std(s_test_ortho_aucs_11, ddof=0))
                    
                results[method][dataset]["y_test_auc_means"] = np.array(y_test_auc_means)
                results[method][dataset]["y_test_auc_stdvs"] = np.array(y_test_auc_stdvs)
                
                results[method][dataset]["s_test_auc_means_00"] = np.array(s_test_auc_means_00)
                results[method][dataset]["s_test_auc_means_01"] = np.array(s_test_auc_means_01)
                results[method][dataset]["s_test_auc_means_10"] = np.array(s_test_auc_means_10)
                results[method][dataset]["s_test_auc_means_11"] = np.array(s_test_auc_means_11)
                
                results[method][dataset]["s_test_auc_stdvs_00"] = np.array(s_test_auc_stdvs_00)
                results[method][dataset]["s_test_auc_stdvs_01"] = np.array(s_test_auc_stdvs_01)
                results[method][dataset]["s_test_auc_stdvs_10"] = np.array(s_test_auc_stdvs_10)                
                results[method][dataset]["s_test_auc_stdvs_11"] = np.array(s_test_auc_stdvs_11)
        elif "recidivism_multiple_1" in dataset:
            if method=="local_sub":
                y_test_auc_means = []                
                y_test_auc_stdvs = []
                
                #age
                s_test_auc_means_0 = []
                s_test_auc_stdvs_0 = []
                
                #race
                s_test_auc_means_1 = []
                s_test_auc_stdvs_1 = []
                
                #gender
                s_test_auc_means_2 = []
                s_test_auc_stdvs_2 = []
                
                for orthogonality in orthogonalities:
                    y_test_ortho_aucs = []
                    
                    s_test_ortho_aucs_0 = []
                    s_test_ortho_aucs_1 = []
                    s_test_ortho_aucs_2 = []
                    for fold in range(n_folds):
                        loc = \
                            (experiments_df["fold"]==fold ) & \
                            (experiments_df["method"]==method ) & \
                            (experiments_df["dataset"]==dataset ) & \
                            (experiments_df["orthogonality"]==orthogonality )
                        y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                        y_test_auc = roc_auc_score(y_test, p_test)
                        y_test_ortho_aucs.append(y_test_auc)
                        
                        s_test_0 = s_test[:,0]
                        s_test_auc_0 = roc_auc_score(s_test_0, p_test)
                        s_test_auc_0 = max(1-s_test_auc_0, s_test_auc_0)
                        s_test_ortho_aucs_0.append(s_test_auc_0)
                        
                        s_test_1 = s_test[:,1]
                        s_test_auc_1 = roc_auc_score(s_test_1, p_test)
                        s_test_auc_1 = max(1-s_test_auc_1, s_test_auc_1)
                        s_test_ortho_aucs_1.append(s_test_auc_1)
                        
                        s_test_2 = s_test[:,2]
                        s_test_auc_2 = roc_auc_score(s_test_2, p_test)
                        s_test_auc_2 = max(1-s_test_auc_2, s_test_auc_2)
                        s_test_ortho_aucs_2.append(s_test_auc_2)
                        
                    y_test_auc_means.append(np.mean(y_test_ortho_aucs))
                    y_test_auc_stdvs.append(np.std(y_test_ortho_aucs, ddof=0))
                    
                    s_test_auc_means_0.append(np.mean(s_test_ortho_aucs_0))
                    s_test_auc_stdvs_0.append(np.std(s_test_ortho_aucs_0, ddof=0))
                    
                    s_test_auc_means_1.append(np.mean(s_test_ortho_aucs_1))
                    s_test_auc_stdvs_1.append(np.std(s_test_ortho_aucs_1, ddof=0))
                    
                    s_test_auc_means_2.append(np.mean(s_test_ortho_aucs_2))
                    s_test_auc_stdvs_2.append(np.std(s_test_ortho_aucs_2, ddof=0))
                    
                results[method][dataset]["y_test_auc_means"] = np.array(y_test_auc_means)
                results[method][dataset]["y_test_auc_stdvs"] = np.array(y_test_auc_stdvs)
                
                results[method][dataset]["s_test_auc_means_0"] = np.array(s_test_auc_means_0)
                results[method][dataset]["s_test_auc_stdvs_0"] = np.array(s_test_auc_stdvs_0)
                results[method][dataset]["s_test_auc_means_1"] = np.array(s_test_auc_means_1)
                results[method][dataset]["s_test_auc_stdvs_1"] = np.array(s_test_auc_stdvs_1)
                results[method][dataset]["s_test_auc_means_2"] = np.array(s_test_auc_means_2)
                results[method][dataset]["s_test_auc_stdvs_2"] = np.array(s_test_auc_stdvs_2)
        elif "recidivism_multiple_2" in dataset:
            if method=="local_sub":
                y_test_auc_means = []                
                y_test_auc_stdvs = []
                
                
                # 000, 001, 010, 011, 100, 101, 110, 111
                # OBF, OBM, OWF, OWM, YBF, YBM, YWF, YWM
                s_test_auc_means_000 = []
                s_test_auc_stdvs_000 = []
                s_test_auc_means_001 = []
                s_test_auc_stdvs_001 = []
                s_test_auc_means_010 = []
                s_test_auc_stdvs_010 = []
                s_test_auc_means_011 = []
                s_test_auc_stdvs_011 = []
                s_test_auc_means_100 = []
                s_test_auc_stdvs_100 = []
                s_test_auc_means_101 = []
                s_test_auc_stdvs_101 = []
                s_test_auc_means_110 = []
                s_test_auc_stdvs_110 = []
                s_test_auc_means_111 = []
                s_test_auc_stdvs_111 = []
            
                
                for orthogonality in orthogonalities:
                    y_test_ortho_aucs = []
                    
                    s_test_ortho_aucs_000 = []
                    s_test_ortho_aucs_001 = []
                    s_test_ortho_aucs_010 = []
                    s_test_ortho_aucs_011 = []
                    s_test_ortho_aucs_100 = []
                    s_test_ortho_aucs_101 = []
                    s_test_ortho_aucs_110 = []
                    s_test_ortho_aucs_111 = []
                    
                    for fold in range(n_folds):
                        loc = \
                            (experiments_df["fold"]==fold ) & \
                            (experiments_df["method"]==method ) & \
                            (experiments_df["dataset"]==dataset ) & \
                            (experiments_df["orthogonality"]==orthogonality )
                        y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
                        y_test_auc = roc_auc_score(y_test, p_test)
                        y_test_ortho_aucs.append(y_test_auc)
                        
                        
                        s_test_000 = s_test[:,0]
                        s_test_auc_000 = roc_auc_score(s_test_000, p_test)
                        s_test_auc_000 = max(1-s_test_auc_000, s_test_auc_000)
                        s_test_ortho_aucs_000.append(s_test_auc_000)
                        
                        s_test_001 = s_test[:,1]
                        s_test_auc_001 = roc_auc_score(s_test_001, p_test)
                        s_test_auc_001 = max(1-s_test_auc_001, s_test_auc_001)
                        s_test_ortho_aucs_001.append(s_test_auc_001)
                        
                        s_test_010 = s_test[:,2]
                        s_test_auc_010 = roc_auc_score(s_test_010, p_test)
                        s_test_auc_010 = max(1-s_test_auc_010, s_test_auc_010)
                        s_test_ortho_aucs_010.append(s_test_auc_010)
                        
                        s_test_011 = s_test[:,3]
                        s_test_auc_011 = roc_auc_score(s_test_011, p_test)
                        s_test_auc_011 = max(1-s_test_auc_011, s_test_auc_011)
                        s_test_ortho_aucs_011.append(s_test_auc_011)
                        
                        s_test_100 = s_test[:,4]
                        s_test_auc_100 = roc_auc_score(s_test_100, p_test)
                        s_test_auc_100 = max(1-s_test_auc_100, s_test_auc_100)
                        s_test_ortho_aucs_100.append(s_test_auc_100)
                        
                        s_test_101 = s_test[:,5]
                        s_test_auc_101 = roc_auc_score(s_test_101, p_test)
                        s_test_auc_101 = max(1-s_test_auc_101, s_test_auc_101)
                        s_test_ortho_aucs_101.append(s_test_auc_101)
                        
                        s_test_110 = s_test[:,6]
                        s_test_auc_110 = roc_auc_score(s_test_110, p_test)
                        s_test_auc_110 = max(1-s_test_auc_110, s_test_auc_110)
                        s_test_ortho_aucs_110.append(s_test_auc_110)
                        
                        s_test_111 = s_test[:,7]
                        s_test_auc_111 = roc_auc_score(s_test_111, p_test)
                        s_test_auc_111 = max(1-s_test_auc_111, s_test_auc_111)
                        s_test_ortho_aucs_111.append(s_test_auc_111)
                        
                    y_test_auc_means.append(np.mean(y_test_ortho_aucs))
                    y_test_auc_stdvs.append(np.std(y_test_ortho_aucs, ddof=0))
                    
                    s_test_auc_means_000.append(np.mean(s_test_ortho_aucs_000))                    
                    s_test_auc_means_001.append(np.mean(s_test_ortho_aucs_001))                    
                    s_test_auc_means_010.append(np.mean(s_test_ortho_aucs_010))
                    s_test_auc_means_011.append(np.mean(s_test_ortho_aucs_011))
                    s_test_auc_means_100.append(np.mean(s_test_ortho_aucs_100))                    
                    s_test_auc_means_101.append(np.mean(s_test_ortho_aucs_101))                    
                    s_test_auc_means_110.append(np.mean(s_test_ortho_aucs_110))
                    s_test_auc_means_111.append(np.mean(s_test_ortho_aucs_111))
                    
                    s_test_auc_stdvs_000.append(np.std(s_test_ortho_aucs_000, ddof=0))
                    s_test_auc_stdvs_001.append(np.std(s_test_ortho_aucs_001, ddof=0))                    
                    s_test_auc_stdvs_010.append(np.std(s_test_ortho_aucs_010, ddof=0))                    
                    s_test_auc_stdvs_011.append(np.std(s_test_ortho_aucs_011, ddof=0))
                    s_test_auc_stdvs_100.append(np.std(s_test_ortho_aucs_100, ddof=0))
                    s_test_auc_stdvs_101.append(np.std(s_test_ortho_aucs_101, ddof=0))                    
                    s_test_auc_stdvs_110.append(np.std(s_test_ortho_aucs_110, ddof=0))                    
                    s_test_auc_stdvs_111.append(np.std(s_test_ortho_aucs_111, ddof=0))
                    
                results[method][dataset]["y_test_auc_means"] = np.array(y_test_auc_means)
                results[method][dataset]["y_test_auc_stdvs"] = np.array(y_test_auc_stdvs)
                
                results[method][dataset]["s_test_auc_means_000"] = np.array(s_test_auc_means_000)
                results[method][dataset]["s_test_auc_means_001"] = np.array(s_test_auc_means_001)
                results[method][dataset]["s_test_auc_means_010"] = np.array(s_test_auc_means_010)
                results[method][dataset]["s_test_auc_means_011"] = np.array(s_test_auc_means_011)
                results[method][dataset]["s_test_auc_means_100"] = np.array(s_test_auc_means_100)
                results[method][dataset]["s_test_auc_means_101"] = np.array(s_test_auc_means_101)
                results[method][dataset]["s_test_auc_means_110"] = np.array(s_test_auc_means_110)
                results[method][dataset]["s_test_auc_means_111"] = np.array(s_test_auc_means_111)
                
                results[method][dataset]["s_test_auc_stdvs_000"] = np.array(s_test_auc_stdvs_000)
                results[method][dataset]["s_test_auc_stdvs_001"] = np.array(s_test_auc_stdvs_001)
                results[method][dataset]["s_test_auc_stdvs_010"] = np.array(s_test_auc_stdvs_010)                
                results[method][dataset]["s_test_auc_stdvs_011"] = np.array(s_test_auc_stdvs_011)
                results[method][dataset]["s_test_auc_stdvs_100"] = np.array(s_test_auc_stdvs_100)
                results[method][dataset]["s_test_auc_stdvs_101"] = np.array(s_test_auc_stdvs_101)
                results[method][dataset]["s_test_auc_stdvs_110"] = np.array(s_test_auc_stdvs_110)                
                results[method][dataset]["s_test_auc_stdvs_111"] = np.array(s_test_auc_stdvs_111)
    return results

def make_2d_fig(n_folds, orthogonalities, methods, datasets):
    results = get_results(n_folds, orthogonalities, methods, datasets)
    fig, axs = plt.subplots(5,2,dpi=200, figsize=(16,3.5*6), sharey=False, sharex=False)
    i = 0
    for dataset in datasets:
        if "multiple" not in dataset:
            suptitle = dataset.split("_")[0].capitalize()  + " (" + dataset.split("_")[1].capitalize() + ")"
            axs = axs.ravel()
            axs[i*2+0].grid()
            axs[i*2+1].grid()
            clr_ctr = 1
            for method in methods:
                if method in ["local_sub"]:
                    y_color = "C0"
                    s_color = "C0"
                    rp_color ="C0"

                    label_y_test = method.split("_")[0] + "_y"
                    label_s_test = method.split("_")[0] + "_s"

                    y_test_auc_means = results[method][dataset]["y_test_auc_means"]
                    s_test_auc_means = results[method][dataset]["s_test_auc_means"]
                    y_test_auc_stdvs = results[method][dataset]["y_test_auc_stdvs"]
                    s_test_auc_stdvs = results[method][dataset]["s_test_auc_stdvs"]


                    # means


                    axs[i*2+0].plot(
                        orthogonalities, y_test_auc_means, marker="^", markersize=5, lw=3,zorder=3,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
                        linestyle="-", label="AUC" + r"$_{\rm Y}$", c=y_color, markeredgewidth=2, #markeredgecolor="black",
                    )
                    axs[i*2+0].plot(
                        orthogonalities, s_test_auc_means, marker="v", markersize=5, lw=3,zorder=3,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
                        linestyle="-", label="AUC" + r"$_{\rm S}$", c=s_color, markeredgewidth=2, #markeredgecolor="black",
                    )



                    # stdvs

                    axs[i*2+0].fill_between(orthogonalities, y_test_auc_means+y_test_auc_stdvs, y_test_auc_means-y_test_auc_stdvs, alpha=0.5, color=y_color)
                    axs[i*2+0].fill_between(orthogonalities, s_test_auc_means+s_test_auc_stdvs, s_test_auc_means-s_test_auc_stdvs, alpha=0.5, color=s_color)
                    axs[i*2+0].set_xticks(np.linspace(0,1,11))
                    axs[i*2+1].plot(
                        s_test_auc_means, y_test_auc_means, marker="d", markersize=5, lw=3,zorder=3,
                        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
                        linestyle="-", label="Our method", c=rp_color, markeredgewidth=2, #markeredgecolor="black",
                    )
                else:
                    y_test_auc_mean = results[method][dataset]["y_test_auc_mean"]
                    s_test_auc_mean = results[method][dataset]["s_test_auc_mean"]
                    if method=="faht":
                        label = "FAHT"
                    else:
                        label = method.split("_")[0].capitalize()+ r"$_{\rm " + method.split("_")[1].capitalize() + "}$"
                    axs[i*2+1].scatter(s_test_auc_mean, y_test_auc_mean, label=label, zorder=5, s=10**2, edgecolor="black", lw=1, marker="o", c="C"+str(clr_ctr))
                    clr_ctr += 1


            if i==4:
                axs[i*2+0].set_xlabel(r"$\Theta$", fontsize=13)
                axs[i*2+1].set_xlabel("AUC" + r"$_{\rm S}$" , fontsize=11)

            axs[i*2+0].set_ylabel("Measure", fontsize=11)
            axs[i*2+1].set_ylabel("AUC" + r"$_{\rm Y}$", fontsize=11)
            axs[i*2+1].yaxis.set_tick_params(labelright=True)
            axs[i*2+1].yaxis.tick_right()
            axs[i*2+1].yaxis.set_label_position("right")

            ymin, ymax = axs[i*2+0].get_ylim()
            axs[i*2+1].set_ylim(ymin, ymax)

            if i==0:
                axs[i*2+0].legend(title="Our method", fontsize=8, loc="lower left")
                axs[i*2+1].legend(fontsize=8, loc="lower right")
            axs[i*2+0].set_title(suptitle, loc="left", fontsize=11)
            i += 1
    plt.subplots_adjust(
        top=.95,
        wspace=0,
        hspace=0.3
    )
    plt.savefig("2d_results_0"+".svg", format="svg")
    plt.show()
    
    # adult_multiple_1
    i=0
    method="local_sub"
    dataset="adult_multiple_1"
    fig, axs = plt.subplots(1,2,dpi=200, figsize=(16,3.5*1), sharey=False, sharex=False)
    suptitle = dataset.split("_")[0].capitalize()  + " (" + dataset.split("_")[1].capitalize() + ")"
    axs = axs.ravel()
    axs[i*2+0].grid()
    axs[i*2+1].grid()
    clr_ctr = 1
    y_color = "C0"
    s_color = "C0"
    rp_color ="C0"
    label_y_test = method.split("_")[0] + "_y"
    label_s_test = method.split("_")[0] + "_s"
    y_test_auc_means = results[method][dataset]["y_test_auc_means"]
    y_test_auc_stdvs = results[method][dataset]["y_test_auc_stdvs"]
    s_test_auc_means_0 = results[method][dataset]["s_test_auc_means_0"]
    s_test_auc_stdvs_0 = results[method][dataset]["s_test_auc_stdvs_0"]
    s_test_auc_means_1 = results[method][dataset]["s_test_auc_means_1"]
    s_test_auc_stdvs_1 = results[method][dataset]["s_test_auc_stdvs_1"]
    
    # means
    axs[i*2+0].plot(
        orthogonalities, y_test_auc_means, marker="^", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm Y}$", c=y_color, markeredgewidth=2, #markeredgecolor="black",
    )
    axs[i*2+0].plot(
        orthogonalities, s_test_auc_means_0, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (R)", c="C1", markeredgewidth=2, #markeredgecolor="black",
    )
    axs[i*2+0].plot(
        orthogonalities, s_test_auc_means_1, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (G)", c="C2", markeredgewidth=2, #markeredgecolor="black",
    )
    # stdvs
    axs[i*2+0].fill_between(orthogonalities, y_test_auc_means+y_test_auc_stdvs, y_test_auc_means-y_test_auc_stdvs, alpha=0.5, color=y_color)
    axs[i*2+0].fill_between(orthogonalities, s_test_auc_means_0+s_test_auc_stdvs_0, s_test_auc_means_0-s_test_auc_stdvs_0, alpha=0.5, color="C1")
    axs[i*2+0].fill_between(orthogonalities, s_test_auc_means_1+s_test_auc_stdvs_1, s_test_auc_means_1-s_test_auc_stdvs_1, alpha=0.5, color="C2")
    
    axs[i*2+0].set_xticks(np.linspace(0,1,11))
    axs[i*2+0].set_xlabel(r"$\Theta$", fontsize=13)
    axs[i*2+0].set_ylabel("Measure", fontsize=11)
    axs[i*2+0].legend(fontsize=8, loc="upper right")#, title="Our method")
    axs[i*2+0].set_title(suptitle, loc="left", fontsize=11)

    method="local_sub"
    dataset="adult_multiple_2"
    suptitle = dataset.split("_")[0].capitalize()  + " (Intersectional)"
    label_y_test = method.split("_")[0] + "_y"
    label_s_test = method.split("_")[0] + "_s"
    y_test_auc_means = results[method][dataset]["y_test_auc_means"]
    y_test_auc_stdvs = results[method][dataset]["y_test_auc_stdvs"]
    
    s_test_auc_means_00 = results[method][dataset]["s_test_auc_means_00"]
    s_test_auc_means_01 = results[method][dataset]["s_test_auc_means_01"]
    s_test_auc_means_10 = results[method][dataset]["s_test_auc_means_10"]
    s_test_auc_means_11 = results[method][dataset]["s_test_auc_means_11"]
    
    s_test_auc_stdvs_00 = results[method][dataset]["s_test_auc_stdvs_00"]
    s_test_auc_stdvs_01 = results[method][dataset]["s_test_auc_stdvs_01"]
    s_test_auc_stdvs_10 = results[method][dataset]["s_test_auc_stdvs_10"]
    s_test_auc_stdvs_11 = results[method][dataset]["s_test_auc_stdvs_11"]
    
    # means
    axs[i*2+1].plot(
        orthogonalities, y_test_auc_means, marker="^", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm Y}$", c=y_color, markeredgewidth=2, #markeredgecolor="black",
    )
    
    axs[i*2+1].plot(
        orthogonalities, s_test_auc_means_00, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (NWF)", c="C1", markeredgewidth=2, #markeredgecolor="black",
    )
    axs[i*2+1].plot(
        orthogonalities, s_test_auc_means_01, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (NWM)", c="C2", markeredgewidth=2, #markeredgecolor="black",
    )
    axs[i*2+1].plot(
        orthogonalities, s_test_auc_means_10, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (WF)", c="C3", markeredgewidth=2, #markeredgecolor="black",
    )
    axs[i*2+1].plot(
        orthogonalities, s_test_auc_means_11, marker="v", markersize=5, lw=3,zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
        linestyle="-", label="AUC" + r"$_{\rm S}$ (WM)", c="C4", markeredgewidth=2, #markeredgecolor="black",
    )
    
    # stdvs
    axs[i*2+1].fill_between(orthogonalities, y_test_auc_means+y_test_auc_stdvs, y_test_auc_means-y_test_auc_stdvs, alpha=0.5, color=y_color)
    axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_00+s_test_auc_stdvs_00, s_test_auc_means_00-s_test_auc_stdvs_00, alpha=0.5, color="C1")
    axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_01+s_test_auc_stdvs_01, s_test_auc_means_01-s_test_auc_stdvs_01, alpha=0.5, color="C2")
    axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_10+s_test_auc_stdvs_10, s_test_auc_means_10-s_test_auc_stdvs_10, alpha=0.5, color="C3")
    axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_11+s_test_auc_stdvs_11, s_test_auc_means_11-s_test_auc_stdvs_11, alpha=0.5, color="C4")
    
    axs[i*2+1].set_xticks(np.linspace(0,1,11))
    axs[i*2+1].set_xlabel(r"$\Theta$", fontsize=13)
    axs[i*2+1].set_ylabel("Measure", fontsize=11)
    axs[i*2+1].legend(fontsize=8, loc="upper right")#, title="Our method")
    axs[i*2+1].set_title(suptitle, loc="left", fontsize=11)
    
    
    axs[i*2+1].yaxis.set_tick_params(labelright=True)
    axs[i*2+1].yaxis.tick_right()
    axs[i*2+1].yaxis.set_label_position("right")
    ymin, ymax = axs[i*2+0].get_ylim()
    axs[i*2+1].set_ylim(ymin, ymax)
    axs[i*2+1].legend(fontsize=8, loc="upper right")
    
    plt.subplots_adjust(
        top=.95,
        wspace=0,
        hspace=0.3
    )
    plt.savefig("2d_results_1"+".svg", format="svg")
    plt.show()
    
#     # recidivism_multiple_1
#     i=0
#     method="local_sub"
#     dataset="recidivism_multiple_1"
#     fig, axs = plt.subplots(1,2,dpi=200, figsize=(16,3.5*1), sharey=False, sharex=False)
#     suptitle = dataset.split("_")[0].capitalize()  + " (" + dataset.split("_")[1].capitalize() + ")"
#     axs = axs.ravel()
#     axs[i*2+0].grid()
#     axs[i*2+1].grid()
#     clr_ctr = 1
#     y_color = "C0"
#     s_color = "C0"
#     rp_color ="C0"
#     label_y_test = method.split("_")[0] + "_y"
#     label_s_test = method.split("_")[0] + "_s"
#     y_test_auc_means = results[method][dataset]["y_test_auc_means"]
#     y_test_auc_stdvs = results[method][dataset]["y_test_auc_stdvs"]
#     s_test_auc_means_0 = results[method][dataset]["s_test_auc_means_0"]
#     s_test_auc_stdvs_0 = results[method][dataset]["s_test_auc_stdvs_0"]
#     s_test_auc_means_1 = results[method][dataset]["s_test_auc_means_1"]
#     s_test_auc_stdvs_1 = results[method][dataset]["s_test_auc_stdvs_1"]
#     s_test_auc_means_2 = results[method][dataset]["s_test_auc_means_2"]
#     s_test_auc_stdvs_2 = results[method][dataset]["s_test_auc_stdvs_2"]
    
#     # means
#     axs[i*2+0].plot(
#         orthogonalities, y_test_auc_means, marker="^", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm Y}$", c=y_color, markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+0].plot(
#         orthogonalities, s_test_auc_means_0, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (A)", c="C1", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+0].plot(
#         orthogonalities, s_test_auc_means_1, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (R)", c="C2", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+0].plot(
#         orthogonalities, s_test_auc_means_2, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (G)", c="C3", markeredgewidth=2, #markeredgecolor="black",
#     )
#     # stdvs
#     axs[i*2+0].fill_between(orthogonalities, y_test_auc_means+y_test_auc_stdvs, y_test_auc_means-y_test_auc_stdvs, alpha=0.5, color=y_color)
#     axs[i*2+0].fill_between(orthogonalities, s_test_auc_means_0+s_test_auc_stdvs_0, s_test_auc_means_0-s_test_auc_stdvs_0, alpha=0.5, color="C1")
#     axs[i*2+0].fill_between(orthogonalities, s_test_auc_means_1+s_test_auc_stdvs_1, s_test_auc_means_1-s_test_auc_stdvs_1, alpha=0.5, color="C2")
#     axs[i*2+0].fill_between(orthogonalities, s_test_auc_means_2+s_test_auc_stdvs_2, s_test_auc_means_2-s_test_auc_stdvs_2, alpha=0.5, color="C3")
    
#     axs[i*2+0].set_xticks(np.linspace(0,1,11))
#     axs[i*2+0].set_xlabel(r"$\Theta$", fontsize=13)
#     axs[i*2+0].set_ylabel("Measure", fontsize=11)
#     axs[i*2+0].legend(fontsize=8, loc="upper right")#, title="Our method")
#     axs[i*2+0].set_title(suptitle, loc="left", fontsize=11)

#     method="local_sub"
#     dataset="recidivism_multiple_2"
#     suptitle = dataset.split("_")[0].capitalize()  + " (Intersectional)"
#     label_y_test = method.split("_")[0] + "_y"
#     label_s_test = method.split("_")[0] + "_s"
#     y_test_auc_means = results[method][dataset]["y_test_auc_means"]
#     y_test_auc_stdvs = results[method][dataset]["y_test_auc_stdvs"]
    
#     s_test_auc_means_000 = results[method][dataset]["s_test_auc_means_000"]
#     s_test_auc_means_001 = results[method][dataset]["s_test_auc_means_001"]
#     s_test_auc_means_010 = results[method][dataset]["s_test_auc_means_010"]
#     s_test_auc_means_011 = results[method][dataset]["s_test_auc_means_011"]
#     s_test_auc_means_100 = results[method][dataset]["s_test_auc_means_100"]
#     s_test_auc_means_101 = results[method][dataset]["s_test_auc_means_101"]
#     s_test_auc_means_110 = results[method][dataset]["s_test_auc_means_110"]
#     s_test_auc_means_111 = results[method][dataset]["s_test_auc_means_111"]
    
#     s_test_auc_stdvs_000 = results[method][dataset]["s_test_auc_stdvs_000"]
#     s_test_auc_stdvs_001 = results[method][dataset]["s_test_auc_stdvs_001"]
#     s_test_auc_stdvs_010 = results[method][dataset]["s_test_auc_stdvs_010"]
#     s_test_auc_stdvs_011 = results[method][dataset]["s_test_auc_stdvs_011"]
#     s_test_auc_stdvs_100 = results[method][dataset]["s_test_auc_stdvs_100"]
#     s_test_auc_stdvs_101 = results[method][dataset]["s_test_auc_stdvs_101"]
#     s_test_auc_stdvs_110 = results[method][dataset]["s_test_auc_stdvs_110"]
#     s_test_auc_stdvs_111 = results[method][dataset]["s_test_auc_stdvs_111"]
    
#     # means
#     axs[i*2+1].plot(
#         orthogonalities, y_test_auc_means, marker="^", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm Y}$", c=y_color, markeredgewidth=2, #markeredgecolor="black",
#     )
    
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_000, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (ONWF)", c="C1", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_001, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (ONWM)", c="C2", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_010, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (OWF)", c="C3", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_011, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (OWM)", c="C4", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_100, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (YNWF)", c="C5", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_101, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (YNWM)", c="C6", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_110, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (YWF)", c="C7", markeredgewidth=2, #markeredgecolor="black",
#     )
#     axs[i*2+1].plot(
#         orthogonalities, s_test_auc_means_111, marker="v", markersize=5, lw=3,zorder=3,
#         path_effects=[pe.Stroke(linewidth=3.5, foreground="k"), pe.Normal()],
#         linestyle="-", label="AUC" + r"$_{\rm S}$ (YWM)", c="C8", markeredgewidth=2, #markeredgecolor="black",
#     )
    
#     # stdvs
#     axs[i*2+1].fill_between(orthogonalities, y_test_auc_means+y_test_auc_stdvs, y_test_auc_means-y_test_auc_stdvs, alpha=0.5, color=y_color)
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_000+s_test_auc_stdvs_000, s_test_auc_means_000-s_test_auc_stdvs_000, alpha=0.5, color="C1")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_001+s_test_auc_stdvs_001, s_test_auc_means_001-s_test_auc_stdvs_001, alpha=0.5, color="C2")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_010+s_test_auc_stdvs_010, s_test_auc_means_010-s_test_auc_stdvs_010, alpha=0.5, color="C3")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_011+s_test_auc_stdvs_011, s_test_auc_means_011-s_test_auc_stdvs_011, alpha=0.5, color="C4")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_100+s_test_auc_stdvs_100, s_test_auc_means_100-s_test_auc_stdvs_100, alpha=0.5, color="C5")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_101+s_test_auc_stdvs_101, s_test_auc_means_101-s_test_auc_stdvs_101, alpha=0.5, color="C6")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_110+s_test_auc_stdvs_110, s_test_auc_means_110-s_test_auc_stdvs_110, alpha=0.5, color="C7")
#     axs[i*2+1].fill_between(orthogonalities, s_test_auc_means_111+s_test_auc_stdvs_111, s_test_auc_means_111-s_test_auc_stdvs_111, alpha=0.5, color="C8")
    
#     axs[i*2+1].set_xticks(np.linspace(0,1,11))
#     axs[i*2+1].set_xlabel(r"$\Theta$", fontsize=13)
#     axs[i*2+1].set_ylabel("Measure", fontsize=11)
#     axs[i*2+1].legend(fontsize=8, loc="upper right")#, title="Our method")
#     axs[i*2+1].set_title(suptitle, loc="left", fontsize=11)
    
    
#     axs[i*2+1].yaxis.set_tick_params(labelright=True)
#     axs[i*2+1].yaxis.tick_right()
#     axs[i*2+1].yaxis.set_label_position("right")
#     ymin, ymax = axs[i*2+0].get_ylim()
#     axs[i*2+1].set_ylim(ymin, ymax)
#     axs[i*2+1].legend(fontsize=8, loc="upper right")
    
#     plt.subplots_adjust(
#         top=.95,
#         wspace=0,
#         hspace=0.3
#     )
#     plt.savefig("2d_results_2"+".svg", format="svg")
#     plt.show()
    
def get_measures(n_folds, orthogonalities, dataset):
    method = "local_sub"
    experiments_df = joblib.load("experiments_df.pkl")
    all_accuracies = []
    all_dem_paries = []
    all_eq_odd_tprs = []
    all_eq_odd_fprs = []
    for orthogonality in orthogonalities:
        ortho_accuracies = []
        ortho_dem_paries = []
        ortho_eq_odd_tprs = []
        ortho_eq_odd_fprs = []
        for fold in range(n_folds):
            loc = \
                (experiments_df["fold"]==fold ) & \
                (experiments_df["method"]==method ) & \
                (experiments_df["dataset"]==dataset ) & \
                (experiments_df["orthogonality"]==orthogonality )
            y_test, s_test, p_test = experiments_df.loc[loc, ["y_test", "s_test", "p_test"]].values[0]
            p_cutoffs = np.quantile(p_test, np.linspace(0, 1, 11).round(2))
            accuracies = []
            dem_paries = []
            eq_odd_tprs = []
            eq_odd_fprs = []
            for p_cutoff in p_cutoffs:
                y_pred = (p_test >= p_cutoff).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
                dem_pary = demographic_parity(s_test, y_pred)
                eq_odd_tpr = equal_odds_tpr(s_test, y_pred, y_test)
                eq_odd_fpr = equal_odds_fpr(s_test, y_pred, y_test)

                accuracies.append(accuracy)
                dem_paries.append(dem_pary)
                eq_odd_tprs.append(eq_odd_tpr)
                eq_odd_fprs.append(eq_odd_fpr)

            ortho_accuracies.append(accuracies)
            ortho_dem_paries.append(dem_paries)
            ortho_eq_odd_tprs.append(eq_odd_tprs)
            ortho_eq_odd_fprs.append(eq_odd_fprs)

        ortho_accuracies = np.mean(ortho_accuracies, axis=0)
        ortho_dem_paries = np.mean(ortho_dem_paries, axis=0)
        ortho_eq_odd_tprs = np.mean(ortho_eq_odd_tprs, axis=0)
        ortho_eq_odd_fprs = np.mean(ortho_eq_odd_fprs, axis=0)

        all_accuracies.append(ortho_accuracies)
        all_dem_paries.append(ortho_dem_paries)
        all_eq_odd_tprs.append(ortho_eq_odd_tprs)
        all_eq_odd_fprs.append(ortho_eq_odd_fprs)
    all_accuracies = np.array(all_accuracies)
    all_dem_paries = np.array(all_dem_paries)
    all_eq_odd_tprs = np.array(all_eq_odd_tprs)
    all_eq_odd_fprs = np.array(all_eq_odd_fprs)
    all_eq_odd_both = abs(all_eq_odd_tprs - all_eq_odd_fprs)
    
    return all_accuracies, all_dem_paries, all_eq_odd_tprs, all_eq_odd_fprs, all_eq_odd_both
    
def make_3d_fig(n_folds, orthogonalities):
    
    datasets = [
        "bank_age",
        "adult_race", "adult_gender",
        "recidivism_race", "recidivism_gender"
    ]
    
    fig = make_subplots(
        rows=5, cols=3,
        specs=[
            [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
            [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
            [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
            [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
            [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
        ],
        vertical_spacing = 0,
        horizontal_spacing = 0,
        subplot_titles=("(a)", "(b)", "(c)")
    )
    i = 1
    for dataset in datasets:
        
        all_accuracies, all_dem_paries, all_eq_odd_tprs, all_eq_odd_fprs, all_eq_odd_both = get_measures(n_folds, orthogonalities, dataset)
    
        y = np.array(np.linspace(0,1,11).tolist()*11).ravel()
        z = all_accuracies.ravel()
        fairness_measures = [
            all_dem_paries,
            all_eq_odd_tprs,
            all_eq_odd_both,
        ]
        j = 1
        for x in fairness_measures:
            x = x.ravel()
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, opacity=.75, colorscale="Bluered", intensity=z, showscale=True if i==0 else False, colorbar=dict(title='Accuracy'),
                    lighting=dict(ambient=.95, diffuse=.75, roughness = 0.75, specular=0.75, fresnel=.75),
                ),

                row=i,
                col=j,
            )
            j+= 1
        i+=1
    camera = dict(eye=dict(x=1.5, y=-1.5, z=-0.2))
    fig.update_layout(
        scene1 = dict(
            xaxis = dict(title = "x: Dem. parity", tickangle = 0),
            yaxis = dict(title = "y: Threshold", tickangle = 0),
            zaxis = dict(title = "z: Accuracy", tickangle = 0),
            camera = camera,
        ),

        scene2 = dict(
            xaxis = dict(title = "x: Eq. opp.", tickangle = 0),
            yaxis = dict(title = "y: Threshold", tickangle = 0),
            zaxis = dict(title = "z: Accuracy", tickangle = 0),
            camera = camera,
        ),

        scene3 = dict(
            xaxis = dict(title = "x: Eq. odds", tickangle = 0),
            yaxis = dict(title = "y: Threshold", tickangle = 0),
            zaxis = dict(title = "z: Accuracy", tickangle = 0),
            camera = camera,
        ),

#         scene4 = dict(
#             xaxis = dict(title = "x: Eq. Opp. (FPR)", tickangle = 0),
#             yaxis = dict(title = "y: Threshold", tickangle = 0),
#             zaxis = dict(title = "z: Accuracy", tickangle = 0),
#             camera = camera,
#         ),
        width=900,
        height=1500,
        autosize=True,
    )
    fig.write_html("3d_results.html")
    fig.write_image("3d_results.svg")
    fig.show()
    
def make_corr_table_spearman(n_folds, orthogonalities, dataset):
    
    all_accuracies, all_dem_paries, all_eq_odd_tprs, all_eq_odd_fprs, all_eq_odd_both = get_measures(n_folds, orthogonalities, dataset)
    
    method = "local_sub"
    results = get_results(n_folds, orthogonalities, [method], [dataset])
    
    data = []
    row = []
    p_val_th = 0.05
    
    y_test_auc_means = results[method][dataset]["y_test_auc_means"]
    s_test_auc_means = results[method][dataset]["s_test_auc_means"]
    
    for i in range(1,10):
        corr, pval = pearsonr(y_test_auc_means, all_accuracies[:, i])
        cell = "$"+str(round(corr, 3))+"$" if pval>p_val_th else "$\mathbf{"+str(round(corr, 3))+"}$"
        row.append(cell)
    data.append(row)

    fairness_measures = [
        all_dem_paries,
        all_eq_odd_tprs,
        all_eq_odd_both,
    ]
    for fairness_measure in fairness_measures:
        row = []
        for i in range(1,10):
            corr, pval = spearmanr(s_test_auc_means, fairness_measure[:, i])
            cell = "$"+str(round(corr, 3))+"$" if pval>p_val_th else "$\mathbf{"+str(round(corr, 3))+"}$"
            row.append(cell)
        data.append(row)


    data = np.array(data)

    display(pd.DataFrame(
        index=["Accuracy", "Dem. Parity", "Eq. Opp.", "Eq. Odds", ],
        columns=["0.1", "0.2", "0.3", "0.4", "0.5" ,"0.6", "0.7", "0.8", "0.9"],
        data=data,
    ))

def make_corr_table_pearson(n_folds, orthogonalities, dataset):
    
    all_accuracies, all_dem_paries, all_eq_odd_tprs, all_eq_odd_fprs, all_eq_odd_both = get_measures(n_folds, orthogonalities, dataset)
    
    method = "local_sub"
    results = get_results(n_folds, orthogonalities, [method], [dataset])
    
    data = []
    row = []
    p_val_th = 0.05
    
    y_test_auc_means = results[method][dataset]["y_test_auc_means"]
    s_test_auc_means = results[method][dataset]["s_test_auc_means"]
    
#     for i in range(1,10):
#         corr, pval = pearsonr(y_test_auc_means, all_accuracies[:, i])
#         cell = "$"+str(round(corr, 3))+"$" if pval>p_val_th else "$\mathbf{"+str(round(corr, 3))+"}$"
#         row.append(cell)
#     data.append(row)

    fairness_measures = [
        all_dem_paries,
        all_eq_odd_tprs,
        all_eq_odd_both,
    ]
    for fairness_measure in fairness_measures:
        row = []
        for i in range(1,10):
            corr, pval = pearsonr(s_test_auc_means, fairness_measure[:, i])
            cell = "$"+str(round(corr, 3))+"$" if pval>p_val_th else "$\mathbf{"+str(round(corr, 3))+"}$"
            row.append(cell)
        data.append(row)


    data = np.array(data)
    df = pd.DataFrame(
        index=["Dem. Parity", "Eq. Opp.", "Eq. Odds", ],
        columns=["0.1", "0.2", "0.3", "0.4", "0.5" ,"0.6", "0.7", "0.8", "0.9"],
        data=data,
    )
    display(df)
    return df