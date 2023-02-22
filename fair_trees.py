import numpy as np
import pandas as pd
import multiprocessing
from scipy import stats as st
from copy import deepcopy as copy
from joblib import delayed, Parallel
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix

def demographic_parity_score(s, y_pred):
    y_pred = np.array(y_pred).ravel()
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
        
    demographic_parities = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            demographic_parities.append(1) 
        
        else:
            dem_par = 0
            for s_unique in np.unique(s[:, s_column]):
                s_cond_0 = (s[:, s_column]==s_unique)
                s_cond_1 = (s[:, s_column]!=s_unique)
                
                dem_par = max(
                    dem_par, 
                    abs(
                        (s_cond_0 & (y_pred==1)).sum() / s_cond_0.sum() - \
                        (s_cond_1 & (y_pred==1)).sum() / s_cond_1.sum()
                    )
                )

            demographic_parities.append(dem_par)
        
    return demographic_parities[0] if len(demographic_parities)==1 else demographic_parities

def equal_opportunity_score(s, y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    
    equal_opportunities = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            equal_opportunities.append(1) 
        else:
            eq_op = 0
            for s_unique in np.unique(s[:, s_column]):
                s_cond_0 = (s[:, s_column]==s_unique)
                s_cond_1 = (s[:, s_column]!=s_unique)

                tn_s_0, fp_s_0, fn_s_0, tp_s_0 = confusion_matrix(y_true[s_cond_0], y_pred[s_cond_0], labels=[0,1]).ravel()
                tn_s_1, fp_s_1, fn_s_1, tp_s_1 = confusion_matrix(y_true[s_cond_1], y_pred[s_cond_1], labels=[0,1]).ravel()
                
                pos_s_0 = (tp_s_0 + fn_s_0)
                pos_s_1 = (tp_s_1 + fn_s_1)
                
                if (pos_s_0==0) and (pos_s_1==0):
                    eq_op = np.nan
                    break
                    
                elif (
                    ((pos_s_0>0) and (pos_s_1==0)) or \
                    ((pos_s_0==0) and (pos_s_1>1))
                ):
                    eq_op = 1
                    break
                
                else:
                    tpr_s_0 = tp_s_0 / pos_s_0
                    tpr_s_1 = tp_s_1 / pos_s_1
                    eq_op = max(eq_op, abs(tpr_s_0 - tpr_s_1))

            equal_opportunities.append(eq_op)
            
    return equal_opportunities[0] if len(equal_opportunities)==1 else equal_opportunities

def equalized_odds_score(s, y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    
    equalized_odds = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            equalized_odds.append(1) 
        else:
            eq_odds = 0
            for s_unique in np.unique(s[:, s_column]):
                s_cond_0 = (s[:, s_column]==s_unique)
                s_cond_1 = (s[:, s_column]!=s_unique)

                tn_s_0, fp_s_0, fn_s_0, tp_s_0 = confusion_matrix(y_true[s_cond_0], y_pred[s_cond_0], labels=[0,1]).ravel()
                tn_s_1, fp_s_1, fn_s_1, tp_s_1 = confusion_matrix(y_true[s_cond_1], y_pred[s_cond_1], labels=[0,1]).ravel()
                
                pos_s_0 = (tp_s_0 + fn_s_0)
                pos_s_1 = (tp_s_1 + fn_s_1)
                neg_s_0 = (tn_s_0 + fp_s_0)
                neg_s_1 = (tn_s_1 + fp_s_1)
                
                if (pos_s_0==0) and (pos_s_1==0):
                    eq_odds = np.nan
                    break
                    
                elif (neg_s_0==0) and (neg_s_1==0):
                    eq_odds = np.nan
                    break
                
                elif (
                    ((pos_s_0>0) and (pos_s_1==0)) or \
                    ((pos_s_0==0) and (pos_s_1>1)) or \
                    ((neg_s_0>0) and (neg_s_1==0)) or \
                    ((neg_s_0==0) and (neg_s_1>1))
                ):
                    eq_odds = 1
                    break
        
                else:
                    tpr_s_0 = tp_s_0 / pos_s_0
                    tpr_s_1 = tp_s_1 / pos_s_1
                    fpr_s_0 = fp_s_0 / neg_s_0
                    fpr_s_1 = fp_s_1 / neg_s_1
                    eq_odds = max(eq_odds, abs(abs(tpr_s_0-tpr_s_1) - abs(fpr_s_0-fpr_s_1)))

            equalized_odds.append(eq_odds)
            
    return equalized_odds[0] if len(equalized_odds)==1 else equalized_odds

def sensitive_auc_score(s, y_prob):
    y_prob = np.array(y_prob)
    s = np.array(s)
    if len(s.shape)==1:
        s = s.reshape(-1,1)
    
    sensitive_aucs = []
    for s_column in range(s.shape[1]):
        if len(np.unique(s[:, s_column]))==1:
            sensitive_aucs.append(1) 
        else:
            sens_auc = 0
            for s_unique in np.unique(s[:, s_column]):
                s_bool = (s[:, s_column]==s_unique)
                auc = roc_auc_score(s_bool, y_prob)
                auc = max(1-auc, auc)
                sens_auc = max(sens_auc, auc)
            sensitive_aucs.append(sens_auc)
    
    return sensitive_aucs[0] if len(sensitive_aucs)==1 else sensitive_aucs

class FairDecisionTreeClassifier():
    def __init__(
        self,
        n_bins=256,
        max_depth=7,
        bootstrap=False,
        random_state=42,
        orthogonality=.5, 
        max_features=1.0,
        oob_pruning=True,
        criterion="scaff",
        min_samples_leaf=3,
        min_samples_split=7, 
        kamiran_method=None,
        split_info_norm=None,
        sampling_proportion=1.0, 
        
    ):
        self.is_fit = False
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.oob_pruning = oob_pruning
        self.split_info_norm = split_info_norm
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sampling_proportion = sampling_proportion
        self.n_bins = np.inf if (n_bins is None) else n_bins
        self.max_depth = np.inf if (max_depth is None) else (max_depth)
        self.kamiran_method = "sum" if (kamiran_method is None) else (kamiran_method)
                                
    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> pandas.df: may contain int float str
        y -> one_dim pandas.df or np.array: only binary int {0, 1}
        s -> any_dim pandas.df or np.array: only str
        kwargs -> for compatibility with scikit-learn: fit_params in cross_validate()
        """
        
        # self.X_source = copy(X)
        # we use pandas to sort out between the numerical and categorical variables
        if "pandas" not in str(type(X)):
            X = pd.DataFrame(X)
            
        self.X = X
        self.y = np.array(y).astype(int)
        
        # for compatibility with scikit-learn since sklearn fit() methods only take X, y
        self.s = np.array(s).astype(str) if (
            "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(str)
        )
        
        if (len(self.X)!=len(self.y)) or (len(self.X)!=len(self.s)) or (len(self.y)!=len(self.s)):
            raise Exception("X, y, and s lenghts do not match")    
        if len(self.y.shape)==1 or len(self.y.ravel())==len(self.X):
            self.y = self.y.ravel()
        if len(self.s.shape)==1 or len(self.s.ravel())==len(self.X):
            self.s = self.s.reshape(-1,1)
        
        np.random.seed(self.random_state)
        if (self.sampling_proportion!=1.0) or (self.bootstrap):
            indexs_to_keep = []
            # ensuring sampling is stratified
            split_groups = (
                pd.DataFrame(self.s).apply(lambda x: "_".join(x), axis=1).astype(str) + "_" + pd.Series(self.y).astype(str) if (
                    self.s.shape[1] > 1
                ) else (
                    pd.Series(self.s.ravel()).astype(str) + "_" + pd.Series(self.y).astype(str) 
                )
            )
            all_indexs = np.array(range(len(self.X)))
            for split_group in np.unique(split_groups):
                indexs = all_indexs[(split_groups==split_group).values].copy()
                sampling_n = max(1, int(round(len(indexs) * self.sampling_proportion)))
                indexs_to_keep += np.random.choice(
                    indexs,
                    size=sampling_n,
                    replace=self.bootstrap
                ).tolist()
            indexs_to_keep = np.array(indexs_to_keep)
            validation_idx = np.array(list(set(all_indexs).difference(set(indexs_to_keep))))
            self.X_validation = self.X.iloc[validation_idx].copy()
            self.y_validation = self.y[validation_idx].copy()
            self.s_validation = self.s[validation_idx].copy()
            self.X = self.X.iloc[indexs_to_keep]
            self.y = self.y[indexs_to_keep]
            self.s = self.s[indexs_to_keep]
        
        # computing once
        self.y_pos_bool = self.y==1
        self.y_neg_bool = ~self.y_pos_bool      
        self.s_bool_dict = {}
        for s_column in range(self.s.shape[1]):
            self.s_bool_dict[s_column] = {}
            unique_s = np.unique(self.s[:, s_column])
            for s in unique_s:
                self.s_bool_dict[s_column][s] = (self.s[:, s_column])==s
        # for binary-unicategorical methods 
        self.s_pos_bool = self.s_bool_dict[0][self.s[0,0]]
        self.s_neg_bool = ~self.s_pos_bool
        self.s_pos_bool
        self.s_neg_bool
        # for compatibility with scikit-learn
        self.classes_ = np.unique(y)
            
        # feature sampling 
        if "int" in str(type(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, self.max_features),
                replace=False
            ))
        elif "float" in str(type(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(self.X.shape[1] * self.max_features))),
                replace=False
            ))
        elif ("auto" in str(self.max_features)) or ("sqrt" in str(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(np.sqrt(self.X.shape[1])))),
                replace=False
            ))
        elif "log" in str(self.max_features):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(np.log2(self.X.shape[1])))),
                replace=False
            ))
        else:
            self.features = self.X.columns
        
        self.X = self.X[self.features]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_part = self.X.select_dtypes(exclude=numerics)
        numeric_part = self.X.select_dtypes(include=numerics)
        # OHE to be applied to X from predict method
        self.ohe = OneHotEncoder(handle_unknown="ignore").fit(categorical_part)
        # X is now an array
        self.X = np.concatenate(
            (
                numeric_part.values.astype(float),
                self.ohe.transform(categorical_part).toarray()
            ), axis= 1
        )
        
        if self.oob_pruning and self.criterion=="scaff":
            self.X_validation = self.X_validation[self.features]
            categorical_part_val = self.X_validation.select_dtypes(exclude=numerics)
            numeric_part_val = self.X_validation.select_dtypes(include=numerics)
            self.X_validation = np.concatenate(
                (
                    numeric_part_val.values.astype(float),
                    self.ohe.transform(categorical_part_val).toarray()
                ), axis= 1
            )
            
        # feature_value_idx_bool[feature][value] = idx_bool
        # represents the indexs for which feature < value
        # ~feature_value_idx_bool[feature][value]: feature >= value
        self.feature_value_idx_bool = {}
        for feature in range(self.X.shape[1]):
            self.feature_value_idx_bool[feature] = {}
            # discarding the first value since there is no "less than lowest"
            unique_values = np.unique(self.X[:,feature])
            if len(unique_values) >= 2:
                # if the number of split values is "too large"
                if len(unique_values) > (self.n_bins):
                    unique_values = np.unique(np.quantile(
                        a=unique_values,
                        q=np.linspace(0,1,self.n_bins),
                        method="nearest"
                    ))
                    
                for value in unique_values[1:]:
                    idx_bool = (self.X[:,feature] < value)
                    self.feature_value_idx_bool[feature][value] = idx_bool
       
        # prediction threshold
        self.pred_th = self.y.sum() / len(self.y)
        self.indexs = np.repeat(True, len(self.X))
        
        if self.criterion=="scaff":
            scaff_parent = (1-self.orthogonality)*0.5 - self.orthogonality*0.5
            def evaluate_split(feature, value, indexs):
                left_bool = self.feature_value_idx_bool[feature][value] & indexs
                right_bool = (~self.feature_value_idx_bool[feature][value]) & indexs
                # if split results in 2 non-empty partitions with min samples leaf size
                if (left_bool.sum() >= self.min_samples_leaf) and (right_bool.sum() >= self.min_samples_leaf):
                    # focusing on either left or right bool is fine as long as we take the max auc

                    # auc_y
                    tpr_y = ((self.y_pos_bool) & left_bool).sum() / ((self.y_pos_bool) & indexs).sum()
                    fpr_y = ((self.y_neg_bool) & left_bool).sum() / ((self.y_neg_bool) & indexs).sum()
                    auc_y = (1 + tpr_y - fpr_y) / 2
                    auc_y = max(auc_y, 1 - auc_y)

                    # auc_s
                    auc_s_list = []
                    for s_column in range(self.s.shape[1]):
                        unique_s = np.unique(self.s[indexs, s_column])
                        # if more than 1 sensitive attribute is present
                        if len(unique_s) >= 2:
                            for s in unique_s:
                                s_pos = self.s_bool_dict[s_column][s]
                                s_neg = ~s_pos
                                tpr_s = (s_pos & left_bool).sum() / (s_pos & indexs).sum()
                                fpr_s = (s_neg & left_bool).sum() / (s_neg & indexs).sum()
                                auc_s = (1 + tpr_s - fpr_s) / 2
                                auc_s = max(auc_s, 1 - auc_s)
                                auc_s_list.append(auc_s)
                                if len(unique_s)==2:
                                    break
                        else:
                            auc_s = 1
                            auc_s_list.append(auc_s)
                            break                         
                    auc_s = max(auc_s_list)
    
                    scaff_child = (1-self.orthogonality)*auc_y - self.orthogonality*auc_s
                    scaff_gain = scaff_child - scaff_parent
                    if self.split_info_norm=="entropy":
                        split_info = st.entropy([left_bool.sum(), right_bool.sum()], base=2)
                    elif self.split_info_norm=="entropy_inv":
                        split_info = 1 / st.entropy([left_bool.sum(), right_bool.sum()], base=2)
                    else:
                        split_info=1

                    score = scaff_gain / split_info

                else:
                    score = -np.inf

                return score
        
        elif self.criterion=="kamiran":
        # return score of split (dependant on criterion)
            def evaluate_split(feature, value, indexs):
                left_bool = self.feature_value_idx_bool[feature][value] & indexs
                right_bool = (~self.feature_value_idx_bool[feature][value]) & indexs
                # if split results in 2 non-empty partitions
                if (left_bool.sum() >= self.min_samples_leaf) and (right_bool.sum() >= self.min_samples_leaf):
                    # information gain class
                    entropy_parent = st.entropy([
                        (self.y_pos_bool & indexs).sum(), 
                        (self.y_neg_bool & indexs).sum()
                    ])
                    entropy_left = st.entropy([
                        (self.y_pos_bool & left_bool).sum(), 
                        (self.y_neg_bool & left_bool).sum()
                    ])
                    entropy_right = st.entropy([
                        (self.y_pos_bool & right_bool).sum(), 
                        (self.y_neg_bool & right_bool).sum()
                    ])
                    information_gain_class = entropy_parent - sum([
                        entropy_left * (left_bool.sum() / indexs.sum()),
                        entropy_right * (right_bool.sum() / indexs.sum())                       
                    ])
                    
                    # information gain sensitive
                    entropy_parent = st.entropy([
                        (self.s_pos_bool & indexs).sum(), 
                        (self.s_neg_bool & indexs).sum()
                    ])
                    entropy_left = st.entropy([
                        (self.s_pos_bool & left_bool).sum(), 
                        (self.s_neg_bool & left_bool).sum()
                    ])
                    entropy_right = st.entropy([
                        (self.s_pos_bool & right_bool).sum(), 
                        (self.s_neg_bool & right_bool).sum()
                    ])
                    information_gain_sensitive = entropy_parent - sum([
                        entropy_left * (left_bool.sum() / indexs.sum()),
                        entropy_right * (right_bool.sum() / indexs.sum())
                    ])
                    
                    if self.kamiran_method=="sum":
                        score = information_gain_class + information_gain_sensitive
                        
                    elif self.kamiran_method=="sub":
                        score = information_gain_class - information_gain_sensitive
                        
                    elif self.kamiran_method=="div":
                        if information_gain_sensitive==0:
                            # division by zero
                            score = np.inf
                            
                        else:
                            score = information_gain_class / information_gain_sensitive

                else:
                    score = -np.inf

                return score
            
        # return best (score, feature, split_value) dependant on criterion and indexs
        def get_best_split(indexs):
            best_score = 0
            best_value = np.nan
            best_feature  = np.nan
            for feature in range(self.X.shape[1]):
                unique_values = np.unique(self.X[indexs,feature]) 
                if len(unique_values) >= 2:
                    unique_intersect = np.intersect1d(
                        unique_values[1:],
                        np.array(list(self.feature_value_idx_bool[feature].keys()))
                    )
                    # we know that the unique_values[0] is no-good as splitter 
                    # it would generate a left empty node, and a right full node
                    for value in unique_intersect:
                        split_score = evaluate_split(feature, value, indexs)
                        if split_score >= best_score:
                            best_score = split_score
                            best_feature = feature
                            best_value = value
        
            return best_score, best_feature, best_value
        
        # recursively grow the actual tree
        def build_tree(indexs, depth=0):
            tree={}
            if (                
                len(np.unique(self.y[indexs]))==1 or ( # no need to split if there is already only 1 y class
                indexs.sum()<self.min_samples_split) or ( # minimum number of samples to consider a split
                depth==self.max_depth) # if we've reached the max depth in the tree
            ):
                class_prob = (self.y_pos_bool & indexs).sum() / indexs.sum()
                if self.criterion=="scaff":
                    return class_prob
                
                elif self.criterion=="kamiran":
                    # https://www.win.tue.nl/~mpechen/publications/pubs/KamiranICDM2010.pdf
                    leaf_y_neg_n = (indexs & self.y_neg_bool).sum()
                    leaf_y_pos_n = (indexs & self.y_pos_bool).sum()
                    leaf_s_neg_n = (indexs & self.s_neg_bool).sum()
                    leaf_s_pos_n = (indexs & self.s_pos_bool).sum()
                    
                    if leaf_y_neg_n > leaf_y_pos_n:
                        delta_acc = (leaf_y_pos_n - leaf_y_neg_n) / len(y)
                        delta_disc = leaf_s_neg_n/self.s_neg_bool.sum() - leaf_s_pos_n/self.s_pos_bool.sum()
                        
                    else:
                        delta_acc = (leaf_y_neg_n - leaf_y_pos_n) / len(y)
                        delta_disc = leaf_s_pos_n/self.s_pos_bool.sum() - leaf_s_neg_n/self.s_neg_bool.sum()
                    
                    if delta_acc==0:
                        return class_prob, delta_acc, delta_disc, np.inf
                    else:
                        return class_prob, delta_acc, delta_disc, abs(delta_disc/delta_acc)
                
            else:
                score, feature, value = get_best_split(indexs)
                if np.isnan(feature): ## in case no more feature values exist for splitting
                    class_prob = (self.y_pos_bool & indexs).sum() / indexs.sum()
                    if self.criterion=="scaff":
                        return class_prob
                    
                    elif self.criterion=="kamiran":
                        # https://www.win.tue.nl/~mpechen/publications/pubs/KamiranICDM2010.pdf
                        leaf_y_neg_n = (indexs & self.y_neg_bool).sum()
                        leaf_y_pos_n = (indexs & self.y_pos_bool).sum()

                        leaf_s_neg_n = (indexs & self.s_neg_bool).sum()
                        leaf_s_pos_n = (indexs & self.s_pos_bool).sum()

                        if leaf_y_neg_n > leaf_y_pos_n:
                            delta_acc = (leaf_y_pos_n - leaf_y_neg_n) / len(y)
                            delta_disc = leaf_s_neg_n/self.s_neg_bool.sum() - leaf_s_pos_n/self.s_pos_bool.sum()

                        else:
                            delta_acc = (leaf_y_neg_n - leaf_y_pos_n) / len(y)
                            delta_disc = leaf_s_pos_n/self.s_pos_bool.sum() - leaf_s_neg_n/self.s_neg_bool.sum()

                        if delta_acc==0:
                            return class_prob, delta_acc, delta_disc, np.inf
                        else:
                            return class_prob, delta_acc, delta_disc, abs(delta_disc/delta_acc)
                    
                else:
                    left_indexs = self.feature_value_idx_bool[feature][value] & indexs
                    right_indexs = (~self.feature_value_idx_bool[feature][value]) & indexs
                    tree[(feature, value)] = {
                        "prob": (self.y[indexs]).sum() / indexs.sum(),
                        "<": build_tree(left_indexs, depth=depth+1),
                        ">=":  build_tree(right_indexs, depth=depth+1),                        
                    }
                    return tree
                
        self.tree = build_tree(self.indexs)
        
        def get_paths_leaves(tree, path=[]):
            output=[]
            if type(tree)==type({}):
                feature, value  = list(tree.keys())[0]
                sub_tree_left = tree[(feature, value)]["<"]
                sub_tree_right = tree[(feature, value)][">="]
                output += get_paths_leaves(sub_tree_left, path+[(feature, value, "<")])
                output += get_paths_leaves(sub_tree_right, path+[(feature, value, ">=")])
                return output

            else:
                leaf = tree
                return [[path, leaf]]
        
        #  prune tree with X_validation
        if self.oob_pruning and self.criterion=="scaff":
            def get_prob(X, tree, y_prob=None, idx_bool=None):
                y_prob = np.repeat(np.nan, len(X)) if (y_prob is None) else (y_prob)
                idx_bool = np.repeat(True, len(X)) if (idx_bool is None) else (idx_bool)
                if type(tree)==type({}):
                    feature, value  = list(tree.keys())[0]
                    left_bool =  (X[:,feature] < value) & idx_bool
                    right_bool = (X[:,feature] >= value) & idx_bool
                    sub_tree_left = tree[(feature, value)]["<"]
                    sub_tree_right = tree[(feature, value)][">="]
                    y_prob = get_prob(X, sub_tree_left, y_prob, left_bool)
                    y_prob = get_prob(X, sub_tree_right, y_prob, right_bool)
                    return y_prob

                else:
                    y_prob[idx_bool] = tree
                    return y_prob

            y_prob = get_prob(self.X_validation, self.tree)
            auc_y = roc_auc_score(self.y_validation, y_prob) 
            auc_s = sensitive_auc(self.s_validation, y_prob) 
            best_score = (1-self.orthogonality)*auc_y - self.orthogonality*auc_s
            
            stop_flag = 0     
            while not stop_flag:
                best_i = None
                paths_leaves = get_paths_leaves(self.tree)
                # len(paths_leaves)==1 means root node
                if len(paths_leaves)>1:
                    for i in range(len(paths_leaves)):
                        path, _ = paths_leaves[i]
                        sub_tree = copy(self.tree)
                        str_path = "sub_tree"
                        for sub_path in path[:-1]:
                            feature, value, sign = sub_path
                            str_loc = "[(" + str(feature) + "," + str(value) + ")]['" + sign+"']"
                            str_path += str_loc
                        feature, value, _ = path[-1]
                        last_str_loc = "[(" + str(feature) + "," + str(value) + ")]['prob']"
                        last_str_path = str_path + last_str_loc
                        new_prob = eval(last_str_path)
                        exec(str_path + " = " + str(new_prob))
                        y_prob = get_prob(self.X_validation, sub_tree)
                        auc_y = roc_auc_score(self.y_validation, y_prob) 
                        auc_s = sensitive_auc(self.s_validation, y_prob) 
                        new_score = (1-self.orthogonality)*auc_y - self.orthogonality*auc_s
                        if new_score > best_score:
                            best_score = new_score
                            best_i = i

                    if best_i is not None:
                        path, _ = paths_leaves[best_i]
                        str_path = "self.tree"
                        for sub_path in path[:-1]:
                            feature, value, sign = sub_path
                            str_loc = "[(" + str(feature) + "," + str(value) + ")]['" + sign+"']"
                            str_path += str_loc
                        feature, value, _ = path[-1]
                        last_str_loc = "[(" + str(feature) + "," + str(value) + ")]['prob']"
                        last_str_path = str_path + last_str_loc
                        new_prob = eval(last_str_path)
                        exec(str_path + " = " + str(new_prob))

                    else:
                        stop_flag = 1
                        
                else:
                    stop_flag = 1
                    
        self.paths_leaves = get_paths_leaves(self.tree)
        self.is_fit = True
        
    def predict_proba(self, X, theta=None):
        if "pandas" not in str(type(X)):
            X = pd.DataFrame(X)

        X = X[self.features]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_part = X.select_dtypes(exclude=numerics)
        numeric_part = X.select_dtypes(include=numerics)
        # X is now an array
        X = np.concatenate(
            (
                numeric_part.values.astype(float),
                self.ohe.transform(categorical_part).toarray()
            ), axis= 1
        )

        if self.criterion=="scaff":
            def get_prob(X, tree, y_prob=None, idx_bool=None):
                y_prob = np.repeat(np.nan, len(X)) if (y_prob is None) else (y_prob)
                idx_bool = np.repeat(True, len(X)) if (idx_bool is None) else (idx_bool)
                if type(tree)==type({}):
                    feature, value  = list(tree.keys())[0]
                    left_bool =  (X[:,feature] < value) & idx_bool
                    right_bool = (X[:,feature] >= value) & idx_bool
                    sub_tree_left = tree[(feature, value)]["<"]
                    sub_tree_right = tree[(feature, value)][">="]
                    y_prob = get_prob(X, sub_tree_left, y_prob, left_bool)
                    y_prob = get_prob(X, sub_tree_right, y_prob, right_bool)
                    return y_prob

                else:
                    y_prob[idx_bool] = tree
                    return y_prob

            y_prob = get_prob(X, self.tree).reshape(-1,1)
        
        elif self.criterion=="kamiran":
            def get_prob(X, tree, y_prob=None, idx_bool=None):
                y_prob = np.repeat(np.nan, len(X)) if (y_prob is None) else (y_prob)
                idx_bool = np.repeat(True, len(X)) if (idx_bool is None) else (idx_bool)
                if type(tree)==type({}):
                    feature, value  = list(tree.keys())[0]
                    left_bool = (X[:,feature] < value) & idx_bool
                    right_bool = (X[:,feature] >= value) & idx_bool
                    sub_tree_left = tree[(feature, value)]["<"]
                    sub_tree_right = tree[(feature, value)][">="]
                    y_prob = get_prob(X, sub_tree_left, y_prob, left_bool)
                    y_prob = get_prob(X, sub_tree_right, y_prob, right_bool)
                    return y_prob

                else:
                    y_prob[idx_bool] = tree[0]
                    return y_prob
            
            def kamiran_discrimination(X="X", s="s"):
                y_pred = (get_prob(X, self.tree) >= 0.5).astype(int)
                y_pred_pos = y_pred==1
                kamiran_discrimination = \
                    (self.s_neg_bool & y_pred_pos).sum() / self.s_neg_bool.sum() - \
                    (self.s_pos_bool & y_pred_pos).sum() / self.s_pos_bool.sum()

                return kamiran_discrimination
        
            kamiran_discrimination = kamiran_discrimination(self.X, self.s)
            if kamiran_discrimination != 0:
                # grabbing leaves and paths to check which ones to swap
                paths = []
                leaves = []
                for path, leaf in self.paths_leaves:
                    class_label, delta_acc, delta_disc, score = leaf
                    # discriminations must have different signs
                    # since we want to reduce the abs(bias)
                    if (delta_disc * kamiran_discrimination) < 0:
                        paths.append(path)
                        leaves.append(leaf)
                        
                # checking which leaves are best to swap and saving their paths
                paths_to_swap = []
                leaves = np.array(leaves)
                ranked_leaf_indxs = np.argsort(leaves[:,-1])[::-1]
                rem_disc = kamiran_discrimination
                counter = 0
                if theta is None:
                    e_bound = (1-self.orthogonality)*kamiran_discrimination
                else:
                    e_bound = (1-theta)*kamiran_discrimination
                for i in range(len(ranked_leaf_indxs)):
                    ranked_idx = ranked_leaf_indxs[i]
                    delta_disc = leaves[ranked_idx, 2]
                    if abs(rem_disc) > abs(e_bound):
                        if abs(rem_disc + delta_disc) > abs(rem_disc):
                            pass
                        else:
                            counter += 1
                            rem_disc += delta_disc
                            path_to_swap = paths[ranked_idx]
                            paths_to_swap.append(path_to_swap)
                            
                swap_tree = copy(self.tree)
                for path in paths_to_swap:
                    path_str = ""
                    for sub_path in path:
                        feature, value, sign = sub_path
                        path_str += "[("+str(feature)+", "+str(value)+")]['"+sign+"']"

                    class_label, delta_acc, delta_disc, score = eval("swap_tree" + path_str)
                    swap_leaf = (1 - class_label, None, None, None) # no need for the scores anymore
                    command = "swap_tree" + path_str + " = "+ str(swap_leaf)
                    exec(command)
                
                y_prob = get_prob(X, swap_tree).reshape(-1,1)
                
            else:
                y_prob = get_prob(X, self.tree).reshape(-1,1)
                
        
        return np.concatenate(
            ((1 - y_prob), y_prob),
            axis=1
        )
       
    def predict(self, X, theta=None):
        y_prob = self.predict_proba(X, theta)[:,1]            
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred
            
    # for compatibility with scikit-learn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # for compatibility with scikit-learn
    def get_params(self, deep=False):
        if deep:
            return copy({
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "orthogonality": self.orthogonality,
                "oob_pruning": self.oob_pruning,
                "kamiran_method": self.kamiran_method,
                "split_info_norm": self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "sampling_proportion": self.sampling_proportion,
            })
        
        else:
            return {
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "orthogonality": self.orthogonality,
                "oob_pruning": self.oob_pruning,
                "kamiran_method": self.kamiran_method,
                "split_info_norm": self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "sampling_proportion": self.sampling_proportion,
            }
        
    def __str__(self):
        string = "FairDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  bootstrap=" + str(self.bootstrap) + "\n" + \
                "  max_features=" + str(self.max_features) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  orthogonality=" + str(self.orthogonality) + "\n" + \
                "  oob_pruning=" + str(self.oob_pruning) + "\n" + \
                "  kamiran_method=" + str(self.kamiran_method) + "\n" + \
                "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                "  min_samples_leaf=" +str(self.min_samples_leaf) + "\n" + \
                "  min_samples_split=" +str(self.min_samples_split) + "\n" + \
                "  sampling_proportion=" +str(self.sampling_proportion)
    
        return string

    def __repr__(self):
        return self.__str__()
    
class FairRandomForestClassifier():
    def __init__(
        self, 
        n_jobs=-1,
        n_bins=256,
        max_depth=7,
        bootstrap=True, 
        random_state=42,
        n_estimators=500,
        orthogonality=.5,
        oob_pruning=True,
        criterion="scaff",
        min_samples_leaf=3,
        min_samples_split=7,
        max_features="auto",
        kamiran_method=None,
        split_info_norm=None,
        sampling_proportion=1.0, 
        # the estimate proportion of unique samples
        # equivalent to sampling_proportion=1, bootstrap=True
        # https://stats.stackexchange.com/questions/126107/
    ):
        """
        Fair Random Forest Classifier
        n_estimators -> int: 
            number of FairDecisionTreeClassifier objects
        n_bins -> int: 
            feature quantiles from which candidate splits are generated
        min_samples_split -> int: 
            smallest number of samples in a node for a split to be considered
        min_samples_leaf -> int: 
            smallest number of samples in each leaf after a split for that split to be considered
        max_depth -> int: 
            max number of allowed splits per tree
        sampling_proportion -> float: 
            proportion of samples to resample in each tree
        max_features -> int: 
            number of samples to bootstrap
                     -> float: 
            proportion of samples to bootstrap
                     -> str:
            "auto"/"sqrt": sqrt of the number of features is used
            "log"/"log2": log2 of the number of features is used
        bootstrap -> bool: 
            bootstrap strategy with (True) or without (False) replacement
        random_state -> int: 
            seed for all random processes
        criterion -> str: 
            score criterion for splitting
            {"scaff", "kamiran"}
        kamiran_method -> str
            operation to combine IGC and IGS when criterion=='kamiran'
            {"sum", "sub", "div"}
        split_info_norm -> str
            denominator in gain normalisation:
            {"entropy", "entropy_inv", None}
        oob_pruning -> bool
            if out of bag samples (when sample_proportion!=1.0 or bootstrap==True) should be used to prune after fitting
        orthogonality -> int/float: 
            strength of fairness constraint in which:
            0 is no fairness constraint (i.e., 'traditional' classifier)
            [0,1] 
        n_jobs -> int: 
            CPU usage; -1 for all threads
        """
        
        self.is_fit = False
        self.n_jobs = n_jobs
        self.n_bins =  n_bins
        self.max_depth = max_depth
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.oob_pruning = oob_pruning
        self.kamiran_method = kamiran_method
        self.split_info_norm = split_info_norm
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sampling_proportion = sampling_proportion
        
    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> any_dim pandas.df or np.array: numerical/categorical
        y -> one_dim pandas.df or np.array: only binary
        s -> any_dim pandas.df or np.array: columns must be binary
        """
        
        def make_batches(iterable, n_jobs=-1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count()
            len_iterable = len(iterable)
            if len_iterable < n_jobs:
                n_jobs = len_iterable
            batches = [[] for i in range(n_jobs)]
            for i in range(len_iterable):
                item = iterable[i]
                batches[i%n_jobs].append(item)
            return batches

        def fit_batch(batch_trees, X, y, s):
            for tree in batch_trees:
                tree.fit(X, y, s)
            return batch_trees
        
        # Generating FairRandomForest
        
        np.random.seed(self.random_state)
        # this is the range of all possible seed values in numpy
        random_states = np.random.randint(0, 2**31, self.n_estimators) 
        while len(np.unique(random_states)) != len(random_states):
            random_states = np.random.randint(0, 2**31, self.n_estimators)
        
        trees = [
            FairDecisionTreeClassifier(
                n_bins=self.n_bins,
                max_depth=self.max_depth,
                bootstrap=self.bootstrap, 
                criterion=self.criterion, 
                random_state=random_state,
                max_features=self.max_features, 
                orthogonality=self.orthogonality,
                kamiran_method=self.kamiran_method,
                oob_pruning=self.oob_pruning,
                split_info_norm=self.split_info_norm, 
                min_samples_leaf=self.min_samples_leaf, 
                min_samples_split=self.min_samples_split, 
                sampling_proportion=self.sampling_proportion, 
            )
            for random_state in random_states
        ]
    
        self.trees = trees
        self.classes_ = np.unique(y)
        self.pred_th = (y==1).sum() / len(y)
        self.s = np.array(s).astype(object) if (
            "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(object)
        )
        
        if self.n_estimators==1:
            self.n_jobs=1
        if self.n_jobs==1:
            for tree in self.trees:
                tree.fit(X, y, s)
        else:
            batches_trees = make_batches(self.trees, n_jobs=self.n_jobs)
            fit_batches_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_batch)(
                    batch_trees, 
                    X,
                    y, 
                    s, 
                ) for batch_trees in batches_trees
            )
            self.trees = [tree for fit_batch_trees in fit_batches_trees for tree in fit_batch_trees]
                         
        self.fit = True
           
    def predict_proba(self, X, theta=None, mean_type="prob"):
        """
        Retuns the predicted probabilties of input X
        theta -> float
            orthogonality parameter for kamiran
            if not specified, the orthogonality parameter given in init is used instead
        mean_type -> str
            Method to compute the probailities across all trees
            {"prob", "pred"}
            "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
            "pred" computes the mean of all tree predicitons {0, 1}
        """
        
        def make_batches(iterable, n_jobs=-1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count()
            len_iterable = len(iterable)
            if len_iterable < n_jobs:
                n_jobs = len_iterable
            batches = [[] for i in range(n_jobs)]
            for i in range(len_iterable):
                item = iterable[i]
                batches[i%n_jobs].append(item)
            return batches

        def predict_proba_batch(batch_trees, X, theta, mean_type):
            batch_prob = []
            if mean_type=="prob":
                for tree in batch_trees:
                    batch_prob.append(tree.predict_proba(X, theta=theta)[:,1])
            elif mean_type=="pred":
                for tree in batch_trees:
                    batch_prob.append(tree.predict(X, theta=theta))
            return batch_prob
        
        if self.n_jobs==1:
            if mean_type=="prob":
                y_prob = np.mean(
                    [tree.predict_proba(X, theta=theta)[:,1] for tree in self.trees], 
                    axis=0
                ).reshape(-1,1)

            elif mean_type=="pred":
                y_prob = np.mean(
                    [tree.predict(X, theta=theta) for tree in self.trees], 
                    axis=0
                ).reshape(-1,1)
                
        else:
            batches_trees = make_batches(self.trees, n_jobs=self.n_jobs)
            proba_batches = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_batch)(
                    batch_trees, 
                    X,
                    theta,
                    mean_type
                ) for batch_trees in batches_trees
            )
            y_prob = np.mean(
                [prob for proba_batch in proba_batches for prob in proba_batch],
                axis=0
            ).reshape(-1,1)
         
        return np.concatenate(
            (1- y_prob, y_prob), 
            axis=1
        )
    
    def predict(self, X, theta=None,  mean_type="prob"):
        """
        Retuns the predicted class label of input X
        theta -> float
            orthogonality parameter for the kamiran method when criterion=="kamiran"
            if not specified, the orthogonality parameter given in init is used instead
        mean_type -> str
            {"prob", "pred"}
            Method to compute the probailities across all trees, with which the np.mean([0.5, self.pred_th]) is the threshold
            Note: self.pred_th is given as the proportion of positive class instances [P(Y=1)]
            "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
            "pred" computes the mean of all tree predicitons {0, 1}
        """
        return (self.predict_proba(X, theta, mean_type)[:,1] >= np.mean([0.5, self.pred_th])).astype(int)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_params(self, deep=False):
        if deep:
            return copy({
                "n_jobs": self.n_jobs,
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_estimators": self.n_estimators,
                "orthogonality": self.orthogonality,
                "criterion": self.criterion,
                "oob_pruning": self.oob_pruning,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features,
                "split_info_norm": self.split_info_norm,
                "sampling_proportion": self.sampling_proportion,
            })
        else:
            return {
                "n_jobs": self.n_jobs,
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_estimators": self.n_estimators,
                "orthogonality": self.orthogonality,
                "criterion": self.criterion,
                "oob_pruning": self.oob_pruning,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features,
                "split_info_norm": self.split_info_norm,
                "sampling_proportion": self.sampling_proportion,
            }
        
    def __str__(self):
        string = "FairDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  bootstrap=" + str(self.bootstrap) + "\n" + \
                "  max_features=" + str(self.max_features) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  n_estimators=" + str(self.n_estimators) + "\n" + \
                "  orthogonality=" + str(self.orthogonality) + "\n" + \
                "  oob_pruning=" + str(self.oob_pruning) + "\n" + \
                "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                "  min_samples_leaf=" +str(self.min_samples_leaf) + "\n" + \
                "  min_samples_split=" +str(self.min_samples_split) + "\n" + \
                "  sampling_proportion=" +str(self.sampling_proportion)
    
        return string

    def __repr__(self):
        return self.__str__()
