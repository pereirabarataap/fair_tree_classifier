import warnings
import numpy as np
import pandas as pd
from math import ceil
import multiprocessing
from tqdm.auto import tqdm
from copy import deepcopy as copy
from joblib import delayed, Parallel
from scipy.stats import mode, entropy
from sklearn.metrics import roc_auc_score

def sns_auc_score(s_true, y_score):
    s_auc = roc_auc_score(s_true, y_score)
    s_auc = max(1-s_auc, s_auc)
    return s_auc

class FairDecisionTreeClassifier():
    
    def __init__(
        self,
        n_bins=100,
        max_depth=None,
        bootstrap=False,
        random_state=42,
        orthogonality=.5, 
        criterion="scaff",         
        min_samples_leaf=5, 
        max_features="auto", 
        split_info_norm=None,
        sampling_proportion=1.0, 
    ):
        self.is_fit = False
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.split_info_norm = split_info_norm
        self.min_samples_leaf = min_samples_leaf
        self.sampling_proportion = sampling_proportion
        self.n_bins = np.inf if (n_bins is None) else n_bins
        self.max_depth = np.inf if (max_depth is None) else (max_depth)
                                
    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> pandas.df: may contain int float str
        y -> one_dim pandas.df or np.array: only binary int {0, 1}
        s -> any_dim pandas.df or np.array: only str
        kwargs -> for compatibility with scikit-learn: fit_params in cross_validate()
        """
        
        self.X = X
        self.y = np.array(y).astype(int)
        # for compatibility with scikit-learn since sklearn fit() methods only take X, y
        self.s = np.array(s).astype(object) if (
            "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(object)
        )
        
        np.random.seed(self.random_state)
       
        if (len(self.X)!=len(self.y)) or (len(self.X)!=len(self.s)) or (len(self.y)!=len(self.s)):
            raise Exception("X, y, and s lenghts do not match")    
        if len(self.y.shape)==1 or len(self.y.ravel())==len(self.X):
            self.y = self.y.ravel()
        if len(self.s.shape)==1 or len(self.s.ravel())==len(self.X):
            self.s = self.s.reshape(-1,1)
        
        if (self.sampling_proportion!=1.0) or (self.bootstrap):
            idx = np.random.choice(
                range(len(self.X)),
                size=int(round(len(self.X) * self.sampling_proportion)),
                replace=self.bootstrap
            )
            self.X = self.X.iloc[idx]
            self.y = self.y[idx]
            self.s = self.s[idx]
        
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
        
        if self.criterion =="scaff":
        # return score of split (dependant on criterion)
            def evaluate_split(feature, value, indexs):
                left_bool = self.feature_value_idx_bool[feature][value] & indexs
                right_bool = (~self.feature_value_idx_bool[feature][value]) & indexs
                # if split results in 2 non-empty partitions
                if (left_bool.sum() >= 1) and (right_bool.sum() >= 1):
                    # focusing on either left or right bool is fine as long as we take the max auc

                    # auc_y
                    tpr_y = (self.y==1)[left_bool].sum() / (self.y==1)[indexs].sum()
                    fpr_y = (self.y==0)[left_bool].sum() / (self.y==0)[indexs].sum()
                    auc_y = (1 + tpr_y - fpr_y) / 2
                    auc_y = max(auc_y, 1 - auc_y)

                    # auc_s
                    auc_s_list = []
                    for s_column in range(self.s.shape[1]):
                        unique_s = np.unique(self.s[indexs, s_column])
                        # if more than 1 sensitive attribute is present
                        if len(unique_s) >= 2:
                            for s in unique_s:
                                tpr_s = (self.s==s)[left_bool].sum() / (self.s==s)[indexs].sum()
                                fpr_s = (self.s!=s)[left_bool].sum() / (self.s!=s)[indexs].sum()
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

                    scaff_parent = (1-self.orthogonality)*0.5 - self.orthogonality*0.5
                    scaff_children = (1-self.orthogonality)*auc_y - self.orthogonality*auc_s
                    scaff_gain = scaff_children - scaff_parent

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
                
        # return best (sscore, feature, split_value) dependant on criterion and indexs
        def get_best_split(indexs):
            best_score = 0
            best_value = np.nan
            best_feature  = np.nan
            for feature in range(self.X.shape[1]):
                unique_values = np.unique(self.X[indexs,feature])
                unique_values = np.intersect1d(
                    unique_values,
                    np.array(list(self.feature_value_idx_bool[feature].keys()))
                )
                if len(unique_values) >= 2:
                    for value in unique_values[1:]:
                        split_score = evaluate_split(feature, value, indexs)
                        if split_score >= best_score:
                            best_score = split_score
                            best_feature = feature
                            best_value = value
            
            return best_score, best_feature, best_value
        
        # recursively grow the actual tree ---> {split1: {...}}
        def build_tree(indexs, depth=0):
            tree={}
            depth = copy(depth)
            indexs = copy(indexs)
            if (                
                len(np.unique(self.y[indexs]))==1 or ( # no need to split if there is already only 1 y class
                indexs.sum()<=self.min_samples_leaf) or ( # minimum number to consider a node as a leaf
                depth==self.max_depth) # if we've reached the max depth in the tree
            ):
                return (self.y[indexs]).sum() / indexs.sum()

            else:
                score, feature, value = get_best_split(indexs)
                
                if np.isnan(feature): ## in case no more feature values exist for splitting
                    return (self.y[indexs]).sum() / indexs.sum()
                
                else:
                    left_indexs = self.feature_value_idx_bool[feature][value] & indexs
                    right_indexs = (~self.feature_value_idx_bool[feature][value]) & indexs

                    tree[(feature, value)] = {
                        "<": build_tree(left_indexs, depth=copy(depth+1)),
                        ">=":  build_tree(right_indexs, depth=copy(depth+1))
                    }

                    return tree
        
        self.tree = build_tree(self.indexs)
        self.is_fit = True
        del self.X
        del self.y
        del self.s
           
    def predict_proba(self, X):
        
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

        def get_proba(X, tree, y_prob=np.repeat(np.nan, len(X)), idx_bool=np.repeat(True, len(X))):
            if type(tree)==type({}):
                feature, value  = list(tree.keys())[0]
                left_bool =  (X[:,feature] < value) & idx_bool
                right_bool = (X[:,feature] >= value) & idx_bool
                sub_tree_left = tree[(feature, value)]["<"]
                sub_tree_right = tree[(feature, value)][">="]
                y_prob = get_proba(X, sub_tree_left, y_prob, left_bool)
                y_prob = get_proba(X, sub_tree_right, y_prob, right_bool)
                return y_prob

            else:
                y_prob[idx_bool] = tree
                return y_prob
            
        y_prob = get_proba(X, self.tree).reshape(-1,1)
        return np.concatenate(
            ((1 - y_prob), y_prob),
            axis=1
        )
    
    def predict(self, X):
        
        y_prob = self.predict_proba(X)
        return (y_prob[:,1] >= self.pred_th).astype(int)
    
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
                "split_info_norm":  self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
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
                "split_info_norm":  self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
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
                "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                "  min_samples_leaf=" +str(self.min_samples_leaf) + "\n" + \
                "  sampling_proportion=" +str(self.sampling_proportion)
    
        return string

    def __repr__(self):
        return self.__str__()
    
class FairRandomForestClassifier():
    def __init__(
        self, 
        n_jobs=-1,
        n_bins=100,
        max_depth=None,
        bootstrap=True, 
        random_state=42,
        n_estimators=100,
        orthogonality=.5,
        criterion="scaff",
        min_samples_leaf=5,
        max_features="auto",
        split_info_norm=None,
        sampling_proportion=1.0, 
    ):
        """
        Fair Random Forest Classifier
        n_estimators -> int: 
            number of FairDecisionTreeClassifier objects
        n_bins -> int: 
            feature quantiles from which candidate splits are generated
        min_samples_leaf -> int: 
            node is terminal if #samples in node <= min_samples_leaf
        max_depth -> int: 
            max number of allowed splits per tree
        sampling_proportion -> float: 
            proportion of samples to bootstrap in each tree
        max_features -> int: 
            number of samples to bootstrap
                     -> float: 
            proportion of samples to bootstrap
                     -> str:
            "auto"/"sqrt": sqrt of features is used
            "log"/"log2": log2 of features is used
        bootstrap -> bool: 
            bootstrap strategy (with out without replacement)
        random_state -> int: 
            seed for all random processes
        criterion -> str: 
            ["scaff"] score criterion for splitting
        split_info_norm -> str
            denominator in gain normalisation:
            ["entropy", "entropy_inv", None]
        orthogonality -> int/float: 
            strength of fairness constraint
        n_jobs -> int: 
            CPU usage; -1 for all 
        """
        
        self.is_fit = False
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.split_info_norm = split_info_norm
        self.min_samples_leaf = min_samples_leaf
        self.sampling_proportion = sampling_proportion
        self.n_bins = np.inf if (n_bins is None) else n_bins
        self.max_depth = np.inf if (max_depth is None) else (max_depth)    

        # Generating FairRandomForest
        trees = [
            FairDecisionTreeClassifier(
                n_bins=self.n_bins,
                max_depth=self.max_depth,
                bootstrap=self.bootstrap, 
                criterion=self.criterion, 
                max_features=self.max_features, 
                random_state=self.random_state+i,
                orthogonality=self.orthogonality, 
                split_info_norm=self.split_info_norm, 
                min_samples_leaf=self.min_samples_leaf, 
                sampling_proportion=self.sampling_proportion, 
            )
            for i in range(self.n_estimators)
        ]
        self.trees = trees
        
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
        
        self.classes_ = np.unique(y)
        self.s = np.array(s).astype(object) if (
            "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(object)
        )
        
        if self.n_jobs==1:
            for tree in self.trees:
                tree.fit(X, y, s)
        else:
            batches_trees = make_batches(self.trees, n_jobs=self.n_jobs)
            fit_batches_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_batch)(
                    batch_trees, 
                    copy(X), 
                    copy(y), 
                    copy(s), 
                ) for batch_trees in batches_trees
            )
            trees = [tree for fit_batch_trees in fit_batches_trees for tree in fit_batch_trees]
            self.trees = trees
            
        self.fit = True
           
    def predict_proba(self, X):
        y_prob = np.mean(
            [tree.predict_proba(X)[:,1] for tree in self.trees], 
            axis=0
        ).reshape(-1,1)
        return np.concatenate((1- y_prob, y_prob), axis=1)
    
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
                "min_samples_leaf": self.min_samples_leaf,
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
                "min_samples_leaf": self.min_samples_leaf,
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
                "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                "  min_samples_leaf=" +str(self.min_samples_leaf) + "\n" + \
                "  sampling_proportion=" +str(self.sampling_proportion)
    
        return string

    def __repr__(self):
        return self.__str__()
