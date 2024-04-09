import joblib
import warnings
import numpy as np
import pandas as pd
from math import sqrt, log2
from collections import Counter
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder as OHE, KBinsDiscretizer as KBD

pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings("ignore", category=UserWarning)

def sdp_score(z_true, y_prob):
    """
    Strong Demographic Parity Score
    Same API as roc_auc_score from sklean.metric
    Returns a value between 0 and 1 in which:
        0 indicates complete algorithmic bias,
        1 indicates complete algorithmic fairness
    """
    sdp = roc_auc_score(z_true, y_prob)
    sdp_score = 1 - (2*abs(sdp - 0.5))
    return sdp_score

def find_first_occurrence(l, x):
    """
    Binary search (requires "l" to be sorted)
    Returns the index of the first occurrence of x in the sorted list l,
    or -1 if x is not found.
    """
    low, high = 0, len(l) - 1
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if l[mid] == x:
            result = mid
            high = mid - 1  # Look on the left side for the first occurrence
        elif l[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return result

def generate_batches(n, n_batches):
    """Generate batch indices for n items into n_batches, distributing items more evenly."""
    batch_size, remainder = divmod(n, n_batches)
    start = 0
    for i in range(n_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        yield range(start, end)
        start = end
        
class Node:
    def __init__(self, num_samples, num_samples_per_class, predicted_class, probabilities=None):
        self.left = None
        self.right = None
        self.threshold = 0
        self.feature_index = 0
        self.num_samples = num_samples
        self.probabilities = probabilities  
        self.predicted_class = predicted_class
        self.num_samples_per_class = num_samples_per_class

class FairDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_bins=256,
        max_depth=None,
        bootstrap=False,
        random_state=42,
        orthogonality=0, 
        max_features=None, 
        min_samples_leaf=1, 
        min_samples_split=2, 
        requires_data_processing=False,
    ):
        self.n_bins=n_bins
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.requires_data_processing = requires_data_processing
        if isinstance(max_depth, type(None)):
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth
            
        np.random.seed(self.random_state)
        self.parent_scaff = (1-self.orthogonality)*0.5 - self.orthogonality*0.5

    def _get_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(log2(n_features) + 1))
        else:
            raise ValueError("Invalid value for max_features")

    def _best_split(self, X, y, z):
        m, n_features = X.shape
        if m < self.min_samples_split:
            return None, None
        
        counts_y = Counter(y)
        num_parent_y = [counts_y[c] for c in range(self.n_classes_)]
        
        # list of lists, each outer list reppresenting a sensitive attribute
        # inner list has the counts of each sensitive value within the specific attribute
        # e.g. [[34,54], [24, 34, 30]] 
        num_parent_z = []
        for i in range(self.n_sens_groups):
            z_group_counts = Counter(z[:,i])
            num_parent_z.append([z_group_counts[s] for s in range(self.n_senses_[i])])
        
        best_split_score = 0
        best_idx, best_thr = None, None
        
        for idx in self.features_indices:
            sorted_idxs = np.argsort(X[:, idx])
            thresholds, classes, senses = X[sorted_idxs, idx], (y[sorted_idxs]).tolist(), (z[sorted_idxs]).T.tolist()
            
            num_left_y = [0, 0]
            num_right_y = num_parent_y
            
            num_left_z = [[0 for s in range(self.n_senses_[i])] for i in range(self.n_sens_groups)]
            num_right_z = num_parent_z
            
            threshold_index_a = 0
            
            unique_thresholds = np.unique(thresholds)
            for i in range(len(unique_thresholds)-1):
                threshold_b = unique_thresholds[i+1]
                threshold_index_b = find_first_occurrence(thresholds, threshold_b)
                class_counts = classes[threshold_index_a:threshold_index_b].count(0)
                class_counts_other = threshold_index_b - threshold_index_a - class_counts
                
                # auc_y compute
                num_left_y[0] += class_counts
                num_right_y[0] -= class_counts
                num_left_y[1] += class_counts_other
                num_right_y[1] -= class_counts_other
                tpr_y = num_left_y[1] / (num_left_y[1] + num_right_y[1]) if num_left_y[1] + num_right_y[1] != 0 else 0
                fpr_y = num_left_y[0] / (num_left_y[0] + num_right_y[0]) if num_left_y[0] + num_right_y[0] != 0 else 1
                auc_y = (1 + tpr_y - fpr_y) / 2
                auc_y = abs(auc_y - 0.5) + 0.5
                
                # auc_z compute
                auc_z_max = 0
                for sens_group in range(self.n_sens_groups):
                    if self.n_senses_[sens_group]==2:
                        sens_counts = senses[sens_group][threshold_index_a:threshold_index_b].count(0)
                        sens_counts_other = threshold_index_b - threshold_index_a - sens_counts
                        num_left_z[sens_group][0] += sens_counts
                        num_right_z[sens_group][0] -= sens_counts
                        num_left_z[sens_group][1] += sens_counts_other
                        num_right_z[sens_group][1] -= sens_counts_other
                        tpr_z = num_left_z[sens_group][1] / (num_left_z[sens_group][1] + num_right_z[sens_group][1]) if (
                            num_left_z[sens_group][1] + num_right_z[sens_group][1] != 0
                        ) else 0
                        fpr_z = num_left_z[sens_group][0] / (num_left_z[sens_group][0] + num_right_z[sens_group][0]) if (
                            num_left_z[sens_group][0] + num_right_z[sens_group][0] != 0 
                        ) else 1
                        auc_z = (1 + tpr_z - fpr_z) / 2
                        auc_z = abs(auc_z - 0.5) + 0.5
                        if auc_z > auc_z_max:
                            auc_z_max = auc_z
                            
                    else:
                        # these loops cannot be merged because all counts must be updated prior to auc computation
                        for s in range(self.n_senses_[sens_group]):
                            sens_counts = senses[sens_group][threshold_index_a:threshold_index_b].count(s)
                            num_left_z[sens_group][s] += sens_counts
                            num_right_z[sens_group][s] -= sens_counts
                        for s in range(self.n_senses_[sens_group]):
                            num_left_z_s = num_left_z[sens_group][s]
                            num_right_z_s = num_right_z[sens_group][s]
                            num_left_z_not_s = sum(num_left_z[sens_group][:s] + num_left_z[sens_group][s+1:])
                            num_right_z_not_s = sum(num_right_z[sens_group][:s] + num_right_z[sens_group][s+1:])

                            tpr_z = num_left_z_s / (num_left_z_s + num_right_z_s) if num_left_z_s + num_right_z_s != 0 else 0
                            fpr_z = num_left_z_not_s / (num_left_z_not_s + num_right_z_not_s) if (
                                num_left_z_not_s + num_right_z_not_s != 0
                            ) else 1
                            auc_z = (1 + tpr_z - fpr_z) / 2
                            auc_z = abs(auc_z - 0.5) + 0.5
                            if auc_z > auc_z_max:
                                auc_z_max = auc_z
                                
                auc_z = auc_z_max
                    
                scaff = ((1-self.orthogonality)*auc_y) - (self.orthogonality*auc_z)
                split_score = (scaff - self.parent_scaff)
              
                if split_score > best_split_score:
                    if (sum(num_left_y) >= self.min_samples_leaf) and (sum(num_right_y) >= self.min_samples_leaf):
                        best_idx = idx
                        best_thr = threshold_b
                        best_split_score = split_score
                
                # updating threshold_index_a for next iteration
                threshold_index_a = threshold_index_b       

        return best_idx, best_thr
    
    def _grow_tree(self, X, y, z, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        total_samples = sum(num_samples_per_class)
        # Method logic up to creating the node remains unchanged
        node = Node(
            num_samples=y.size,
            predicted_class=predicted_class,
            num_samples_per_class=num_samples_per_class,
            probabilities=[n / total_samples for n in num_samples_per_class]
        )
     
        if depth < self.max_depth and y.size >= self.min_samples_split:
            idx, thr = self._best_split(X, y, z)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left, z_left = X[indices_left], y[indices_left], z[indices_left]
                X_right, y_right, z_right = X[~indices_left], y[~indices_left], z[~indices_left]
                if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._grow_tree(X_left, y_left, z_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, z_right, depth + 1)
        return node
    
    def _prepare_input_fit(self, X, y, z):
        X = pd.DataFrame(X)
        self.kbd = KBD(n_bins=self.n_bins, encode="ordinal")
        self.ohe = OHE(sparse_output=False, handle_unknown="ignore")
        # splitting columns based on data type
        num_columns = X.select_dtypes(exclude=['object']).columns
        bins_columns_loc = X[num_columns].nunique() > self.n_bins
        self.bin_columns = num_columns[bins_columns_loc].tolist()
        self.num_columns = num_columns[~bins_columns_loc].tolist()
        self.str_columns = X.select_dtypes(include=['object']).columns.tolist()
        # creating concatenation list based on transformations
        X_concat_list = []
        if len(self.str_columns):
            self.ohe.fit(X[self.str_columns])
            X_concat_list.append(self.ohe.transform(X[self.str_columns]))
        if len(self.bin_columns):
            self.kbd.fit(X[self.bin_columns])
            X_concat_list.append(self.kbd.transform(X[self.bin_columns]))
        if len(self.num_columns):
            X_concat_list.append(X[self.num_columns].values.astype(float))
        # concatenating X into numpy array
        X = np.concatenate(X_concat_list, axis=1)
        
        pre_vars = [y, z]
        for i in range(len(pre_vars)):
            pre_vars[i] = pd.DataFrame(pre_vars[i])
            for column in pre_vars[i].columns:
                pre_vars[i][column] = pre_vars[i][column].replace(
                    {pre_vars[i][column].unique()[j]: j for j in range(len(pre_vars[i][column].unique()))}
                )
            pre_vars[i] = pre_vars[i].values.astype(int)
        y, z = pre_vars
        
        return X, y.ravel(), z
    
    def _prepare_input_predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_concat_list = []
        if len(self.str_columns):
            X_concat_list.append(self.ohe.transform(X[self.str_columns]))
            
        if len(self.bin_columns):
            X_concat_list.append(self.kbd.transform(X[self.bin_columns]))
            
        if len(self.num_columns):
            X_concat_list.append(X[self.num_columns].values.astype(float))

        X = np.concatenate(X_concat_list, axis=1)
        
        return X
    
    def fit(self, X, y, z=None):
        
        if self.requires_data_processing:
            X, y, z = self._prepare_input_fit(X, y, z)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(np.unique(y))
       
        self.n_sens_groups = z.shape[1]
        self.n_senses_ = {
            sens_group: len(np.unique(z[:, sens_group])) for sens_group in range(self.n_sens_groups)
        }
        
        self.features_indices = np.random.choice(X.shape[1], self._get_max_features(X.shape[1]), replace=False)
        
        if self.bootstrap:
            resample_idx = np.random.choice(len(y), len(y), replace=True)
            X, y, z = X[resample_idx], y[resample_idx], z[resample_idx]
        
        self.tree_ = self._grow_tree(X, y, z)
        return self
        
    def _predict_proba(self, node, X):
        num_samples = len(X)
        probabilities = np.empty((num_samples, self.n_classes_))
        stack = [(node, np.arange(num_samples))]  # Convert slice to list of indices
        while stack:
            node, indices = stack.pop()
            mask = X[indices, node.feature_index] < node.threshold
            if node.left is None and node.right is None:  # Leaf node
                probabilities[indices] = node.probabilities
            else:
                left_indices = indices[mask]  # Filter indices using mask
                right_indices = indices[~mask]  # Filter indices using mask
                if len(left_indices):  # If any samples go to the left child
                    stack.append((node.left, left_indices))
                if len(right_indices):  # If any samples go to the right child
                    stack.append((node.right, right_indices))

        return probabilities
    
    def predict_proba(self, X):
        if self.requires_data_processing:
            X = self._prepare_input_predict(X)
        return self._predict_proba(self.tree_, X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
class FairRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        n_jobs=-1, 
        n_bins=256,
        max_depth=None, 
        bootstrap=True,
        random_state=42,
        orthogonality=0, 
        n_estimators=500,
        min_samples_leaf=1, 
        min_samples_split=2,
        max_features="sqrt",
        requires_data_processing=True,
    ):
        # forest-specific
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        # tree-specific
        self.n_bins=n_bins
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.requires_data_processing = requires_data_processing
        if isinstance(max_depth, type(None)):
            self.max_depth = np.inf
        else:
            self.max_depth = max_depth
            
        np.random.seed(self.random_state)
        
        base_rs = np.random.randint(1, 1e9)
        self.trees = [FairDecisionTreeClassifier(
            n_bins=self.n_bins,
            random_state=base_rs+i,
            max_depth=self.max_depth,
            bootstrap=self.bootstrap,
            max_features=self.max_features,
            orthogonality=self.orthogonality,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
        ) for i in range(n_estimators)]
    
    def _fit_trees_batch(self, tree_indices, X, y, z):
        fitted_trees_batch = []
        for i in tree_indices:
            tree = self.trees[i]
            tree.fit(X, y, z)
            fitted_trees_batch.append(tree)
        return fitted_trees_batch
    
    def _prepare_input_fit(self, X, y, z):
        X = pd.DataFrame(X)
        self.kbd = KBD(n_bins=self.n_bins, encode="ordinal")
        self.ohe = OHE(sparse_output=False, handle_unknown="ignore")
        # splitting columns based on data type
        num_columns = X.select_dtypes(exclude=['object']).columns
        bins_columns_loc = X[num_columns].nunique() > self.n_bins
        self.bin_columns = num_columns[bins_columns_loc].tolist()
        self.num_columns = num_columns[~bins_columns_loc].tolist()
        self.str_columns = X.select_dtypes(include=['object']).columns.tolist()
        # creating concatenation list based on transformations
        X_concat_list = []
        if len(self.str_columns):
            self.ohe.fit(X[self.str_columns])
            X_concat_list.append(self.ohe.transform(X[self.str_columns]))
        if len(self.bin_columns):
            self.kbd.fit(X[self.bin_columns])
            X_concat_list.append(self.kbd.transform(X[self.bin_columns]))
        if len(self.num_columns):
            X_concat_list.append(X[self.num_columns].values.astype(float))
        # concatenating X into numpy array
        X = np.concatenate(X_concat_list, axis=1)
        
        pre_vars = [y, z]
        for i in range(len(pre_vars)):
            pre_vars[i] = pd.DataFrame(pre_vars[i])
            for column in pre_vars[i].columns:
                pre_vars[i][column] = pre_vars[i][column].replace(
                    {pre_vars[i][column].unique()[j]: j for j in range(len(pre_vars[i][column].unique()))}
                )
            pre_vars[i] = pre_vars[i].values.astype(int)
        y, z = pre_vars
        
        return X, y.ravel(), z
    
    def _prepare_input_predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_concat_list = []
        if len(self.str_columns):
            X_concat_list.append(self.ohe.transform(X[self.str_columns]))
            
        if len(self.bin_columns):
            X_concat_list.append(self.kbd.transform(X[self.bin_columns]))
            
        if len(self.num_columns):
            X_concat_list.append(X[self.num_columns].values.astype(float))

        X = np.concatenate(X_concat_list, axis=1)
        
        return X
    
    def fit(self, X, y, z):
        
        if self.requires_data_processing:
            X, y, z = self._prepare_input_fit(X, y, z)
            
        self.classes_ = np.unique(y)
        # Determine the number of jobs
        n_jobs = self.n_jobs if self.n_jobs > 0 else joblib.cpu_count()
        
        # Generate batches
        batches = list(generate_batches(self.n_estimators, n_jobs))
        
        # Fit trees in batches in parallel
        trees_batches = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_trees_batch)(batch, X, y, z) for batch in batches)
        
        # Flatten the list of trees
        self.fitted_trees = [tree for batch in trees_batches for tree in batch]
        return self
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.fitted_trees])
        mode_pred, _ = st.mode(predictions, axis=0)
        return mode_pred.ravel()
        
    def predict_proba(self, X):
        if self.requires_data_processing:
            X = self._prepare_input_predict(X)
        proba_predictions = np.array([tree.predict_proba(X) for tree in self.fitted_trees])
        mean_proba = np.mean(proba_predictions, axis=0)
        return mean_proba
