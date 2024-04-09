# Fair Tree Classifier using Strong Demographic Parity

## Requirements
<code>pip install -r requirements.txt</code>

## 3D Figures
https://htmlpreview.github.io/?https://github.com/pereirabarataap/fair_tree_classifier/main/3d/index.html


## Usage
```python
import joblib
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold as SKF
from fair_trees import FairRandomForestClassifier as FRFC, sdp_score

datasets = joblib.load("datasets.pkl")

results_data = []
for dataset in tqdm(datasets):
    X = datasets[dataset]["X"]
    y = datasets[dataset]["y"]
    z = datasets[dataset]["z"]
    
    fold = 0
    skf = SKF(n_splits=5, random_state=42, shuffle=True)
    # ensuring stratified kfold w.r.t. y and z
    splitter_y = pd.concat([y, z], axis=1).astype(str).apply(
        lambda row:
            row[y.name] + "".join([row[col] for col in z.columns]),
        axis=1
    ).values
    for train_idx, test_idx in tqdm(skf.split(X,splitter_y), desc=f"dataset={dataset} | processing folds", leave=False):
        
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        z_train, z_test = z.loc[train_idx], z.loc[test_idx]
        
        for theta in tqdm(np.linspace(0,1,11).round(1), desc=f"fold={fold} | fitting thetas", leave=False):
            clf = FRFC(
                n_jobs=-1,
                n_bins=256,
                max_depth=None,
                bootstrap=True,
                random_state=42,
                n_estimators=500,
                min_samples_leaf=1,
                min_samples_split=2,
                orthogonality=theta,
                max_features="sqrt",
                requires_data_processing=True
            ).fit(X_train, y_train, z_train)
            y_prob = clf.predict_proba(X_test)[:,1]

            auc = roc_auc_score(y_test, y_prob)

            sdp_min = np.inf
            for sens_att in z.columns:
                if len(np.unique(z_test[sens_att]))==2:
                    sens_val = np.unique(z_test[sens_att])[0]
                    z_true = z_test[sens_att]==sens_val
                    sdp = sdp_score(z_true, y_prob)
                    if sdp < sdp_min:
                        sdp_min = sdp
                else:
                    for sens_val in np.unique(z_test[sens_att]):
                        z_true = z_test[sens_att]==sens_val
                        sdp = sdp_score(z_true, y_prob)
                        if sdp < sdp_min:
                            sdp_min = sdp
            
            data_row = [dataset, fold, theta, auc, sdp_min]
            results_data.append(data_row)
            
        fold += 1
        
results_df = pd.DataFrame(
    data=results_data,
    columns=["dataset", "fold", "theta", "performance", "fairness"]
)

fig, ax = plt.subplots(1,1,dpi=100, figsize=(8,4))
sb.lineplot(
    data=results_df.groupby(by=["dataset", "theta"]).mean(),
    x="fairness",
    y="performance", 
    hue="dataset",
    ax=ax
)
plt.show()
```

## Reproducibility
To reproduce the entirety of our experiments, simply run the <code>reproduce.ipynb</code> jupyter notebook in a Python environment.</br>
Expected compute time: <code>24 hours</code> using <code>64 AMD EPYC 7601-kernen @ 2.40 GHz (128 threads)</code>.</br>
Please run the following snippets prior to the run:

<code>pip install openml;</code>

<code>conda install tqdm --y;</code>

<code>conda install numpy --y;</code>

<code>conda install pandas --y;</code>

<code>conda install plotly --y;</code>

<code>conda install jupyter --y;</code>

<code>conda install seaborn --y;</code>

<code>conda install jupyterlab --y;</code>

<code>conda install matplotlib --y;</code>

<code>conda install ipywidgets --y;</code>

<code>conda install scikit-learn --y;</code>

<code>conda install -c conda-forge python-kaleido --y;</code>

<code>conda install nodejs -c conda-forge --repodata-fn=repodata.json --y;</code>

<code>jupyter nbextension enable --py widgetsnbextension;</code>

<code>jupyter labextension install jupyterlab-plotly;</code>

<code>jupyter labextension install @jupyter-widgets/jupyterlab-manager;</code>

<code>jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget;</code>
