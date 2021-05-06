# Fair Tree Learning

## Usage
```python
from ftl import *
from datasets import *

X, y, s = get_recidivism_gender(show=False)

clf = FairRandomForestClassifier(
    n_bins=2,
    n_jobs=-1,
    max_depth=2,
    n_samples=1.0,
    bootstrap=True,
    random_state=42,
    n_estimators=500,
    orthogonality=0.5,
    max_features='auto',
)

clf.fit(X, y, s)
y_prob = clf.predict_proba(X)[:,1]

y_auc = roc_auc_score(y, y_prob)
s_auc = sns_auc_score(s, y_prob)
```

## Reproduce
To reproduce the entirety of our experiments, simply run the <code>reproduce.ipynb</code> jupyter notebook in a Python environment.</br>
Please note that some modules should be installed prior to the run:

<code>
conda install tqdm --y;
conda install sympy --y;
conda install numpy --y;
conda install pandas --y;
conda install plotly --y;
conda install jupyter --y;
conda install seaborn --y;
conda install networkx --y;
conda install ipywidgets --y;
conda install jupyterlab --y;
conda install matplotlib --y;
conda  install ipywidgets --y;
conda install scikit-learn --y;
conda install -c conda-forge cvxpy --y;
conda install nodejs -c conda-forge --repodata-fn=repodata.json --y;
pip install dccp;
conda install -c conda-forge python-kaleido --y;
jupyter nbextension enable --py widgetsnbextension;
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager;
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget;
</code>
