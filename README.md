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

<code>conda install tqdm --y;</code>
conda install sympy --y;</code>
conda install numpy --y;</code>
conda install pandas --y;</br>
conda install plotly --y;</br>
conda install jupyter --y;</br>
conda install seaborn --y;</br>
conda install networkx --y;</br>
conda install ipywidgets --y;</br>
conda install jupyterlab --y;</br>
conda install matplotlib --y;</br>
conda  install ipywidgets --y;</br>
conda install scikit-learn --y;</br>
conda install -c conda-forge cvxpy --y;</br>
conda install nodejs -c conda-forge --repodata-fn=repodata.json --y;</br>
pip install dccp;</br>
conda install -c conda-forge python-kaleido --y;</br>
jupyter nbextension enable --py widgetsnbextension;</br>
jupyter labextension install jupyterlab-plotly;</br>
jupyter labextension install @jupyter-widgets/jupyterlab-manager;</br>
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget;</br>
</code>
