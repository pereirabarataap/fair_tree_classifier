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
