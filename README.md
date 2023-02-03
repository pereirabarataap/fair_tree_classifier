# Fair Tree Classifier using Strong Demographic Parity

## Usage
```python
from fair_trees import *
from datasets import *

X, y, s = get_recidivism_gender(show=False)

# s ---> numpy array where each column must be 
#        a binary {0, 1} representation of each
#        unique sensitive attribute value

clf = FairRandomForestClassifier(
    orthogonality=0.5
)

clf.fit(X, y, s)
y_prob = clf.predict_proba(X)[:,1]

y_auc = roc_auc_score(y, y_prob) # ---> classification auc
s_auc = max(roc_auc_score(s, y_prob), 1 - roc_auc_score(s, y_prob)) # ---> sensitive auc
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
