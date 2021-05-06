# fair_tree_learning

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
