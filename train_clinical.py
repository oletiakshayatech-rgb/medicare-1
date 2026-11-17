"""Clinical ML baseline. Needs YOUR authorised, de-identified CSV. Usage: python train_clinical.py data.csv label_col
Prints metrics only from a held-out patient-level test split."""
import sys, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from medscope.datasets import patient_split
df = pd.read_csv(sys.argv[1]); y = sys.argv[2]; tr, va, te = patient_split(df)
f = [c for c in df.select_dtypes("number").columns if c not in (y, "subject_id")]; med = tr[f].median()
m = LogisticRegression(max_iter=1000).fit(tr[f].fillna(med), tr[y])
print("held-out test AUROC:", roc_auc_score(te[y], m.predict_proba(te[f].fillna(med))[:, 1]))
