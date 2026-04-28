import pickle
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
with open("../results/untouched_perm_oss20b.pkl", "rb") as f:
    r = pickle.load(f)

df = pd.read_csv('../data/Perm2.csv')

probs = np.asarray([np.mean([1 if 'Yes' in k or 'yes' in k else 0 for k in rr]) for rr in r])

labels = df['label'].values
print(roc_auc_score(labels, probs)) 
