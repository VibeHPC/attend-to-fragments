import pickle
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
with open("../results/word_scrammble_key_intact_oss20.pkl", "rb") as f:
    r = pickle.load(f)

df = pd.read_csv('../data/Perm2.csv')

head_probs = np.asarray([np.mean([1 if 'Yes' in k or 'yes' in k else 0 for k in rr]) for rr in r[0]])
mid_probs = np.asarray([np.mean([1 if 'Yes' in k or 'yes' in k else 0 for k in rr]) for rr in r[1]])
tail_probs = np.asarray([np.mean([1 if 'Yes' in k or 'yes' in k else 0 for k in rr]) for rr in r[2]])

labels = df['label'].values[:len(head_probs)]
print(roc_auc_score(labels, head_probs)) 
print(roc_auc_score(labels, mid_probs)) 
print(roc_auc_score(labels, tail_probs))
print(np.mean([roc_auc_score(labels, head_probs), roc_auc_score(labels, mid_probs), roc_auc_score(labels, tail_probs)])) 
