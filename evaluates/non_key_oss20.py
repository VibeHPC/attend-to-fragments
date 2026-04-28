import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import float16
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import copy
import json
import pickle
import itertools
import json
import pandas as pd
import pickle
import torch
from sklearn.metrics import roc_auc_score

with open("../results/non_key_oss20b.pkl", "rb") as f:
    r = pickle.load(f)

df = pd.read_csv('../data/Perm2.csv')

labels = df['label'].values

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

probs = np.asarray([np.mean([1 if 'Yes' in k or 'yes' in k else 0 for k in rr]) for rr in r])



print("nonkey", roc_auc_score(labels, probs))
