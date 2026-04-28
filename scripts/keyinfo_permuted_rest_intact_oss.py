import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from ollama import chat
from ollama import ChatResponse
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
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
block = 15


model_name = 'gpt-oss:20b'


data = pd.read_csv('data/Perm2.csv')

prompt_template = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Only Answer Yes or No.'





head_results = []
middle_results = []
tail_results = []
for i in tqdm(range(len(data))):
    support = data.iloc[i]['Support']        
    convo = data.iloc[i]['convo']
    summ = data.iloc[i]['inf_summ']
    index = data.iloc[i]['index']
    dataset = data.iloc[i]['Dataset'] 

    
    sents = sent_tokenize(convo)
    rest_parts = []
    if "|" not in support: 
        start_id = convo.find(support)    
        assert start_id != -1
        if start_id == 0:
            rest_sent = convo[len(support):]
        elif start_id + len(support) == len(convo):
            rest_sent = convo[:start_id]
        else:
            rest_sent = convo[:start_id] + " " + convo[start_id + len(support):]
        joined_support = support
    else:
        supports = [s.strip() for s in support.split("|")]
        start_ids = [convo.find(s) for s in supports] 
        supports = [supports[new_id] for new_id in np.argsort(start_ids)]
        start_ids.sort()
        start_ids2 = [convo.find(s) for s in supports] 
        for i in range(len(start_ids)):
            assert start_ids[i] == start_ids2[i] 
        try:
            for i in range(len(start_ids) - 1):
                assert start_ids[i] < start_ids[i + 1]
        except:
            print(dataset, index, supports, start_ids, len(supports))
            for s in supports:
                print(s)
        prev = 0
        for s in supports:
            start_id = convo.find(s)    
            assert start_id != -1
            rest_parts.append(convo[prev:start_id])
            prev = start_id + len(s)             
        rest_sent = " ".join(rest_parts)
    
        joined_support  = " ".join(supports)
    joined_support = joined_support.split(" ")
    if block > len(joined_support):
        blocktg = len(joined_support)
    else:
       blocktg = block
    block_order = np.arange(blocktg)
    np.random.shuffle(block_order)
    new_convo  = ""
    chunk_size = len(joined_support) // blocktg
    for kk, o in enumerate(block_order):
        if o != blocktg - 1:
            new_convo += " ".join(joined_support[o * chunk_size : (o + 1) * chunk_size])
        else:
            new_convo += " ".join(joined_support[o * chunk_size : ])
        new_convo += " "  
    

    joined_support = new_convo 
    head_convo = joined_support + " " + rest_sent
    tail_convo = rest_sent + " " + joined_support
    mid_point = len(rest_sent) // 2
    mid_convo = rest_sent[:mid_point] + " " + joined_support +  " " + rest_sent[mid_point:] 
    
    prompt = prompt_template.replace("{{premise}}", head_convo) 
    prompt = prompt.replace("{{hypothesis}}", summ) 
    messages = [
        {"role": "user", "content": prompt}
    ]
    tmp = []
    for i in range(10):
        response: ChatResponse = chat(model=model_name, messages=messages)
        tmp.append(response.message.content)
    head_results.append(tmp) 


    prompt = prompt_template.replace("{{premise}}", mid_convo) 
    prompt = prompt.replace("{{hypothesis}}", summ) 
    messages = [
        {"role": "user", "content": prompt}
    ]
    tmp = []
    for i in range(10):
        response: ChatResponse = chat(model=model_name, messages=messages)
        tmp.append(response.message.content)
    middle_results.append(tmp) 


    prompt = prompt_template.replace("{{premise}}", tail_convo) 
    prompt = prompt.replace("{{hypothesis}}", summ) 
    messages = [
        {"role": "user", "content": prompt}
    ]
    tmp = []
    for i in range(10):
        response: ChatResponse = chat(model=model_name, messages=messages)
        tmp.append(response.message.content)
    tail_results.append(tmp) 


    with open("results/key_shuffeled" + str(block)+"-20b.pkl", "wb") as f:
        pickle.dump([head_results, middle_results, tail_results], f)

