import pandas as pd
from ollama import chat
from ollama import ChatResponse
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle


model_name = 'gpt-oss:20b'


data = pd.read_csv('data/Perm2.csv')

prompt_template = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Only Answer Yes or No.'
probs = []
output_objects = []

results =  []
for i in tqdm(range(len(data))):
    support = data.iloc[i]['Support']        
    convo = data.iloc[i]['convo']
    summ = data.iloc[i]['inf_summ']
    index = data.iloc[i]['index']
    dataset = data.iloc[i]['Dataset'] 

    sents = sent_tokenize(support)
    prompt = prompt_template.replace("{{premise}}", convo) 
    prompt = prompt.replace("{{hypothesis}}", summ) 
    messages = [
        {"role": "user", "content": prompt}
    ]
    tmp = []
    for k in range(10):
        response: ChatResponse = chat(model=model_name, messages=messages)
        tmp.append(response.message.content) 
    results.append(tmp) 
    with open('results/untouched_perm_oss20b.pkl', "wb") as f:
        pickle.dump(results, f)
