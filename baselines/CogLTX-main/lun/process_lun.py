import re
import json
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from itertools import chain
import os
import sys
import pickle
import logging
import pdb
from bisect import bisect_left
import string
import pandas as pd
import numpy as np
import io

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
from buffer import Buffer, Block
from utils import DEFAULT_MODEL_NAME
from hotpotqa.cogqa_utils import find_start_end_after_tokenized
from sklearn.datasets import fetch_20newsgroups



def process(text, label,  split_name):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    cnt, batches = 0, []
    for i in tqdm(range(len(text))):
    # for i in range(20):
        d, l = text[i], label[i]
        label_name = label[i]
        qbuf, cnt = Buffer.split_document_into_blocks([tokenizer.cls_token], tokenizer, cnt=cnt, hard=False, properties=[('label_name', label_name), ('label', l), ('_id', str(i)), ('blk_type', 0)])
        dbuf, cnt = Buffer.split_document_into_blocks(tokenizer.tokenize(d), tokenizer, cnt, hard=False)
        batches.append((qbuf, dbuf))
    with open(os.path.join(root_dir, 'data', f'lun_{split_name}.pkl'), 'wb') as fout: 
        pickle.dump(batches, fout)
    return batches

# add the code for LUN data
def prepare_lun_data(lun_data_path= '../../data/lun/'):
    file_name = {
        'train': 'xtrain.txt',
        'dev': 'xdev.txt',
        'test': 'xtest.csv',
    }
    text_set = {}
    label_set = {}
    for split in ['train','dev','test']:
        file_path = os.path.join(lun_data_path, file_name[split])
        text = []
        labels = []
        if split in ['train','dev']:
            with io.open(file_path, 'r', encoding='latin-1') as f:
                for line in f.readlines():
                    text.append(line.split(None,1)[1])
                    labels.append(line.split(None,1)[0])
        elif split == 'test':
            data = pd.read_csv(file_path,header=None)
            text.extend(data[1].values.tolist())
            labels.extend(data[0].values.tolist())
        labels = [ int(i) for i in labels]
        text_set[split] = text
        label_set[split] = labels
    

    return text_set, label_set

text_set, label_set = prepare_lun_data()
process(text_set['train'], label_set['train'], 'train')
process(text_set['dev'], label_set['dev'], 'val')
process(text_set['test'], label_set['test'], 'val')