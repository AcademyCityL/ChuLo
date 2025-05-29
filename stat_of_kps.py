import pandas as pd
import os
import numpy as np
import json
import math
import typing
from typing import List,Dict,Any
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer
import nltk
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, Linear
import math
import datasets
import dataset
from tools.params import get_params
import argparse

def save_box(pddata,name,data_name):
    boxplot = pddata.boxplot()
    fig = boxplot.get_figure()
    fig.suptitle(name)
    # fig.get_axes()[0].set_xlabel("test")
    fig.savefig('results/stat/{}_{}.png'.format(data_name,name),dpi=300)
    fig.clf()

def prepare_envs():
    if not os.path.exists('results/stat/'):
        os.mkdir('results/stat/')
    if not os.path.exists('results/cache/'):
        os.mkdir('results/cache/')
    if not os.path.exists('results/cache/tokenized_results/'):
        os.mkdir('results/cache/tokenized_results/')
    if not os.path.exists('results/cache/vocabs/'):
        os.mkdir('results/cache/vocabs/')

if __name__=="__main__":
    prepare_envs()
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default='ERROR')
    args = parser.parse_args()
    config_file = args.config
    data_name = config_file.split('_')[0].split('/')[-1]
    config = get_params(config_file)
    data_config = config['DATA']

    daobj = dataset.get_data(data_config,'gpu',config)
    print("***************Stat before tokenisation (after clean the dataset)***************:") 
    print("The stat of length (by split(' ') ) of each document is: ")
    stat = []
    for split in ['train','test']:
        for doc in daobj.datasets[split].data:
            stat.append(len(doc.split(' ')))
    pd_data = pd.DataFrame(stat)
    print(pd_data.describe())
    save_box(pd_data,"bt_document_length",data_name)


    all_kps = {}
    cache_prefix = {'hp':'hyperpartisan','imdb':'imdb'}
    for split in ['train','test']:
        f = 'results/cache/key_phrase_split/{}_key_phrase_split_{}_512_kps.pt'.format(cache_prefix[data_name],split)
        a = torch.load(f)
        for k, v in a.items():
            if k not in all_kps:
                all_kps[k] = v
            else:
                if type(v) == list:
                    all_kps[k].extend(v)
                else:
                    all_kps[k] = max(all_kps[k],v)
    # print(all_kps.keys())
    print("Total number of key phrases occurances in the first 512 words of each document (by split(' ')): ")
    pd_data = pd.DataFrame([all_kps['chunk_num_stastics'][i]-1 for i in range(len(all_kps['chunk_num_stastics']))])
    print(pd_data.describe())
    save_box(pd_data,"bt_kp_num_in512",data_name)
    print("The number of unique key phrases in the first 512 words of each document (by split(' ')): ")
    all_kp_stat = []
    for all_kp in all_kps['all_kp']:
        all_kp_stat.append(len(all_kp))
    pd_data = pd.DataFrame(all_kp_stat)
    print(pd_data.describe())
    save_box(pd_data,"bt_ukp_num_in512",data_name)
    print("The stat of which the number of unique key phrases is less than 15: ")
    outlier = []
    for i in all_kp_stat:
        if i!= 15:
            outlier.append(i)
    pd_data = pd.DataFrame(outlier)
    print(pd_data.describe())
    save_box(pd_data,"bt_outlier_kp_num_in512",data_name)
    print(outlier)

    print("***************Stat after tokenisation (after clean the dataset)***************:") 
    print("The stat of token lenth of each document is: ")
    original_doc = []
    tokenized_stat = []
    for split in ['train','test']:
        for doc in daobj.datasets[split].data:
            original_doc.append(doc)
    bs = 512
    for idx in range(0,len(original_doc),bs):
        tokenized_doc = daobj.tokenizer(
                        original_doc[idx:idx+bs],
                        truncation=False,
                        padding=False,
                        return_attention_mask = True,
                        return_offsets_mapping=True,
                        return_special_tokens_mask = True,
                        return_token_type_ids = True,
                        return_tensors = None,
                    )
        for i in tokenized_doc['input_ids']:
            tokenized_stat.append(len(i))
    pd_data = pd.DataFrame(tokenized_stat)
    print(pd_data.describe())
    save_box(pd_data,"at_document_length",data_name)
    print("The number of tokens in the first 512 words of each document: ")
    tokenzed_512_stat = []
    doc_512 = []
    for i in original_doc:
        doc_512.append(' '.join(i.split(' ')[:512]))
    for idx in range(0,len(doc_512),bs):
        tokenized_doc = daobj.tokenizer(
                        doc_512[idx:idx+bs],
                        truncation=False,
                        padding=False,
                        return_attention_mask = True,
                        return_offsets_mapping=True,
                        return_special_tokens_mask = True,
                        return_token_type_ids = True,
                        return_tensors = None,
                    )
        for i in tokenized_doc['input_ids']:
            tokenzed_512_stat.append(len(i))
    pd_data = pd.DataFrame(tokenzed_512_stat)
    print(pd_data.describe())
    save_box(pd_data,"at_token_num_in512",data_name)
    print("The number of keyphrase tokens in the first 512 words of each document: ")
    loc_tokenized_stat = []
    rep_kp_doc_512 = []
    all_kps_512 = []
    print("len ",len(doc_512),)
    for i in range(len(doc_512)):
        kps = all_kps['all_kp'][i]
        rep_kp_doc_512.append(doc_512[i])
        for kp in kps:
            rep_kp_doc_512[i] = rep_kp_doc_512[i].replace(kp, '[LOC]')
        all_kps_512.append(rep_kp_doc_512[i].count('[LOC]'))
    # print("doc_512[0] ",doc_512[0])
    # print("rep_kp_doc_512[0] ",rep_kp_doc_512[0],len(rep_kp_doc_512),len(doc_512))
    for idx in range(0,len(rep_kp_doc_512),bs):
        tokenized_doc = daobj.tokenizer(
                        rep_kp_doc_512[idx:idx+bs],
                        truncation=False,
                        padding=False,
                        return_attention_mask = True,
                        return_offsets_mapping=True,
                        return_special_tokens_mask = True,
                        return_token_type_ids = True,
                        return_tensors = None,
                    )
        for i in tokenized_doc['input_ids']:
            loc_tokenized_stat.append(len(i))
    kp_tokenized_stat = []
    # print("len(loc_tokenized_stat) ",len(tokenzed_512_stat),len(loc_tokenized_stat),len(all_kps_512))
    for i in range(len(loc_tokenized_stat)):
        kp_tokenized_stat.append(tokenzed_512_stat[i] - loc_tokenized_stat[i] + all_kps_512[i])
    pd_data = pd.DataFrame(kp_tokenized_stat)
    print(pd_data.describe())
    save_box(pd_data,"at_kp_num_in512",data_name)
    print("The number of non keyphrase tokens in the first 512 words of each document: ")
    nonkp_tokenized_stat = []
    for i in range(len(loc_tokenized_stat)):
        nonkp_tokenized_stat.append(loc_tokenized_stat[i] - all_kps_512[i])
    pd_data = pd.DataFrame(nonkp_tokenized_stat)
    save_box(pd_data,"at_non_kp_num_in512",data_name)
    print(pd_data.describe())