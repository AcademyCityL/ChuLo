# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from tools.tokenizer import get_tokenizer
from tools.params import get_params
import sys
from transformers import AutoTokenizer, AutoModel
import dataset
from sklearn.decomposition import SparsePCA

old_dataset = ['mr']
# def process_quantum_emb(tokenizer,data,same_dir,save_name):
if __name__=="__main__":
    
    config_file = sys.argv[1] 
    config = get_params(config_file)   

    dataset = dataset.get_data(config['DATA'],'gpu',config['TRAIN'])
    all_embs = {}
    if config['source'] == 'bert': # token level embedding
        tokenizer = AutoTokenizer.from_pretrained(config['bert_name'])
        model = AutoModel.from_pretrained(config['bert_name'])
        for split,data in dataset.data.items():
            x = data['x']
            for sentence in x:
                input = tokenizer([sentence])
                output  = model(**input)
                last_hidden_state = output['hidden_states'][-1]
                for i in range(len(input['input_ids'][0])):
                    id = input['input_ids'][i]
                    if id not in all_embs:
                        all_embs[id] = [last_hidden_state[0,i,:].item()]
                    else:
                        all_embs[id].append(last_hidden_state[0,i,:].item())

    emb_list = []
    for id, embs in all_embs.items():
        for embedding in embs:
            emb_list.append(embedding)

    sparsepca = SparsePCA
    run(config)
