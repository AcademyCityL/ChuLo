# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import pytorch_lightning as pl
from customlayers.embedding import EmbeddingLayer
from tools.tokenizer import get_tokenizer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Simple_Text_NN(nn.Module):
    def __init__(self,config,params):
        super(Simple_Text_NN, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()

    def init_attr_from_config(self):
        model_config = self.global_config['MODEL']
        self.hidden_dim = model_config.get('output_dim',1024)
        self.layers = model_config.get('layers',1)
        self.dropout = model_config.get('dropout',0.1)
        self.embedding_params = model_config.get('embedding',{})
        self.mean_dim = model_config.get('mean_dim',None)

        data_config = self.global_config['DATA']
        self.tokenizer_type = data_config.get('tokenizer_type','non_bert')
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_real_name = 'files/tokenizers/{}/'.format(self.tokenizer_name)

    def init_model(self):
        # init embedding
        if self.tokenizer_type == 'non_bert':
            self.embeddinglayer = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.params['vocab'],\
                    **self.embedding_params['kwargs'])
        elif self.tokenizer_type == 'bert':
            self.tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
            self.embeddinglayer = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.tokenizer.get_vocab(),\
                    **self.embedding_params['kwargs'])
       
        self.embedding_dim = self.embeddinglayer.embedding.emb_dim
        self.linear_relu_stack = nn.Sequential()
        init_dim = self.embedding_dim
        for i in range(self.layers):
            self.linear_relu_stack.append(nn.Linear(init_dim, self.hidden_dim))
            self.linear_relu_stack.append(nn.ReLU())
            init_dim = self.hidden_dim
           
        self.drop = nn.Dropout(p=self.dropout)

    
    def forward(self, input, **kwargs):
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, sl]
        # def print_attn_score(hidden_out):
        #     with torch.no_grad:
        #         attn = hidden_out@hidden_out.transform(1,2)
        #         attn = torch.nn.functional.softmax(attn, dim=-1)
        #         print('attn  ',attn)
                
        hidden_out = self.embeddinglayer(input)  
        # print(input[:2]) 
        # print(len(hidden_out[hidden_out!=0]),hidden_out.shape)
        # print(hidden_out[:2]) 
        # print_attn_score(hidden_out)
        # for layer in self.linear_relu_stack:
        #     hidden_out = layer(hidden_out)
        hidden_out = self.linear_relu_stack(hidden_out)
        # hidden_out[kwargs['attention_mask'] == 0] = 0
        # concat the hidden state
            # print(hidden_out[:2]) 
            # print_attn_score(hidden_out)
        hidden_out = self.drop(hidden_out)
        if self.mean_dim != None:
            # self.mean_dim use a negative number
            denom = torch.sum(kwargs['attention_mask'], -1, keepdim=True)
            hidden_out = torch.sum(hidden_out * kwargs['attention_mask'].unsqueeze(-1), dim=self.mean_dim) 
            hidden_out = hidden_out / denom 

        # print(hidden_out[:2])
        return hidden_out