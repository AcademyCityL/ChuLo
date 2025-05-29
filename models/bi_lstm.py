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

class BI_LSTM(nn.Module):
    def __init__(self,config,params):
        super(BI_LSTM, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()

    def init_attr_from_config(self):
        model_config = self.global_config['MODEL']
        self.hidden_dim = model_config.get('hidden_dim',1024)
        self.layers = model_config.get('layers',1)
        self.dropout = model_config.get('dropout',0.1)
        self.embedding_params = model_config.get('embedding',{})

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
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.layers,\
                          batch_first =True, bidirectional=True)
        self.drop = nn.Dropout(p=self.dropout)
    
    def forward(self, input, **kwargs):
        x = self.embeddinglayer(input)    
        x, (h_n, c_n) = self.lstm(x)
        # concat the hidden state
        hidden_out = torch.cat((h_n[-2,:,:],h_n[-1,:,:]),1)
        hidden_out = self.drop(hidden_out)
        return hidden_out