# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import AutoModel,AutoModelForSequenceClassification

class Bert(nn.Module):
    def __init__(self, data, option='normal',chunk = False, ret_features=False,bert_name = "bert-base-cased", num_labels = 2, task = "text classification"):
        super(Bert, self).__init__()
        self.bert_name = bert_name
        self.num_labels = num_labels
        self.task = task
        if self.task == "text classification":
            self.bert = AutoModelForSequenceClassification.from_pretrained(bert_name,num_labels=num_labels)
        self.option = option
        self.ret_features = ret_features
        self.chunk = chunk

    def _forward(self, input):
        # print(input['input_ids'][:,0])
        overflow_to_sample_mapping = input.pop('overflow_to_sample_mapping')
        if "longformer" in self.bert_name:
            global_attention_mask = torch.zeros(
                input['input_ids'].shape, dtype=torch.long, device=input['input_ids'].device
            )  # initialize to global attention to be deactivated for all tokens
            global_attention_mask[:,0] = 1
            input['global_attention_mask'] = global_attention_mask
        output = self.bert(**input,output_hidden_states=True)
#        output = torch.log10(output)
        if self.ret_features == False:
            # print(output['hidden_states'][0].shape,output['hidden_states'][1].shape,output['logits'].shape)
            output = output['logits']
        else: 
            # print(output['hidden_states'][0][:,0,:])
            output = output['hidden_states'][-1][:,0,:]
#        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        real_num = len(set(overflow_to_sample_mapping))
        if self.chunk == True:
            slices = [0]*real_num
            for i in overflow_to_sample_mapping:
                slices[i] += 1
            output = torch.split(output,slices)
            tmp = []
            for o in output:
                tmp.append(o.mean(dim=0,keepdim=True))
            output = torch.cat(tmp)
        return output

    def forward(self, input):
        if self.option == "normal":
            output = self._forward(input[0])
        elif self.option == "triplet":
            output = []
            for single_input in input[:-1]:
                output.append(self._forward(single_input))
            output = tuple(output)

        return output