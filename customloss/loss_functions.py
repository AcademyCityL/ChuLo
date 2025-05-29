# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from tools.distance import get_distance_func

def ce_loss(**kwargs):
    ce = nn.CrossEntropyLoss(**kwargs)
    def loss_func(input, output,targets):
        loss = ce(output,targets)
        return loss
    return loss_func

def batch_pairwise_loss(**kwargs):
    def loss_func(input, output,targets):
        targets = nn.functional.one_hot(targets, num_classes = output.shape[1])
        pos = torch.mean(targets * output)
        neg = torch.mean((1. - targets) * output)
        return torch.maximum(neg - pos + kwargs['margin'], torch.tensor(0.))
    return loss_func

def triplet_loss(**kwargs):
    def loss_func(input, output,targets):
        # print("len(output)  ",len(output))
        if len(output) != 3:
            return 0
        qf,paf,naf = output
        distance = get_distance_func(kwargs['distance']['name'],**kwargs['distance']['kwargs'])
        if kwargs['distance']['name'] == 'cosine_similarity':
            pos_dis = distance(qf,paf)
            neg_dis = distance(qf,naf)
        basic_loss = pos_dis-neg_dis+kwargs['margin']
        # print("basic_loss  ",basic_loss)
        # print(basic_loss)
        # print(torch.maximum(basic_loss, torch.tensor(0.)))
        return  torch.mean(torch.maximum(basic_loss, torch.tensor(0.)))
    return loss_func

def hotpotqa_loss(**kwargs):
    nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=kwargs['ignore_index'])
    nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=kwargs['ignore_index'])
    nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=kwargs['ignore_index'])
    def loss_func(input, output,targets):
        y1,y2,ids,q_type,is_support,eval_file = targets['y1'],targets['y2'],targets['ids'],\
        targets['q_type'],targets['is_support'],targets['eval_file']
        logit1, logit2, predict_type, predict_support, yp1, yp2 = output
        # print(predict_type.shape,q_type, logit1.shape,y1,logit2.shape,y2)
        loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / input['context_idxs'].size(0)
        loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
        loss = loss_1 + kwargs['sp_lambda'] * loss_2
        return loss
    return loss_func