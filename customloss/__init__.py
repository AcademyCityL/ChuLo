# -*- coding: utf-8 -*-
from customloss.loss_functions import *
import torch
import torch.nn as nn

def get_loss(params):
    print("loss name: ", params['name'])
    print("loss params: ", params['kwargs'])
    if params['name'] == 'crossentropy':
        return ce_loss(**params['kwargs'])
    elif params['name'] == 'batchpairloss':
        criterion = batch_pairwise_loss(**params['kwargs'])
    elif params['name'] == 'tripletloss':
        criterion = triplet_loss(**params['kwargs'])
    elif params['name'] == 'hotpotqa_loss':
        criterion = hotpotqa_loss(**params['kwargs'])
    else:
        raise Exception("loss not supported: {}".format(params['name']))
    return criterion