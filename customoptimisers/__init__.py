# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def get_optimizer(model, params):    
    print("optimizer name: ", params['name'])
    print("optimizer params: ", params['kwargs'])
    if params['name'] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **params['kwargs'])
    elif params['name'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **params['kwargs'])
    elif params['name'] == "adam":
        optimizer = torch.optim.SGD(model.parameters(), **params['kwargs'])
    else:
        raise Exception("optimized not supported: {}".format(params['name']))
    return optimizer