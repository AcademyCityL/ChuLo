# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def get_distance_func(dis_name,**kwargs):
    if dis_name == "cosine_similarity":
        distance_func = cosine_distance(**kwargs)
    return distance_func

def cosine_distance(**kwargs):
    # distance [0,2] (cosine [-1,1]), 0 is closest
    cosine_sim = nn.CosineSimilarity(**kwargs)
    def func(a, b):
        return (1 - cosine_sim(a, b))
    return func