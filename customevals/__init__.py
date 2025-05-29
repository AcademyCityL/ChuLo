# -*- coding: utf-8 -*-
from customevals.metrics import *

def get_metric(params,data):
    print("metric name: ", params['name'])
    print("metric params: ", params['kwargs'])
    if params['name'] == 'mapmrr':
        metric = MeanScoreRank(params['name'],data,**params['kwargs'])
    elif params['name'] == 'acc':
        metric = Accuracy(params['name'],**params['kwargs'])
    elif params['name'] == 'loss':
        metric = Loss(params['name'],)
    elif params['name'] == 'f1score':
        metric = F1score(params['name'],**params['kwargs'])
    elif params['name'] == 'qaacc':
        metric = QAAccuracy(params['name'],data, **params['kwargs'])
    elif params['name'] == 'hotpotqa':
        metric = HotpotQA_metrics(params['name'],data, **params['kwargs'])
    else:
        raise Exception("metric not supported: {}".format(params['name']))
    return metric

def get_metrics(params,data):
    # print(params)
    metricers = []
    best_metric = None
    for metric_config in params:
        metricer = get_metric(metric_config,data)
        if metric_config['best'] == True:
            best_metric = metricer
        metricers.append(metricer)
    if best_metric is None:
        best_metric = metricers[0]
    return metricers,best_metric

def show_metrics(metricers):
    for metricer in  metricers:
        metricer.show(end_char = '')
    print("")

def reset_metrics(metricers):
    for metricer in  metricers:
        metricer.reset()

def update_metrics(metricers,input, output, targets, **kwargs):
    for metricer in  metricers:
        metricer.update(input, output, targets, **kwargs)

def compute_metrics(metricers):
    for metricer in  metricers:
        metricer.compute()