# -*- coding: utf-8 -*-
import os
import dataset
import models
import torch
import customloss
import customoptimisers
from customevals import *
from tools.params import get_params
import sys
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger,TensorBoardLogger
from tools.progressbar import MyTQDMProgressBar
import yaml
import argparse
from tensorboard.backend.event_processing import event_accumulator

def current_exp_time(exp_result_dir):
    '''
    Experiment time starts from 0
    '''
    log_path = os.path.join(exp_result_dir,'lightning_logs')
    all_exps = os.listdir(log_path)
    count = len(all_exps)
    return count

def run_one_exp(config,test_params = None):
    print('Start experiment: {}, seed: {}'.format(config['name'],config['seed']))
    if test_params is not None:
        print('Test params: ',test_params)
    seed_everything(config['seed'])
    exp_config = config['EXPERIMENT']
    test_checkpoint = exp_config.get('test_checkpoint','best')
    retrain_ckp = exp_config.get('retrain_ckp', True)
    # self.do_train = exp_config.get('do_train', True)
    # self.do_val = exp_config.get('do_val', True)
    # self.do_test = exp_config.get('do_test', False)
    # self.do_predict = exp_config.get('do_predict', True)
    exp_result_dir = exp_config.get('save_path',"results/experiments/")
    TBlogger = TensorBoardLogger(save_dir=exp_result_dir)
    # Clogger = CSVLogger(save_dir=exp_result_dir)
    ori_exp_name = config.get('name',"default_name")
    current_time = current_exp_time(exp_result_dir)
    time_exp_name = '{}_{}'.format(ori_exp_name,current_time)
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_result_dir,
        filename = time_exp_name+'_best_{epoch:04d}',
        save_top_k=exp_config.get('save_top_k',1),
        verbose = exp_config.get('verbose',True),
        monitor = exp_config.get('monitor','val_loss'),
        save_last=exp_config.get('save_last',False),
        mode = exp_config.get('mode','min'),
        save_weights_only  = exp_config.get('save_weights_only',True),
        every_n_epochs = exp_config.get('every_n_epochs',1),
        every_n_train_steps  =  exp_config.get('every_n_train_steps',0),
        save_on_train_epoch_end = exp_config.get('save_on_train_epoch_end',False),
    )
    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=exp_result_dir,
        filename =  time_exp_name+'_{epoch:04d}',
        verbose = exp_config.get('verbose',True),
        save_last=exp_config.get('save_last',False),
        every_n_epochs = 1,
        save_on_train_epoch_end = False,
    )
    model = dataset.get_experiment_model(config)   
    progress_bar = MyTQDMProgressBar(refresh_rate=1)
    trainer = pl.Trainer(strategy='ddp' if exp_config.get('devices',1) > 1 else None,
                         track_grad_norm=-1, max_epochs=exp_config.get('epochs',10),
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=exp_config.get('accumulate_grad_batches',1),
                         val_check_interval = exp_config.get('val_check_interval',1.0),
                         limit_train_batches = exp_config.get("limit_train_batches", 1.0),
                         limit_val_batches = exp_config.get("limit_val_batches", 1.0),
                         limit_test_batches = exp_config.get("limit_test_batches", 1.0),    
                         limit_predict_batches = exp_config.get("limit_predict_batches", 0.),
                         num_sanity_val_steps = 2,
                         # check_val_every_n_epoch=2,
                         logger=TBlogger,
                         callbacks=[progress_bar,checkpoint_callback,checkpoint_callback_epoch],
                         enable_progress_bar=True,
                         precision=exp_config.get('precision',32),
                         accelerator= exp_config.get('accelerator','gpu'),
                         devices=exp_config.get('devices',1),
                         log_every_n_steps = 50,
                         resume_from_checkpoint = exp_config.get('checkpoint',None),
                         )
    if test_checkpoint == 'best': # when using multiple callbacks, it will use the best one in the first callbacks
        trainer.fit(model,model.data)
    if exp_config.get("limit_test_batches", 1.0) > 0:
        trainer.test(model,model.data,ckpt_path = test_checkpoint )
    if exp_config.get("limit_predict_batches", 0.) > 0:
        trainer.predict(model,model.data,ckpt_path = test_checkpoint)
    if not retrain_ckp:
        state_dict_best = checkpoint_callback.state_dict()
        state_dict_epoch = checkpoint_callback_epoch.state_dict()
        for state_dict in (state_dict_best,state_dict_epoch):
            for k_name in ['best_model_path','last_model_path']:
                file_path = state_dict[k_name]
                os.remove(file_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default='ERROR')
    parser.add_argument('--monitor', default='val_loss')
    parser.add_argument('--monitor_id', type=int,default=0)
    parser.add_argument('--minmax', default='min')
    parser.add_argument('--metric', default='test_acc')
    parser.add_argument('--metric_id', type=int,default=1)
    args = parser.parse_args()
    config_file = args.config
    monitor_name = args.monitor
    monitor_log_id = args.monitor_id
    minmax = args.minmax
    if minmax not in ('min', 'max'):
        print('Error min max value!')
        sys.exit(0)
    metric_name = args.metric
    log_id = args.metric_id
    config = get_params(config_file)
    grid_search_flag = False
    if 'PARAMSGRID' in config:
        grid_search_flag = True
        print("Extract best hyper parameters from a grid search experiment: {}, seed: {}!!".format(config['name'],config['seed']))
        print('Searched parameters: ',config['PARAMSGRID'])
        gs_params = config['PARAMSGRID']
    else:
        print("Extract best hyper parameters from normal experiment: {}, seed: {}!!".format(config['name'],config['seed']))
    exp_config = config['EXPERIMENT']
    exp_result_dir = exp_config.get('save_path',"results/experiments/")
    log_path = os.path.join(exp_result_dir,'lightning_logs')
    all_logs = os.listdir(log_path)
    all_exp_results = {}
    for log_version in all_logs:
        if not log_version.startswith('version'):
            continue
        sub_log_dir = os.path.join(log_path,log_version)
        all_exp_results[log_version] = {}
        if os.path.isdir(sub_log_dir):
            print('===================================== \n Process {}'.format(sub_log_dir))
            hp_file_path = os.path.join(sub_log_dir,'hparams.yaml') 
            with open(hp_file_path, 'r') as f:
                hp = yaml.safe_load(f)['config']
                # print('gs_params ',gs_params,hp.keys())
                if grid_search_flag and 'PARAMSGRID' in hp and gs_params == hp['PARAMSGRID']:
                    all_exp_results[log_version]['hp'] = hp
                else:
                    print('This log version doesn\'t belong to this experiment.')
                    all_exp_results.pop(log_version)
                    continue
            file_list = os.listdir(sub_log_dir)
            event_flies = []
            for file_name in file_list:
                print('file_name ',file_name,log_id)
                if file_name.endswith('.yaml'):
                    continue
                event_flies.append(file_name)
            s_event_flies = sorted(event_flies,key=lambda x: int(x.split('.')[-1]))
            if len(s_event_flies) <= log_id:
                print('ERROR! Incompleted logs! skip!!')
                all_exp_results.pop(log_version)
                continue

            # extract best epoch:
            file_name = s_event_flies[monitor_log_id]
            print('Extract Best eopch from file: ',file_name)
            log_file_path = os.path.join(sub_log_dir,file_name)
            #load log data
            ea=event_accumulator.EventAccumulator(log_file_path) 
            ea.Reload()
            print('monitor metric_name ',monitor_name,ea.scalars.Keys())
            if monitor_name not in ea.scalars.Keys():
                print('Arg monitor metric name {} error! The log does\'t has the metric'.format(metric_name))
                sys.exit(0)
            metric_values = [ i.value for i in ea.scalars.Items(monitor_name)]
            if minmax == 'min':
                best_epoch = np.argmin(np.array(metric_values))
            elif minmax == 'max':
                best_epoch = np.argmax(np.array(metric_values))
            all_exp_results[log_version]['epoch'] = best_epoch
            file_name = s_event_flies[log_id]
            print('use file: ',file_name)
            log_file_path = os.path.join(sub_log_dir,file_name)
            #load log data
            ea=event_accumulator.EventAccumulator(log_file_path) 
            ea.Reload()
            print('metric_name ',metric_name,ea.scalars.Keys())
            if metric_name not in ea.scalars.Keys():
                print('Arg metric name {} error! The log does\'t has the metric'.format(metric_name))
                sys.exit(0)
            all_exp_results[log_version]['value'] = ea.scalars.Items(metric_name)[-1].value
            all_exp_results[log_version]['all_metric'] = {}
            for key in ea.scalars.Keys():
                all_exp_results[log_version]['all_metric'][key] = ea.scalars.Items(key)[-1].value
            print('current version: {}, best value: {}, epoch: {}'.format(log_version, all_exp_results[log_version]['value'],\
                                                            all_exp_results[log_version]['epoch']))
    max_value_version = max(all_exp_results.keys(), key = lambda k: all_exp_results[k]['value'])
    data = all_exp_results[max_value_version]
    print('Max value version: {}, value: {}, best epoch: {}'.format(max_value_version, data['value'],data['epoch']))
    print('All of the metric in the best version: ',data['all_metric'])
    if grid_search_flag == True:
        test_params = {}
        hp = data['hp']
        for combination_name, value_list in gs_params.items():
            all_names = combination_name.split('_-')
            # print('all_names ',all_names)
            param_name = all_names[-1]
            sub_config = hp
            for p_name in all_names[:-1]:
                if p_name in sub_config:
                    sub_config = sub_config[p_name]
                else:
                    print('ERROR config of ',combination_name)
                    sys.exit(0)
            value = sub_config[param_name]
            if value not in value_list:
                print('ERROR value of {}, value is {}, but valus list is {}'.format(combination_name,value,str(value_list)))
                sys.exit(0)
            test_params[combination_name] = sub_config[param_name]
        print('Best test params are: ',test_params)
    print('All of the hyper parameters are: ',hp)