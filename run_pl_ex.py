# -*- coding: utf-8 -*-
import os
import dataset
import models
import torch
import customloss
import customoptimisers
import argparse
from customevals import *
from tools.params import get_params
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger,TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from tools.progressbar import MyTQDMProgressBar
from sklearn.model_selection import ParameterGrid
import shutil
from types import MethodType
import json
from tensorboard.backend.event_processing import event_accumulator
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_avg_std(log_dirs,dataname):
    all_results = {}
    log_id = 1 # Two log files, the first 0 is for train and valid, the second is for test 
    for log_dir_path in log_dirs:
        print('===================================== \n Process {}'.format(log_dir_path))
        file_list = os.listdir(log_dir_path)
        event_flies = []
        for file_name in file_list:
            print('file_name ',file_name,log_id)
            if file_name.endswith('.yaml'):
                continue
            event_flies.append(file_name)
        s_event_flies = sorted(event_flies,key=lambda x: int(x.split('.')[-1]))
        if len(s_event_flies) <= log_id:
            print('ERROR! Incompleted logs! skip!!')
            continue

        # extract best epoch: 
        if dataname in ['bs', 'bs-pair', 'eurlex', 'eurlex-inverse', 'gum','quac']:
            metric_name = 'test_micro_f1'
        else:
            metric_name = 'test_acc'  
        file_name = s_event_flies[log_id]
        print('use file: ',file_name)
        log_file_path = os.path.join(log_dir_path,file_name)
        #load log data
        ea=event_accumulator.EventAccumulator(log_file_path) 
        ea.Reload()
        print('metric_name ',metric_name,ea.scalars.Keys())
        if metric_name not in ea.scalars.Keys():
            print('Arg metric name {} error! The log does\'t has the metric'.format(metric_name))
            sys.exit(0)
        # all_exp_results[log_version]['value'] = ea.scalars.Items(metric_name)[-1].value
        # all_exp_results[log_version]['all_metric'] = {}
        for key in ea.scalars.Keys():
            if key not in all_results:
                all_results[key] = [ea.scalars.Items(key)[-1].value]
            else:
                all_results[key].append(ea.scalars.Items(key)[-1].value)

    for k,v in all_results.items():
        print('key: {}, mean: {}, std: {}'.format(k,np.mean(v),np.std(v)))
        print('all values: ',v)


def current_exp_count(exp_result_dir):
    '''
    Experiment count starts from 0
    '''
    log_path = os.path.join(exp_result_dir,'lightning_logs')
    all_exps = os.listdir(log_path)
    count = len(all_exps)
    return count

def run_one_exp(config,args,test_params = None, times = 1):
    print('Start experiment: {}, seed: {}, times: {}'.format(config['name'],config['seed'],times))
    if test_params is not None:
        print('Test params: ',test_params)
    exp_config = config['EXPERIMENT']
    dataname = config['DATA']['dataset_name']
    test_checkpoint = exp_config.get('test_checkpoint','best')
    retain_ckp = config.get('retain_ckp', True)
    ori_exp_name = config.get('name',"default_name")
    stop_strategy = exp_config.get('stop_strategy','early_stop')
    # self.do_train = exp_config.get('do_train', True)
    # self.do_val = exp_config.get('do_val', True)
    # self.do_test = exp_config.get('do_test', False)
    # self.do_predict = exp_config.get('do_predict', True)
    exp_result_dir = exp_config.get('save_path',"results/experiments/")
    log_dirs = []
    for i in range(times):
        seed_everything(config['seed']*(i+1),True)
        TBlogger = TensorBoardLogger(save_dir=exp_result_dir)
        # Clogger = CSVLogger(save_dir=exp_result_dir)
        if os.path.exists(exp_result_dir):
            current_time = current_exp_count(exp_result_dir)
        else:
            current_time = 0
        if args.resume is not None: # resume from last experiment
            current_time -= 1

        time_exp_name = '{}_{}'.format(ori_exp_name,current_time)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(
            dirpath=exp_result_dir,
            filename = time_exp_name+'_best_{epoch:04d}',
            save_top_k=exp_config.get('save_top_k',1),
            verbose = exp_config.get('verbose',True),
            monitor = exp_config.get('monitor','val_loss'),
            save_last=exp_config.get('save_last',False),
            mode = exp_config.get('mode','min'),
            save_weights_only  = exp_config.get('save_weights_only',False),
            every_n_epochs = exp_config.get('every_n_epochs',1),
            every_n_train_steps  =  exp_config.get('every_n_train_steps',0),
            save_on_train_epoch_end = exp_config.get('save_on_train_epoch_end',False),
        )
        checkpoint_callback_epoch = ModelCheckpoint(
            dirpath=exp_result_dir,
            filename =  time_exp_name+'_every_epoch',
            save_weights_only = False,
            verbose = exp_config.get('verbose',True),
            save_last=exp_config.get('save_last',False),
            every_n_epochs = 1,
            save_on_train_epoch_end = True,
        )
        model = dataset.get_experiment_model(config)   
        progress_bar = MyTQDMProgressBar(refresh_rate=1)
        callbacks = [lr_monitor, progress_bar,checkpoint_callback]
        if retain_ckp == True:
            callbacks.append(checkpoint_callback_epoch)
        if stop_strategy == 'early_stop':
            early_stop_callback = EarlyStopping(monitor=exp_config.get('monitor','val_loss'), min_delta=1e-6, \
                patience=exp_config.get('stop_patience',10), verbose=False, mode=exp_config.get('mode','min'), \
                    check_on_train_epoch_end=False)
            callbacks.append(early_stop_callback)
        print('Start Experiment, time: {}, configs: '.format(i))
        print(json.dumps(config, indent=4))
        print("torch.cuda.device_count() ",torch.cuda.device_count())
        trainer = pl.Trainer(strategy='ddp' if exp_config.get('devices',1) > 1 else 'auto',
                            max_epochs=exp_config.get('epochs',10),
                            accumulate_grad_batches=exp_config.get('accumulate_grad_batches',1),
                            val_check_interval = exp_config.get('val_check_interval',1.0),
                            limit_train_batches = exp_config.get("limit_train_batches", 1.0),
                            limit_val_batches = exp_config.get("limit_val_batches", 1.0),
                            limit_test_batches = exp_config.get("limit_test_batches", 1.0),    
                            limit_predict_batches = exp_config.get("limit_predict_batches", 0.),
                            num_sanity_val_steps = 2,
                            # check_val_every_n_epoch=2,
                            logger=TBlogger,
                            callbacks=callbacks,
                            enable_progress_bar=True,
                            precision=exp_config.get('precision',32),
                            accelerator= exp_config.get('accelerator','gpu'),
                            devices=exp_config.get('devices',1),
                            log_every_n_steps = 50,
                            # resume_from_checkpoint = exp_config.get('checkpoint',None),
                            )
        if args.test_only == True:
            trainer.test(model,model.data,ckpt_path = args.resume)
        else:
            if test_checkpoint == 'best': # when using multiple callbacks, it will use the best one in the first callbacks
                trainer.fit(model,model.data,ckpt_path = args.resume)
                if getattr(model,'embedding_params')['initialization'] == 'original':
                    pass
                elif hasattr(model.model.embeddinglayer.embedding,'learnable_weight') and \
                    model.model.embeddinglayer.embedding.learnable_weight in \
                        ('init_specific_kp_weight','init_kp_weight_randomly','init_kp_weight_randomly2'):
                    print("learned kp weights ",model.model.embeddinglayer.embedding.kp_bias_emb.weight)
            if exp_config.get("limit_test_batches", 1.0) > 0:
                trainer.test(model,model.data,ckpt_path = test_checkpoint )
            if exp_config.get("limit_predict_batches", 0.) > 0:
                trainer.predict(model,model.data,ckpt_path = test_checkpoint)
            print('retain_ckp   ',retain_ckp)
            if not retain_ckp:
                state_dict_best = checkpoint_callback.state_dict()
                # print("...........")
                # print(state_dict_best)
                # state_dict_epoch = checkpoint_callback_epoch.state_dict()
                # print(state_dict_epoch)
                for state_dict in [state_dict_best]:
                    file_path = state_dict['best_model_path']
                    file_path = os.path.join(exp_result_dir,file_path.split('/')[-1])
                    # print(file_path)
                    os.remove(file_path)
            log_dirs.append(os.path.join(TBlogger.save_dir, TBlogger.name, "version_{}".format(TBlogger.version)))
        calculate_avg_std(log_dirs,dataname)

def prepare_envs():
    if not os.path.exists('results/stat/'):
        os.mkdir('results/stat/')
    if not os.path.exists('results/cache/'):
        os.mkdir('results/cache/')
    if not os.path.exists('results/analysis/'):
        os.mkdir('results/analysis/')
    if not os.path.exists('results/cache/key_phrase_split/'):
        os.mkdir('results/cache/key_phrase_split/')
    if not os.path.exists('results/cache/tokenized_results/'):
        os.mkdir('results/cache/tokenized_results/')
    if not os.path.exists('results/cache/vocabs/'):
        os.mkdir('results/cache/vocabs/')

if __name__=="__main__":
    prepare_envs()
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default='ERROR')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--start_version', type=int, default=0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--run_times',type=int, default=1)
    args = parser.parse_args()
    config_file = args.config
    run_times = args.run_times
    config = get_params(config_file)
    exp_result_dir = config['EXPERIMENT'].get('save_path',"results/experiments/")
    if os.path.exists(exp_result_dir) and args.clean == True:
        shutil.rmtree(exp_result_dir)
    if 'PARAMSGRID' in config:
        print("This is a hyper paremeters grid search experiment: {}, seed: {}!!".format(config['name'],config['seed']))
        params_grid = list(ParameterGrid(config['PARAMSGRID']))
        start_version = args.start_version # the version increases from 0
        print(start_version, len(params_grid))
        print("tune seed!")
        ori_seed = config['seed']
        for add_s in range(1):
            config['seed'] = ori_seed + add_s
            for i in range(start_version, len(params_grid)):
                params = params_grid[i]
                for combination_name, value in params.items():
                    all_names = combination_name.split('_-')
                    param_name = all_names[-1]
                    sub_config = config
                    for p_name in all_names[:-1]:
                        if p_name in sub_config:
                            sub_config = sub_config[p_name]
                        else:
                            print('ERROR config of ',combination_name)
                            sys.exit(0)
                    sub_config[param_name] = value
                print("---------------------")
                print('Total param groups: {}, current: {}'.format(len(params_grid), i+1))
                # when searchning the parameters, run_times should be 1
                # with torch.autograd.set_detect_anomaly(True):
                with torch.autograd.set_detect_anomaly(True):
                    run_one_exp(config,args,params,times=1)
    elif 'PARAMSLIST' in config:
        for combination_name, value_list in config['PARAMSLIST'].items():
            all_names = combination_name.split('_-')
            param_name = all_names[-1]
            sub_config = config
            for p_name in all_names[:-1]:
                if p_name in sub_config:
                    sub_config = sub_config[p_name]
                else:
                    print('ERROR config of ',combination_name)
                    sys.exit(0)
            for i in range(len(value_list)):
                sub_config[param_name] = value_list[i]
                print("---------------------")
                print('current param groups: {}, current: {}'.format(len(value_list), i+1))
                # when searchning the parameters, run_times should be 1
                run_one_exp(config,args,value_list[i],times=1)
    else:
        run_one_exp(config,args,times=run_times)