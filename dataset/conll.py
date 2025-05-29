from tools.textprocesser import Preprocesser
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from tools.tokenizer import get_tokenizer
import pytorch_lightning as pl
import torchmetrics
import torch
from customlayers.embedding import EmbeddingLayer
import pandas as pd
import torch.utils.data as data
import models
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
import io
import dataset
import datasets 
from . import text_helper as th
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
import math
import dadaptation
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ReduceLROnPlateau
from tools.lrschedulers import SequentialLRwithRLROP
import json
from tools.grokfast import gradfilter_ma, gradfilter_ema
from conllu import parse_incr
from datasets import load_dataset

class ConllDataset(Dataset):

    def __init__(self, file_path, max_seq_len, ratio = 1, tokenizer = None, split='train',\
                 attn_mode={'name':'default','param1':0}):
        # print(file_path)
        self.token_wise_task = True
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.split = split
        self.data, self.labels, self.ori_token_list = self.loadFile(split)
        self.attn_mode = attn_mode
        print('Init dataset: {}, split {}, num of samples: {}, real num of samples {}'.\
              format(self.__class__,self.split, len(self.data), self.__len__()))
    
    def _loadFile(self, split):

        split_cfg = {"train":("train", 120),"val":("validation", 20),"test":("test", 20)}
        da = load_dataset("ontonotes/conll2012_ontonotesv5","english_v12")
        cache_data = []
        real_spilt, top_num = split_cfg[split]
        for doc in da[real_spilt]:
            doc_len = 0
            tokens = []
            labels = []
            for sentence_obj in doc['sentences']:
                doc_len += len(sentence_obj['named_entities'])
                tokens.extend(sentence_obj['words'])
                named_entities = [[i] for i in sentence_obj['named_entities']]# to reuse the code written for gum (nested ner, each token has multiple labels). Will transform to normal format when calculating the metric and the loss
                labels.extend(named_entities)
            cache_data.append((doc_len, tokens, labels))
           
        sorted_data = sorted(cache_data,key = lambda x: x[0], reverse=True)
        f_data = sorted_data[:top_num]
    
        text = []
        labels = []
        ori_token_list = []
        for data in f_data:
            text.append(' '.join(data[1]))
            ori_token_list.append(data[1])
            labels.append(data[2])

        return text, labels, ori_token_list

    def loadFile(self,split):
        text,labels,ori_token_list = self._loadFile(split)
        assert len(text) == len(labels),'ERROR, the lenths are different'
        # labels = np.array([ int(i) for i in labels])
        return text, labels, ori_token_list

    def __len__(self):
        return int(len(self.data) * self.ratio)

    def __getitem__(self, idx):
        # print("???????? get ",idx)
        return (self.data[idx],self.labels[idx],self.ori_token_list[idx], idx)

    def set_tokenizer(self,tokenizer):
        self.tokenizer = tokenizer

    def set_vocab(self, token2id,id2token):
        self.token2id = token2id
        self.id2token = id2token

    def set_cache_tokenize(self, cache_tokenize):
        # print('set_cache_tokenize ???')
        self.cache_tokenize = cache_tokenize

    def set_lable_encoder(self,label_encoder):
        self.label_encoder = label_encoder

    def cut_and_pad(self,sentences):
        return th.cut_and_pad(self,sentences)
    
    def collate_fn_non_bert(self,examples):
        return th.collate_fn_non_bert(self,examples)

    def collate_fn_bert(self, examples):
        return th.collate_fn_bert(self,examples)

class Conll(pl.LightningDataModule):
    def __init__(self,data_path, config, pre_cache=True):
        super(Conll,self).__init__()
        self.data_path =data_path
        self.global_config = config
        self.pre_cache = pre_cache
        self.token_wise_task = True
        #### common procedure
        self.init_attr_from_config()
        self.init_datasets()

    def init_attr_from_config(self):
        # customed method
        data_config = self.global_config['DATA']
        self.batch_size = data_config.get('batch_size',32)
        self.train_ratio = data_config.get('train_ratio',1.)
        self.val_split_ratio = data_config.get('val_split_ratio',0.1)
        self.label_encoder = MultiLabelBinarizer()
        self.tokenizer_type = data_config.get('tokenizer_type',"non_bert")
        self.tokenizer_name = data_config.get('tokenizer_name','nltk_tweet')
        self.tokenizer_params = data_config.get('tokenizer_params',{})
        self.num_workers = data_config.get('num_workers',1)
        
        self.max_seq_len = data_config.get('max_seq_len',512) 
        self.chunking = data_config.get('chunking',False)
        self.preprocesser_cfg = data_config.get('processer',dict(remove_punctuation=False,stem=False,lower=False,stopword=False)) 
        self.set_datasets = False
        self.set_tokenizer = False
        self.datasets = {'train':None}

        model_config = self.global_config['MODEL']
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})

    def init_datasets(self):
        for split in ['train','val','test']:# 
            self.datasets[split] = ConllDataset(file_path=self.data_path,tokenizer=None,
                                max_seq_len=self.max_seq_len,ratio = 1.,split = split,attn_mode=self.attn_mode)

        self.init_tokenizer()
        # preprocess
        self.preprocess()
        # To avoid multithreading conflication???, reset the tokenizer
        self.init_collect_fn()
        self.set_datasets = True
        print('Init datasets done')

    def setup(self, stage):
        self.stage = stage
        if self.set_tokenizer == False:
            for split,dataset in self.datasets.items():
                if self.tokenizer_type == 'bert':
                    tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
                elif self.tokenizer_type == 'non_bert':
                    tokenizer = self.tokenizer
                dataset.set_tokenizer(tokenizer)
            self.set_tokenizer = True

    def preprocess(self):
        return th.preprocess(self)


    def construct_vocab(self,all_corpus):
        return th.construct_vocab(self,all_corpus)

    def add_tokens(self,tokens):
        return th.add_tokens(self,tokens)

    def init_tokenizer(self):
        return th.init_tokenizer(self)


    def prepare_data(self):
        '''
        Downloading and saving data with multiple processes (distributed settings) will 
        result in corrupted data. Lightning ensures the prepare_data() is called only within
         a single process on CPU, so you can safely add your downloading logic within.
         prepare_data is called from the main process. It is not recommended to assign state 
         here (e.g. self.x = y) since it is called on a single process and if you assign states 
         here then they won’t be available for other processes.
        '''
        
        pass

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(self.datasets['train'], examples))

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(self.datasets['val'], examples))

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(self.datasets['test'], examples))

    def predict_dataloader(self):
        pass
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def init_collect_fn(self):
        if self.tokenizer_type == 'bert':
            self.train_val_test_collect_fn = ConllDataset.collate_fn_bert
        elif self.tokenizer_type == 'non_bert':
            self.train_val_test_collect_fn = ConllDataset.collate_fn_non_bert
        else:
            print("ERROR! {} is not supported".format(self.tokenizer_type))

class ExperimentConll(pl.LightningModule):
    # to complete
    '''
    Each dataset is also an experiment environment, with specific metrics, loss, and a head at the top of the model.
    In this way, it's more convenient to compare different models in the same setting. And also in this style, each model 
    only takes charge of the feature extraction.
    '''
    def __init__(self, config):
        super(ExperimentConll, self).__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.grads = None
        # self.automatic_optimization = False

        self.save_hyperparameters('config')
        self.global_config = config
        self.init_attr_from_config()
        self.init_model()
        self.init_head()
        self.init_metrics()
        self.init_analysis_data()

    
    def init_attr_from_config(self):
        # ---------------------------------------------------------------- #
        data_config = self.global_config['DATA']
        self.dataset_name = data_config.get('dataset_name','bbc')
        self.tokenizer_type = data_config.get('tokenizer_type','bert')
        self.tokenizer_name = data_config.get('tokenizer_name','bert-base-uncased')
        self.tokenizer_params = data_config.get('tokenizer_params',{})
        self.use_tr_tokenizer = data_config.get('use_tr_tokenizer',False)
        self.use_chunk_emb = data_config.get('use_chunk_emb',False)
        self.add_cls = self.tokenizer_params.get('add_cls',False) if self.tokenizer_type == 'non_bert' else True
        # ---------------------------------------------------------------- #
        experiment_config = self.global_config['EXPERIMENT']
        self.steps = experiment_config.get('steps', 0)
        self.warmup = experiment_config.get('warmup',0)
        self.lr = experiment_config.get('lr', 1e-3)
        self.lm = experiment_config.get('lm', 0.)
        self.optimizer = experiment_config.get('optimizer', 'adam')
        self.optimizer_params = experiment_config.get('optimizer_params', {})
        self.lrscheduler = experiment_config.get('lrscheduler', 'warmupReduceLROnPlateau')
        self.lrscheduler_params = experiment_config.get('lrscheduler_params', {})
        self.loss = experiment_config.get('loss', "ce")
        self.epochs = experiment_config.get('epochs', 1)
        data = dataset.get_data(data_config,experiment_config.get('accelerator','gpu'),self.global_config)
        self.data = data
        # ---------------------------------------------------------------- #
        model_config = self.global_config['MODEL']
        self.head_input_dim = model_config.get('output_dim',512)
        self.total_steps = math.ceil(len(self.data.datasets['train']) / self.data.batch_size) * self.epochs
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})
        self.embedding_params = model_config.get('embedding',{})

        self.random_seed = self.global_config['seed']
        self.custom_save_name = self.global_config.get('cn','no')
        self.eval_results = []

    def init_analysis_data(self):
        self.test_results = {'preds':[],'test_data':[],'attns':[]}

    def cache_analysis_data(self,preds,batch,attn):
        self.test_results['preds'].append(preds.detach().cpu())
        cache_data = {}
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                cache_data[k] = v.detach().cpu()
            else:
                cache_data[k] = v
        self.test_results['test_data'].append(cache_data)
        if attn is not None:
            self.test_results['attns'].append(attn.sum(dim=1).cpu())
    
    def save_analysis_data(self):
        save_file_name = self.dataset_name + f'_{self.custom_save_name}_{self.random_seed}_' + self.lrscheduler + '_'
        for name_config in [self.attn_mode,self.embedding_params]:
            for k,v in name_config.items():
                if type(v) == dict:
                    for kk,vv in v.items():
                        if kk == 'model_name':
                            save_file_name += str(kk)+str(vv).replace('/','_')+'_'
                        else:
                            save_file_name += str(kk)+str(vv)+'_'
                elif type(v) == list:
                    for vv in v:
                        save_file_name += str(vv)+'_'
                else:
                    save_file_name += str(k)+str(v)+'_'
        save_file_name += self.tokenizer_type + '_' + self.tokenizer_name.replace('/','_') + '_.pt'
        print("save analysis data to ",save_file_name)
        if len(save_file_name) > 256:
            save_file_name = save_file_name.replace('conll','cl')
            save_file_name = save_file_name.replace('param1','p1')
            save_file_name = save_file_name.replace('param2','p2')
            save_file_name = save_file_name.replace('param2','p3')
            save_file_name = save_file_name.replace('linearwarmup','lwup')
            save_file_name = save_file_name.replace('cosinewarmup','cwup')
            save_file_name = save_file_name.replace('initilization','init')
            save_file_name = save_file_name.replace('longformer','lf')
        torch.save(self.test_results,'results/analysis/{}'.format(save_file_name))

    def init_model(self):
        params = {}
        params['vocab'] = self.data.token2id
        params['daobj'] = self.data
        self.model = models.get_model(params,"",self.global_config)

    def init_head(self):
        if self.global_config['MODEL']['name'] != 'BERT':
            self.head = nn.Linear(self.head_input_dim,self.data.nclasses)
        self.s = nn.Sigmoid()

    def init_metrics(self):
        self.accuracy = {}
        self.microf1score = {}
        for split in ['train','val','test','predict']:
            # use setattr to make the matric module being moved to the same device of the parent model.
            # is only use the dict to cache the matric module, it need to be moved to the right device manually
            microf1_metric_attr = '__'+split+'_microf1'
            self.__setattr__(microf1_metric_attr,torchmetrics.F1Score(task='multiclass', average = 'micro', num_classes=self.data.nclasses))
            self.microf1score[split]  = self.__getattr__(microf1_metric_attr)

    def forward(self, batch, batch_idx,loss=True):
        inputs = batch['input_ids']
        model_output, attn = self.model(inputs,max_chunk_len=batch.get('max_chunk_len',None),\
                    attention_mask=batch['attention_mask'],special_tokens_mask = batch['special_tokens_mask'],\
                        kp_token_weights=batch.get('kp_token_weights',None),map_ids=batch.get('map_ids',None),sent_map_ids = batch.get('sent_map_ids',None),sentence_textrank_scores=batch.get('sentence_textrank_scores',None),o_input_ids=batch.get('o_input_ids',None),o_attention_mask=batch.get('o_attention_mask',None),o_token_labels=batch.get('o_token_labels',None))
        if self.global_config['MODEL']['name'] != 'BERT':
            if self.add_cls == True:
                model_output = model_output[:,0,:]
            head_output = self.head(model_output) 
        else:
            head_output = model_output
        # print(model_output[:2])
        # print(head_output[:2])
        if loss == True:
            loss = self.compute_loss(head_output,batch)
        else:
            loss = 0
            print('model_output ',head_output.shape,head_output[0],head_output[1],inputs[0],model_output[0][:10],model_output[1][:10])
        return loss, head_output, attn

    def compute_metrics_step(self,split, preds, batch):
        targets = batch['o_token_labels']
        preds,targets, extra_targets = self.collect_logits_targets(preds,targets)
        # print("targets.max()",set(targets.argmax(dim=-1).tolist()))
    
        targets = targets.argmax(dim=-1)
        preds = preds.detach()
        preds = torch.argmax(preds,dim=-1)
        microf1score = self.microf1score[split](preds, targets)
        bs = len(targets)
        if extra_targets is not None:
            extra_targets = extra_targets.argmax(dim=-1)
            preds = torch.ones(extra_targets.shape,device = extra_targets.device) * -1
            microf1score = self.microf1score[split](preds, extra_targets)
            bs += len(extra_targets)

        return microf1score,bs
    
    def compute_metrics_epoch(self,split):
        microf1score = self.microf1score[split].compute()
        self.microf1score[split].reset()
        return microf1score

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_output_list = []
        return
    
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []
        return
    
    def on_test_epoch_start(self):
        super().on_validation_epoch_start()
        self.test_output_list = []
        return

    def backward(self, loss, *args, **kwargs) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your
        own implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()
        """
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)
            self.grads = gradfilter_ema(self, grads=self.grads)

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()
        # opt.zero_grad()

        loss,logits,attn = self.forward(batch, batch_idx)
        preds = self.s(logits)
        microf1_score,bs = self.compute_metrics_step('train', preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': bs}
        ret = {'loss': loss, 'bs': len(logits)}
        # print(ret)
        self.train_output_list.append(log_ret)


        # automatically applies scaling, etc...
        # self.manual_backward(loss)

        ### Option 1: Grokfast (has argument alpha, lamb)
        # self.grads = gradfilter_ema(self, grads=self.grads)
        ### Option 2: Grokfast-MA (has argument window_size, lamb)
        # grads = gradfilter_ma(model, grads=grads, window_size=window_size, lamb=lamb)

        # opt.step()

        return ret

    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        loss,logits,attn = self.forward(batch, batch_idx)
        preds = self.s(logits)
        self.eval_results.append(preds.detach().cpu())
        microf1_score,bs = self.compute_metrics_step('val', preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': bs}
        ret = {'loss': loss, 'bs': len(logits)}
        # print(ret)
        self.val_output_list.append(log_ret)
        return ret

    def on_train_epoch_end(self):
        logs = {}
        total_loss = 0
        total_samples = 0
        for batch_outputs in self.train_output_list:
            total_loss += batch_outputs['loss'] * batch_outputs['bs']
            total_samples += batch_outputs['bs']
           

        loss = total_loss/total_samples
        microf1_score = self.compute_metrics_epoch('train')
        logs = {'train_loss': loss, 'train_micro_f1':microf1_score}
        # print(logs)
        self.log_dict(logs,prog_bar=True)
        return None # on_train_epoch_end can only return None

    def check_error(self):
        pass

    def on_validation_epoch_end(self):
        ret = {}
        logs = {}
        total_loss = 0
        total_samples = 0
        for batch_outputs in self.val_output_list:
            total_loss += batch_outputs['loss'] * batch_outputs['bs']
            total_samples += batch_outputs['bs']
            # print(total_loss,batch_outputs['loss'],batch_outputs['bs'],total_samples)
           

        loss = total_loss/total_samples
        microf1_score = self.compute_metrics_epoch('val')
        # print('loss ',loss,acc)
        logs = {'val_loss': loss, 'val_micro_f1':microf1_score}
        self.log_dict(logs,prog_bar=True)
        # self.check_error()
        self.eval_results = []
        return None

    def test_step(self, batch, batch_idx):
        loss,logits,attn = self.forward(batch, batch_idx)
        preds = self.s(logits)
        microf1_score,bs = self.compute_metrics_step('test', preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': bs}
        ret = {'loss': loss, 'bs': len(logits)}
        # print(ret)
        self.test_output_list.append(log_ret)
        # for analysis
        self.cache_analysis_data(preds,batch,attn)
        return ret

    def on_test_epoch_end(self):
        ret = {}
        logs = {}
        total_loss = 0
        total_samples = 0
        for batch_outputs in self.test_output_list:
            total_loss += batch_outputs['loss'] * batch_outputs['bs']
            total_samples += batch_outputs['bs']
           

        loss = total_loss/total_samples
        microf1_score = self.compute_metrics_epoch('test')
        # print('loss ',loss,acc)
        logs = {'test_loss': loss,'test_micro_f1':microf1_score}
        self.log_dict(logs,prog_bar=True)
        self.save_analysis_data()
        return None

    def configure_optimizers(self):

        def lr_lambda(current_step):
            # new lr = lr * lr_lambda(current_step)
            
            if self.warmup > 0 and current_step < self.warmup:
                return float(current_step) / float(max(1, self.warmup))
            elif self.steps > 0:
                return max(0.0, float(self.steps - current_step) / float(max(1, self.steps - self.warmup)))
            else:
                return 1
        # for name, param in self.named_parameters():
        #     print(name, param.size())
        if self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'd_adaptation':
            # By setting decouple=True, it use AdamW style weight decay
            # lr is needed, see https://github.com/facebookresearch/dadaptation
            optimizer = dadaptation.DAdaptAdam(self.parameters(), lr=self.lr, \
                                               decouple=True,log_every=10) 
        
        if self.lrscheduler == 'warmupReduceLROnPlateau':
            lrscheduler1 = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=self.warmup,last_epoch=-1)
            lrscheduler2 = ReduceLROnPlateau(optimizer, patience=self.global_config['EXPERIMENT'].get('stop_patience',10)//2,\
                                             verbose=True)
            slrscheduler = SequentialLRwithRLROP(optimizer,[lrscheduler1,lrscheduler2],milestones=[self.warmup],last_epoch=-1)
            return [optimizer], [{"scheduler": slrscheduler, "interval": "step"}, \
                                           {"scheduler": slrscheduler, "interval": "epoch", "monitor":\
                                  self.global_config['EXPERIMENT']['monitor'],'reduce_on_plateau':True}]
        elif self.lrscheduler == 'cosinewarmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup, self.total_steps, last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.lrscheduler == 'warmup':
            scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=self.warmup*self.total_steps,last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.lrscheduler == 'linearwarmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=self.warmup*self.total_steps,num_training_steps=self.total_steps,last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def compute_loss(self,output,batch):
        if self.loss == 'ce': 
            outputs, targets, extra_targets = self.collect_logits_targets(output,batch['o_token_labels'])
            # print(targets)
            # assert len(targets[0]) == 1, "error token label length!"
            targets = targets.argmax(dim=-1)
            # print(targets.max())
            loss = nn.functional.cross_entropy(outputs, targets, reduction ='mean')
        elif self.loss == 'bce':
            pass
            # print('output ',output.dtype, batch['targets'].dtype)
            # outputs, targets = self.collect_logits_targets(output,batch['o_token_labels'])
            # loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, weight=None, size_average=None,reduce=None, reduction='mean')
            # print(loss)
            # print(output.shape,batch['targets'].shape)
            # print(output[:10],batch['targets'][:10])
        return loss
    
    def collect_logits_targets(self, outputs, targets):
        # print("outputs ",outputs.shape,targets.shape)
        if targets.shape[1] > outputs.shape[1]:
            s_targets_1 = targets[:,:outputs.shape[1],:].reshape(-1,self.data.nclasses).float()
            s_targets_2 = targets[:,outputs.shape[1]:,:].reshape(-1,self.data.nclasses).float()
            # print(s_targets_1.shape, s_targets_2.shape)

            outputs = outputs.view(-1,self.data.nclasses)

            select_index = s_targets_1.sum(dim=-1) > 0
            outputs = outputs[select_index]
            r_targets_1 = s_targets_1[select_index]

            select_index = s_targets_2.sum(dim=-1) > 0
            r_targets_2 = s_targets_2[select_index]
            return outputs, r_targets_1, r_targets_2
        else:
            if targets.shape[1] < outputs.shape[1]:
                outputs = outputs[:,:targets.shape[1],:]
                outputs = outputs.reshape(-1,self.data.nclasses)
            else:
                outputs = outputs.view(-1,self.data.nclasses)
            targets = targets.view(-1,self.data.nclasses).float()

            # print("\n\n============================================================")
            # print("before_classifier, logits ",before_classifier.shape, logits.shape)
            select_index = targets.sum(dim=-1) > 0
            outputs = outputs[select_index]
            targets = targets[select_index]
        
            return outputs, targets, None