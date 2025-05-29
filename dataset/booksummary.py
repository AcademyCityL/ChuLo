from lxml import etree
from tools.textprocesser import Preprocesser
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from tools.tokenizer import get_tokenizer
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
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
import csv
import random
from sklearn.metrics import accuracy_score, f1_score

class BSDataset(Dataset):
    # multi-label dataset
    file_name = 'booksummaries.txt'
    def __init__(self, file_path, max_seq_len, ratio = 1, tokenizer = None, split='train',\
                 attn_mode={'name':'default','param1':0},pair=False):
        super(BSDataset,self).__init__()
        # print(file_path)
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.split = split
        self.attn_mode = attn_mode
        self.pair = pair
        self.data, self.labels = self.loadFile(split)
        print('Init dataset: {}, split {}, num of samples: {}'.format(self.__class__,self.split, len(self.data)))

    def loadFile(self,split):
        data = []
        with open('data/bs/booksummaries.txt', 'r') as f:
            reader = csv.reader(f, dialect='excel-tab')
            for row in reader:
                data.append(row)
        # convert data to pandas dataframe
        books = pd.DataFrame.from_records(data, columns=['book_id', 'freebase_id', 'book_title', 'author', 'publication_date', 'genre', 'summary'])
        # print(books.iloc[0])
        all_sentences = []
        all_genres = []
        def parse_genre_entry(genre_info):
            if genre_info == '':
                return []
            genre_dict = json.loads(genre_info)
            genres = list(genre_dict.values())
            return genres

        books['genre'] = books['genre'].apply(parse_genre_entry)

        for i in range(len(books)):
            if len(books.iloc[i]['genre']) > 0 and len(books.iloc[i]['summary'].split(' ')) > 10:
                all_sentences.append(books.iloc[i]['summary'])
                all_genres.append(books.iloc[i]['genre'])
        # follow the train-test split: https://aclanthology.org/2022.acl-short.79.pdf, but only the same number of 
        # train/test docs because they didn't supply the split file
        # Efficient Classification of Long Documents Using Transformers
        text = []
        labels = []
        if os.path.exists('data/bs/train_test_split.pt'):
            split_file = torch.load('data/bs/train_test_split.pt')
        else:
            test_sample_ids = random.sample(range(len(all_sentences)),1279)
            train_sample_ids = []
            for i in range(len(all_sentences)):
                if i not in test_sample_ids:
                    train_sample_ids.append(i)
            print(len(train_sample_ids),len(test_sample_ids))
            split_file = {'train':train_sample_ids,'test':test_sample_ids}
            torch.save(split_file,"data/bs/train_test_split.pt")
        for i in range(len(all_sentences)):
            if i in split_file[split]:
                text.append(all_sentences[i])
                labels.append(all_genres[i])
        print("self.pair ???? ",self.pair,len(text),len(labels))
        if self.pair == True:
            new_text = []
            new_labels = []
            for i in range(0, len(text) - 1, 2):
                new_text.append(text[i] + text[i+1])
                new_labels.append(list(set(labels[i] + labels[i+1])))
            text = new_text
            labels = new_labels
        print("len(text) ",len(text),len(labels))    
        return text, labels

    def __len__(self):
        return int(len(self.data) * self.ratio)

    def __getitem__(self, idx):
        # print("???????? get ",idx)
        return (self.data[idx],self.labels[idx],idx)

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
        

class BS(pl.LightningDataModule):
    """
    """
    def __init__(self,data_path, config, pre_cache=True,pair=False):
        super(BS,self).__init__()
        self.data_path =data_path
        self.global_config = config
        self.pre_cache = pre_cache
        self.pair = pair
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
        self.datasets = {'train':None,'test':None}

        model_config = self.global_config['MODEL']
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})


    def init_datasets(self):
        for split in ['train', 'test']:
            ratio = self.train_ratio if split == 'train' else 1
            self.datasets[split] = BSDataset(file_path=self.data_path,tokenizer=None,
                                max_seq_len=self.max_seq_len,ratio = ratio,split = split,attn_mode=self.attn_mode,pair=self.pair)
        seed = torch.Generator().manual_seed(0)
        if self.val_split_ratio > 0:
            val_len = int(len(self.datasets['train']) * self.val_split_ratio)
            self.train_set, self.valid_set = data.random_split(self.datasets['train'], \
                    [len(self.datasets['train'])-val_len, val_len], generator=seed)
        else:
            self.train_set = self.datasets['train']
            self.valid_set = self.datasets['test']
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(self.datasets['train'], examples))

    def val_dataloader(self):
        dataset = self.datasets['train'] if self.val_split_ratio > 0 else self.datasets['test']
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(dataset, examples))

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
            self.train_val_test_collect_fn = BSDataset.collate_fn_bert
        elif self.tokenizer_type == 'non_bert':
            self.train_val_test_collect_fn = BSDataset.collate_fn_non_bert
        else:
            print("ERROR! {} is not supported".format(self.tokenizer_type))

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " x ".join(map(str, size))


def dump_tensors(gpu_only=True,cache={}):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    cache = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    key = "%s:%s%s %s" % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_size(obj.size()),
                        )
                    # if len(pretty_size(obj.size())) == 0:
                    #     print("obj.size() ",obj.size(),obj.numel())
                    cache[key] = cache.get(key,0)+1
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                continue
                if not gpu_only or obj.is_cuda:
                    key = "%s → %s:%s%s%s%s %s" % (
                            type(obj).__name__,
                            type(obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(obj.data.size()),
                        )
                    cache[key] = cache.get(key,0)+1
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    gc.collect()
    print("Total size:", total_size)
    print("cache ",cache)
    # return cache

class ExperimentBS(pl.LightningModule):
    # to complete
    '''
    Each dataset is also an experiment environment, with specific metrics, loss, and a head at the top of the model.
    In this way, it's more convenient to compare different models in the same setting. And also in this style, each model 
    only takes charge of the feature extraction.
    '''
    def __init__(self, config):
        super(ExperimentBS, self).__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.save_hyperparameters('config')
        self.global_config = config
        self.init_attr_from_config()
        self.init_model()
        self.init_head()
        self.init_metrics()
        self.init_analysis_data()
        # dump_tensors()
    
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
        self.loss = experiment_config.get('loss', "bce")
        self.lossparams = experiment_config.get('lossparams', {})
        self.epochs = experiment_config.get('epochs', 1)
        data = dataset.get_data(data_config,experiment_config.get('accelerator','gpu'),self.global_config)
        self.data = data
        # ---------------------------------------------------------------- #
        model_config = self.global_config['MODEL']
        self.head_input_dim = model_config.get('output_dim',512)
        self.total_steps = math.ceil(len(self.data.train_set) / self.data.batch_size) * self.epochs
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})
        self.embedding_params = model_config.get('embedding',{})
        self.p_value = model_config.get('pvalue',0.5)

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
        save_file_name = self.dataset_name + '_' + str(self.global_config['seed']) + '_' + self.lrscheduler + '_' + 'ep_{}_wp_{}_'.format(self.epochs,self.warmup)
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
            # save_file_name = save_file_name.replace('hyperpartisan','hp')
            save_file_name = save_file_name.replace('param1','p1')
            save_file_name = save_file_name.replace('param2','p2')
            save_file_name = save_file_name.replace('param2','p3')
            save_file_name = save_file_name.replace('linearwarmup','lwup')
            save_file_name = save_file_name.replace('cosinewarmup','cwup')
            save_file_name = save_file_name.replace('initilization','init')
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
        self.f1score = {}
        for split in ['train','val','test','predict']:
            # use setattr to make the matric module being moved to the same device of the parent model.
            # is only use the dict to cache the matric module, it need to be moved to the right device manually
            acc_metric_attr = '__'+split+'_acc'
            f1_metric_attr = '__'+split+'_f1'
            # print("self.data.nclasses ",self.data.nclasses, type(self.data.nclasses))
            self.__setattr__(acc_metric_attr,torchmetrics.ExactMatch(task='multilabel', num_labels=self.data.nclasses))
            # use the same evaluation metric with https://aclanthology.org/2022.acl-short.79.pdf, note that he use multilabel micro f1
            # in the validation end
            self.__setattr__(f1_metric_attr,torchmetrics.F1Score(task='multilabel', average = 'micro', num_labels=self.data.nclasses))
            self.accuracy[split] = self.__getattr__(acc_metric_attr)
            self.f1score[split]  = self.__getattr__(f1_metric_attr)

    def forward(self, batch, batch_idx,loss=True):
        # dump_tensors()
        inputs = batch['input_ids']
        model_output, attn = self.model(inputs,max_chunk_len=batch.get('max_chunk_len',None),\
                    attention_mask=batch['attention_mask'],special_tokens_mask = batch['special_tokens_mask'],\
                        kp_token_weights=batch.get('kp_token_weights',None),map_ids=batch.get('map_ids',None),sent_map_ids = batch.get('sent_map_ids',None),sentence_textrank_scores=batch.get('sentence_textrank_scores',None))
        if self.global_config['MODEL']['name'] != 'BERT':
            if self.add_cls == True:
                model_output = model_output[:,0,:]
            head_output = self.head(model_output) 
        else:
            head_output = model_output
        # print(model_output[:2])
        # print(head_output[:2])
        # print("head_output ",head_output.shape)
        # print(torch.cuda.memory_summary())
        if loss == True:
            loss = self.compute_loss(head_output,batch)
        else:
            loss = 0
            print('model_output ',head_output.shape,head_output[0],head_output[1],inputs[0],model_output[0][:10],model_output[1][:10])
        # dump_tensors()
        # del loss

        # print("batch ",batch['targets'].shape)
        # if hasattr(self,'lcache') == False:
        #     self.lcache = {}
        # dump_tensors(True,self.lcache)
        # set1 = set(dict1.items())
        # set2 = set(dict2.items())
        # diff = cache.items() ^ self.last_cache.items()
        return loss, head_output, attn

    def compute_metrics_step(self,split, preds, batch):
        targets = batch['targets']
        # print(preds.device,targets.device)
        # print(preds)
        # print(targets)
        # print("preds ",preds.max())
        # print("preds[0] ",preds[0])
        # print("targets[0] ",targets[0])
        preds = preds.detach()
        preds = torch.where(preds>self.p_value,1,0)
        acc = self.accuracy[split](preds, targets)
        f1score = self.f1score[split](preds, targets)
        # print("f1 score ",f1_score(targets,preds), f1score)
        return acc, f1score
    
    def compute_metrics_epoch(self,split):
        acc = self.accuracy[split].compute()
        self.accuracy[split].reset()
        f1score = self.f1score[split].compute()
        self.f1score[split].reset()
        return acc, f1score

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

    def training_step(self, batch, batch_idx):
        loss,logits,attn = self.forward(batch, batch_idx)
        if self.loss == 'softmax_opt':
            preds = logits
            index1 = preds<self.p_value
            index2 = preds>self.p_value
            preds[index1] = 0
            preds[index2] = 1
        else:
            preds = self.s(logits)
        acc, f1_score = self.compute_metrics_step('train', preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': len(logits)}
        ret = {'loss': loss, 'bs': len(logits)}
        # print(ret)
        self.train_output_list.append(log_ret)
        return ret

    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        loss,logits,attn = self.forward(batch, batch_idx)
        if self.loss == 'softmax_opt':
            preds = logits
            index1 = preds<self.p_value
            index2 = preds>self.p_value
            preds[index1] = 0
            preds[index2] = 1
        else:
            preds = self.s(logits)
        acc, f1_score = self.compute_metrics_step('val',preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': len(logits)}
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
        acc,micro_f1 = self.compute_metrics_epoch('train')
        logs = {'train_loss': loss, 'train_micro_f1': micro_f1, 'train_em': acc}
        # print(logs)
        self.log_dict(logs,prog_bar=True)
        return None # on_train_epoch_end can only return None

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
        acc,micro_f1 = self.compute_metrics_epoch('val')
        # print('loss ',loss,acc)
        logs = {'val_loss': loss, 'val_micro_f1': micro_f1, 'val_em': acc}
        self.log_dict(logs,prog_bar=True)
        return None

    def test_step(self, batch, batch_idx):
        loss,logits,attn = self.forward(batch, batch_idx)
        if self.loss == 'softmax_opt':
            preds = logits
            index1 = preds<self.p_value
            index2 = preds>self.p_value
            preds[index1] = 0
            preds[index2] = 1
        else:
            preds = self.s(logits)
        acc, f1_score = self.compute_metrics_step('test',preds, batch)
        #self.log_dict({'loss': loss, 'micro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': len(logits)}
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
        acc,micro_f1 = self.compute_metrics_epoch('test',)
        # print('loss ',loss,acc)
        logs = {'test_loss': loss, 'test_micro_f1': micro_f1, 'test_em': acc}
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
            # print("self.parameters() ")
            # for k,_ in self.named_parameters():
            #     print(k)
            # for name, parameters in self.named_parameters().items():
            #     print("name ",name)
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
        def multilabel_categorical_crossentropy(y_true, y_pred):
            """多标签分类的交叉熵
            说明：y_true和y_pred的shape一致，y_true的元素非0即1，
                1表示对应的类为目标类，0表示对应的类为非目标类。
            警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
                不用加激活函数，尤其是不能加sigmoid或者softmax！预测
                阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
                本文。
            """
            y_pred = (1 - 2 * y_true) * y_pred
            y_pred_neg = y_pred - y_true * 1e12
            y_pred_pos = y_pred - (1 - y_true) * 1e12
            zeros = torch.zeros_like(y_pred[..., :1])
            y_pred_neg = torch.concatenate([y_pred_neg, zeros], axis=-1)
            y_pred_pos = torch.concatenate([y_pred_pos, zeros], axis=-1)
            neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
            pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
            return neg_loss + pos_loss
        if self.loss == 'ce': 
            # print("ce: ",output.shape,batch['targets'].shape,output[0],batch['targets'][0])
            loss = nn.functional.cross_entropy(output, batch['targets'], weight=None, size_average=None,\
                 ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=self.lm)
        elif self.loss == 'bce':
            # print('output ',output.dtype, batch['targets'].dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(output, batch['targets'].float(), weight=None, size_average=None,reduce=None, reduction='mean')
            # print(loss)
            # print(output.shape,batch['targets'].shape)
            # print(output[:10],batch['targets'][:10])
        elif self.loss == 'fbce':
            gamma = 5
            output = output.reshape(-1)
            targets = batch['targets'].float().reshape(-1)
            p = torch.sigmoid(output)
            pt = torch.where(targets >= 0.5, p, 1-p)
            logp = torch.log(pt)
            loss = -1*((1-pt)**gamma)*logp
            loss = loss.mean()
        elif self.loss == 'hinge':
            hingelosslabel = [[]]*len(batch['targets'])
            for sample_id, label_id in torch.argwhere(batch['targets']==1).tolist():
                hingelosslabel[sample_id].append(label_id)
            # pad
            for i in range(len(hingelosslabel)):
                hingelosslabel[i] = hingelosslabel[i] + [-1]*(len(batch['targets'][0]) - len(hingelosslabel[i]))
            hingelosslabel = torch.tensor(hingelosslabel).cuda()
            p = torch.sigmoid(output)
            loss = nn.functional.multilabel_margin_loss(p, hingelosslabel,reduction='mean')
        elif self.loss == 'softmax_opt':
            loss = multilabel_categorical_crossentropy(batch['targets'].float(),output)
            loss = loss.mean()
            # print("lsos ",loss.shape)
        return loss