from tools.textprocesser import Preprocesser
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
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
from tools.quac_eva import _eval_fn

class QUACDataset(Dataset):

    def __init__(self, file_path, max_seq_len, ratio = 1, tokenizer = None, split='train',\
                 attn_mode={'name':'default','param1':0}):
        # print(file_path)
        self.token_wise_task_qa = True
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.split = split
        self.did, self.data, self.questions, self.labels = self.loadFile(split)
        self.pair_qas()
        self.attn_mode = attn_mode
        print('Init dataset: {}, split {}, num of samples: {}, real num of samples {}'.\
              format(self.__class__,self.split, len(self.data), self.__len__()))
    
    def _loadFile(self, split):

        try:
            dataset = datasets.load_from_disk('./data/quac/')
        except:
            dataset = datasets.load_dataset('allenai/quac')
            dataset.save_to_disk('./data/quac/')
        text = []
        questions = []
        labels = []
        did = []
        split = 'validation' if split == 'val' else 'train'
        for item in dataset[split]:
            did.append(item['dialogue_id'])
            text.append(item['context'])
            questions.append(item['questions'])
            labels.append(item['answers'])

        return did, text, questions, labels

    def loadFile(self,split):
        did, text, questions, labels = self._loadFile(split)
        assert len(text) == len(labels),'ERROR, the lenths are different'
        # labels = np.array([ int(i) for i in labels])
        return did, text, questions, labels

    def pair_qas(self):
        p_did = []
        p_data = []
        p_q = []
        p_ans = []
        for dialogue_id, doc, doc_questions, doc_answers in zip(self.did, self.data, self.questions, self.labels):
            for i, question in enumerate(doc_questions):
                p_did.append(dialogue_id)
                p_data.append(doc)
                p_q.append(question)
                one_q_ans_list = [] # for training split, each question only has one answer, for val split, multiple answers
                for ans, ans_start in zip(doc_answers['texts'][i], doc_answers['answer_starts'][i]):
                    # for ans, ans_start in zip(ans_list, ans_start_list):
                    one_q_ans_list.append((ans, ans_start))
                p_ans.append(one_q_ans_list)
        self.p_did = p_did
        self.p_data = p_data
        self.p_q = p_q
        self.p_ans = p_ans

    def __len__(self):
        return int(len(self.p_data) * self.ratio)

    def __getitem__(self, idx):
        # print("???????? get ",idx)
        return (self.p_data[idx],self.p_q[idx], self.p_ans[idx], self.p_did[idx], idx)

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

class QUAC(pl.LightningDataModule):
    def __init__(self,data_path, config, pre_cache=True):
        super(QUAC,self).__init__()
        self.token_wise_task_qa = True
        self.data_path =data_path
        self.global_config = config
        self.pre_cache = pre_cache
        #### common procedure
        self.init_attr_from_config()
        self.init_datasets()

    def init_attr_from_config(self):
        # customed method
        data_config = self.global_config['DATA']
        self.batch_size = data_config.get('batch_size',32)
        self.train_ratio = data_config.get('train_ratio',1.)
        self.val_split_ratio = data_config.get('val_split_ratio',0.1)
        self.label_encoder = LabelEncoder()
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
        for split in ['train','val']:# 
            ratio = self.train_ratio if split == 'train' else 1
            self.datasets[split] = QUACDataset(file_path=self.data_path,tokenizer=None,
                                max_seq_len=self.max_seq_len,ratio = ratio,split = split,attn_mode=self.attn_mode)

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
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_test_collect_fn(self.datasets['val'], examples))

    def predict_dataloader(self):
        pass
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def init_collect_fn(self):
        if self.tokenizer_type == 'bert':
            self.train_val_test_collect_fn = QUACDataset.collate_fn_bert
        elif self.tokenizer_type == 'non_bert':
            self.train_val_test_collect_fn = QUACDataset.collate_fn_non_bert
        else:
            print("ERROR! {} is not supported".format(self.tokenizer_type))

class ExperimentQUAC(pl.LightningModule):
    # to complete
    '''
    Each dataset is also an experiment environment, with specific metrics, loss, and a head at the top of the model.
    In this way, it's more convenient to compare different models in the same setting. And also in this style, each model 
    only takes charge of the feature extraction.
    '''
    def __init__(self, config):
        super(ExperimentQUAC, self).__init__()
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
        self.eval_results = {}
        self.cache_S = 0
        self.max_answer_length = 1024 # doesn't limit 

    def init_analysis_data(self):
        self.test_results = {'test_data':[],"eval_results":None}

    def cache_analysis_data(self,batch):
        cache_data = {}
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                cache_data[k] = v.detach().cpu()
            else:
                cache_data[k] = v
        self.test_results['test_data'].append(cache_data)
    
    def save_analysis_data(self):
        self.test_results["eval_results"] = self.eval_results
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
            save_file_name = save_file_name.replace('gum','hp')
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
        self.s = nn.Softmax(dim=1)

    def init_metrics(self):
        pass
        # self.accuracy = {}
        # self.f1score = {}
        # for split in ['train']:
        #     # use setattr to make the matric module being moved to the same device of the parent model.
        #     # is only use the dict to cache the matric module, it need to be moved to the right device manually
        #     acc_metric_attr = '__'+split+'_acc'
        #     f1_metric_attr = '__'+split+'_f1'
        #     self.__setattr__(acc_metric_attr,torchmetrics.Accuracy(task='multiclass', num_classes=self.data.nclasses))
        #     self.__setattr__(f1_metric_attr,torchmetrics.F1Score(task='multiclass', average = 'macro', num_classes=self.data.nclasses))
        #     self.accuracy[split] = self.__getattr__(acc_metric_attr)
        #     self.f1score[split]  = self.__getattr__(f1_metric_attr)

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
            # print('model_output ',head_output.shape,head_output[0],head_output[1],inputs[0],model_output[0][:10],model_output[1][:10])
        return loss, head_output, attn

    def compute_metrics_step(self,split, preds, batch):
        pass
        # targets = batch['targets']
        # # print(preds.device,targets.device)
        # # print(preds)
        # # print(targets)
        # acc = self.accuracy[split](preds.detach(), targets)
        # f1score = self.f1score[split](preds.detach(), targets)
        # return acc, f1score
    
    def compute_metrics_epoch(self,split):
        if split in ['val','test']:
            matric_json = _eval_fn(self.eval_results)
        else:
            matric_json = {}
        return matric_json

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
        # preds = self.s(logits)
        # acc, f1_score, = self.compute_metrics_step('train', preds, batch)
        #self.log_dict({'loss': loss, 'macro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = {'loss': loss.detach(), 'bs': len(logits)}
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
        loss,logits,attn = self.forward(batch, batch_idx,loss=False)
        # preds = self.s(logits)
        self.update_model_output(logits, batch, self.data.tokenizer, self.eval_results)
        # acc, f1_score, microf1_score = self.compute_metrics_step('val',preds, batch)
        #self.log_dict({'loss': loss, 'macro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        log_ret = { 'bs': len(logits)}
        ret = { 'bs': len(logits)}
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
        # acc,macro_f1,microf1_score = self.compute_metrics_epoch('train')
        logs = {'train_loss': loss}
        # print(logs)
        self.log_dict(logs,prog_bar=True)
        return None # on_train_epoch_end can only return None

    def get_output_mask(self, outer):
        S = outer.size(1)
        # S = 20
        if S <= self.cache_S:
            return torch.tensor(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), self.max_answer_length) # 15 is used as the  max span length
        # print(outer.dtype,np_mask.dtype,outer.device)
        self.cache_mask = torch.tensor(np_mask,dtype = outer.dtype).to(device = outer.device)
        return torch.tensor(self.cache_mask, requires_grad=False)

    def update_model_output(self,logits,batch, tokenizer, eval_results):
        logits = logits.detach().cpu()
        o_context_start_positions = batch['o_context_start_positions'].tolist()
        o_attention_mask = batch['o_attention_mask'].tolist()
        o_offset_mapping = batch['o_offset_mapping']
        input_ids = batch['o_input_ids']
        dialogue_ids = batch['o_dialogue_ids']
        ans_list_list = batch['targets']
        contexts = batch['o_sentences']
        questions = batch['o_questions']
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # don't consider question tokens
        # print("start_logits shape ",start_logits.shape)
        for i, pos in enumerate(o_context_start_positions):
            # 将索引位置及其之前的位置置-10000
            pad_num = o_attention_mask[i].count(0) + 1# +one [SEP] token
            # print("o_attention_mask.sum() ",pad_num)
            start_logits[i, :pos] = -10000
            end_logits[i, :pos] = -10000
            start_logits[i, -pad_num:] = -10000
            end_logits[i, -pad_num:] = -10000


        outer = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)
        # we process the logits above, hence we don't need to process the mask
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask)
        # print(outer)
        topk_answers_indices = outer.flatten(start_dim=1).topk(k=1).indices
        # print(topk_answers_indices)
        # # print(outer.size())
        pred_start_positions = topk_answers_indices.div(outer.size()[1],rounding_mode='floor') #start_index: bs*max_answer_num
        pred_end_positions = topk_answers_indices%outer.size()[1]
        # print(pred_start_positions)
        # print(pred_end_positions)

        # pred_start_positions = torch.argmax(start_logits,dim=-1)
        # pred_end_positions = torch.argmax(end_logits,dim=-1)
        for pred_start, pred_end, token_ids, did, q_str, ans_list, context, offset in zip(pred_start_positions, pred_end_positions, input_ids, dialogue_ids, questions, ans_list_list, contexts, o_offset_mapping):
            dia_results = eval_results.get(did, {"context":context, "qas":{}})
            ans_list = [ans_turple[0] for ans_turple in ans_list]
            pred_start = pred_start[0] # only predict one answer
            pred_end = pred_end[0]
           
            if pred_start > pred_end:
                dia_results["qas"][q_str] = (ans_list, None)
            else:
                pred_answer = context[offset[pred_start][0]:offset[pred_end][1]]
                # pred_answer = tokenizer.decode(token_ids[pred_start:pred_end+1])
                # print("pred_answer pre: ",pred_answer, len(pred_answer), context.lower() in tokenizer.decode(token_ids).lower())
                # Because we use uncased bert, so we need to revert its format
                # s = context.lower().find(pred_answer)
                # e = s + len(pred_answer)
                # pred_answer = context[s:e]
                print(pred_start, pred_end,offset[pred_start],token_ids[pred_start], offset[pred_end], token_ids[pred_end], len(offset),len(token_ids))
                print(pred_answer)
                dia_results["qas"][q_str] = (ans_list, pred_answer)
                # print(dia_results["qas"][q_str], dia_results["qas"][q_str][1] in context)
                # print(tokenizer.decode(token_ids)[:30], tokenizer.decode(token_ids)[30:], tokenizer.decode(token_ids) == context, context[:30])
            eval_results[did] = dia_results

    def check_error(self):
        c = 0
        check_preds = []
        for i in range(len(self.eval_results)):
            for pred in (self.eval_results[i]):
                if c in [7, 19, 34]:
                    check_preds.append(pred)
                c += 1
        print("preds for  [7, 19, 34], targets: [1,1,1]: ", check_preds)

    def on_validation_epoch_end(self):
        ret = {}
        logs = {}
        total_loss = 0
        total_samples = 0
        # for batch_outputs in self.val_output_list:
        #     total_loss += batch_outputs['loss'] * batch_outputs['bs']
        #     total_samples += batch_outputs['bs']
            # print(total_loss,batch_outputs['loss'],batch_outputs['bs'],total_samples)
           

        # loss = total_loss/total_samples
        matric_json = self.compute_metrics_epoch('val')
        # print('loss ',loss,acc)
        logs = {}
        logs.update(matric_json)
        self.log_dict(logs,prog_bar=True)
        # self.check_error()
        self.eval_results = {}
        return None

    def test_step(self, batch, batch_idx):
        loss,logits,attn = self.forward(batch, batch_idx,loss=False)
        # preds = self.s(logits)
        # acc, f1_score,microf1_score = self.compute_metrics_step('test',preds, batch)
        #self.log_dict({'loss': loss, 'macro_f1': f1_score, 'acc': acc},batch_size=len(batch))
        self.update_model_output(logits, batch, self.data.tokenizer, self.eval_results)
        log_ret = {'bs': len(logits)}
        ret = {'bs': len(logits)}
        # print(ret)
        self.test_output_list.append(log_ret)
        # for analysis
        self.cache_analysis_data(batch)
        return ret

    def on_test_epoch_end(self):
        ret = {}
        logs = {}
        total_loss = 0
        total_samples = 0
        # for batch_outputs in self.test_output_list:
        #     total_loss += batch_outputs['loss'] * batch_outputs['bs']
        #     total_samples += batch_outputs['bs']
           

        # loss = total_loss/total_samples
        matric_json = self.compute_metrics_epoch('test')
        # print('loss ',loss,acc)
        logs = {}
        logs.update(matric_json)
        self.log_dict(logs,prog_bar=True)
        self.save_analysis_data()
        self.eval_results = {}
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

            logits = output
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            start_positions, end_positions = batch['o_start_positions'], batch['o_end_positions']
    
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            # print("start_logits shape",start_logits.shape)
            # print("start_positions ",start_positions)
            # print("end_positions ",end_positions)
            # print("start_logits ",torch.argmax(start_logits,dim=-1))
            # print("end_logits ",torch.argmax(end_logits,dim=-1))

            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            print("total_loss ", total_loss,start_loss,end_loss)
            # print(output.shape,batch['targets'].shape)
            # print(output[:10],batch['targets'][:10])
        return total_loss

    