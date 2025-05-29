import os
import io
from lxml import etree
from tools.textprocesser import Preprocesser
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tools.tokenizer import get_tokenizer
import pytorch_lightning as pl
import torch
from customlayers.embedding import EmbeddingLayer
import pandas as pd
import torch.utils.data as data
import models
from torch.optim.lr_scheduler import LambdaLR
import dataset
from . import text_helper as th
FILES = {
        'train':'BBC_News_Train.csv',
        'test':'BBC_News_Test.csv',
}

class BBCDataset(Dataset):
    """

    """
    def __init__(self, file_path, max_seq_len, ratio = 1, tokenizer = None, split='train'):
        print(file_path)
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.split = split
        self.ids, self.data, self.labels = self.loadFile(self.file_path)
    
    def loadFile(self, fpath):
        data_df = pd.read_csv(fpath)
        ids = data_df['ArticleId'].tolist()
        data = data_df['Text'].tolist()
        if self.split == 'train':
            labels = data_df['Category'].tolist()
        else:
            labels = [-1]*len(data)
        return ids, data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx],self.labels[idx],self.ids[idx],idx)
    
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

class BBC(pl.LightningDataModule):
    """
    """
    def __init__(self,data_path, config, pre_cache=True):
        super(BBC,self).__init__()
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
        self.preprocesser_cfg = data_config.get('processer',{'remove_punctuation':True,'stem':False,'lower':True,'stopword':False})
        self.tokenizer_type = data_config.get('tokenizer_type',"non_bert")
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_params = data_config.get('tokenizer_params',{})
        self.num_workers = data_config.get('num_workers',1)
        
        self.max_seq_len = data_config.get('max_seq_len',512) 
        self.set_datasets = False
        self.set_tokenizer = False
        self.datasets = {'train':None,'test':None}
        self.vocab_name = data_config.get('vocab','results/vocabs/mr.pt')

    def init_datasets(self):
        file_names = FILES
        for split in ['train', 'test']:
            ratio = self.train_ratio if split == 'train' else 1
            filepath = os.path.join(self.data_path, file_names[split])
            self.datasets[split] = BBCDataset(file_path=filepath, tokenizer=None,
                                max_seq_len=self.max_seq_len,ratio = ratio,split = split)

        seed = torch.Generator().manual_seed(0)
        val_len = int(len(self.datasets['train']) * self.val_split_ratio)
        self.train_set, self.valid_set = data.random_split(self.datasets['train'], \
                [len(self.datasets['train'])-val_len, val_len], generator=seed)
        self.init_tokenizer()
        # preprocess
        self.preprocess()
        # To avoid multithreading conflication???, reset the tokenizer
        self.init_collect_fn()
        self.set_datasets = True

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
    
    def init_collect_fn(self):
        if self.tokenizer_type == 'bert':
            self.train_val_test_collect_fn = BBCDataset.collate_fn_bert
        elif self.tokenizer_type == 'non_bert':
            self.train_val_test_collect_fn = BBCDataset.collate_fn_non_bert
        else:
            print("ERROR! {} is not supported".format(self.tokenizer_type))

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
        pass

    def predict_dataloader(self):
        pass
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

class ExperimentBBC(pl.LightningModule):
    # to complete
    '''
    Each dataset is also an experiment environment, with specific metrics, loss, and a head at the top of the model.
    In this way, it's more convenient to compare different models in the same setting. And also in this style, each model 
    only takes charge of the feature extraction.
    '''
    def __init__(self, config):
        super(ExperimentBBC, self).__init__()
        self.global_config = config
        self.init_attr_from_config()
        self.init_model()

    
    def init_attr_from_config(self):
        # model_config = self.global_config['MODEL']
        # self.num_labels = model_config['num_labels']
        # self.ret_features = model_config['ret_features']
        # self.max_sequence_length = model_config['max_sequence_length']
        # self.measurement_size = model_config['measurement_size']
        # self.dropout = model_config['dropout']
        # self.use_lexicon_as_measurement = model_config['use_lexicon_as_measurement']
        # self.embedding_params = model_config['embedding']
        # ---------------------------------------------------------------- #
        data_config = self.global_config['DATA']
        self.dataset_name = data_config.get('dataset_name','bbc')
        # ---------------------------------------------------------------- #
        experiment_config = self.global_config['EXPERIMENT']
        self.do_warmup = experiment_config.get('do_warmup', False)
        self.steps = experiment_config.get('steps', 30000)
        self.warmup = experiment_config.get('warmup',1000)
        self.lr = experiment_config.get('lr', 1e-3)
        self.optimizer = experiment_config.get('optimizer', 'adam')
        self.optimizer_params = experiment_config.get('optimizer_params', {})
        self.loss = experiment_config.get('loss', "ce")
        self.use_discocat = experiment_config.get('use_discocat', False)
        data = dataset.get_data(data_config,experiment_config.get('accelerator','gpu'),self.global_config)
        self.data = data
        # ---------------------------------------------------------------- #
        model_config = self.global_config['MODEL']
        self.input_dim = model_config['output_dim']

    def init_model(self):
        params = {}
        params['vocab'] = self.data.token2id
        self.model = models.get_model(params,"",self.global_config)

    def init_head(self):
        self.head = nn.Linear(self.input_dim,self.data.nclasses)

    def forward(self, batch, batch_idx,loss=True):
        inputs = batch['input_ids']
        if self.use_discocat == False:
            model_output, attn = self.model(inputs)
        else:
            pass # todo
        head_output = self.head(model_output)
        if loss == True:
            loss = self.calculate_loss(head_output,batch)
        else:
            loss = 0
        return loss, head_output, attn

    def training_step(self, batch, batch_idx):
        loss,output = self.forward(batch, batch_idx)
        start_logits, end_logits = output
        f1_scores,em_scores,answer_scores,qids = self._generate_answer(batch, start_logits, end_logits)
        self.log_dict({'loss': loss, 'answer_scores': torch.tensor(answer_scores).float().mean(),\
            'f1': torch.tensor(f1_scores).float().mean(), \
            'em': torch.tensor(em_scores).float().mean()})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx,dataloader_idx):
        loss,output = self.forward(batch, batch_idx)
        start_logits, end_logits = output
        f1_scores,em_scores,answer_scores,qids = self._generate_answer(batch, start_logits, end_logits)
        # return {'loss': loss,'qids': qids,'answer_scores': answer_scores,}
        return {'loss': loss,'qids': qids, 'answer_scores': answer_scores,'f1': f1_scores, 'em': em_scores,
        'dataloader_idx':dataloader_idx}

    def on_validation_epoch_end(self, epoch_outputs):
        # epoch_outputs contains two dataloader's outputs, val and vval
        ret = {}
        pre_fix = ['avg_val_','avg_verified_val_']
        logs = {}
        for i in range(len(epoch_outputs)):
            outputs = epoch_outputs[i]
            n_samples = 0
            avg_loss = 0
            for x in outputs:
                n_samples += len(x['qids'])
                avg_loss += x['loss']*len(x['qids'])
            avg_loss = avg_loss/n_samples
            string_qids = [item for sublist in outputs for item in sublist['qids']]
            answer_scores = [item for sublist in outputs for item in sublist['answer_scores']]
            f1_scores = [item for sublist in outputs for item in sublist['f1']]
            em_scores = [item for sublist in outputs for item in sublist['em']]
            print(f'before sync --> sizes: {len(string_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')
            # if self.trainer.use_ddp:
            #     torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            #     avg_loss /= self.trainer.world_size
            #     torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
            #     avg_em /= self.trainer.world_size

            #     string_qids = self.sync_list_across_gpus(string_qids, avg_loss.device, torch.int)
            #     answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            #     f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            #     em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.int)
            print(f'after sync --> sizes: {len(string_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')

            # Because of having multiple documents per questions, some questions might have multiple corresponding answers
            # Here, we only keep the answer with the highest answer_score
            qa_with_duplicates = {}
            for qid, answer_score, f1_score, em_score in zip(string_qids, answer_scores, f1_scores, em_scores):
                if qid not in qa_with_duplicates:
                    qa_with_duplicates[qid] = []
                qa_with_duplicates[qid].append({'answer_score': answer_score, 'f1': f1_score, 'em': em_score})
            f1_scores = []
            em_scores = []
            for qid, answer_metrics in qa_with_duplicates.items():
                top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
                f1_scores.append(top_answer['f1'])
                em_scores.append(top_answer['em'])
            avg_val_f1 = sum(f1_scores) / len(f1_scores)
            avg_val_em = sum(em_scores) / len(em_scores)

            logs[pre_fix[i]+'loss'] = avg_loss
            logs[pre_fix[i]+'f1'] = avg_val_f1
            logs[pre_fix[i]+'em']: avg_val_em
            ret[pre_fix[i]+'loss']=avg_loss
            ret['log'] = logs
            ret['progress_bar'] = logs
        return ret

    def predict_step(self, batch, batch_idx):
        _,output = self.forward(batch, batch_idx,False)
        start_logits, end_logits = output
        answers,qids = self._generate_answer(batch, start_logits, end_logits)
        return {'answers': answers,'qids': qids}

    def predict_epoch_end(self, outputs):
        qids = [item for sublist in outputs for item in sublist['qids']]
        answers = [item for sublist in outputs for item in sublist['answers']]

        qa_with_duplicates = {}
        for qid, answer in zip(qids, answers):
            qa_with_duplicates[qid].append({'answer_score': answer['score'], 'answer_text': answer['text'], })

        qid_to_answer_text = {}
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            qid_to_answer_text[qid] = top_answer['answer_text']

        with open('predictions.json', 'w') as f:
            json.dump(qid_to_answer_text, f)

        return {'count': len(qid_to_answer_text)}

    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.warmup:
                return float(current_step) / float(max(1, self.warmup))
            return max(0.0, float(self.steps - current_step) / float(max(1, self.steps - self.warmup)))
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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

    def calculate_loss(self,output,batch):
        if self.loss == 'ce': 
            loss = nn.functional.cross_entropy(output, batch['targets'], weight=None, size_average=None,\
                 ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
        return loss

    def sync_list_across_gpus(self, list_to_sync, device, dtype):
        l_tensor = torch.tensor(list_to_sync, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()
    
    # def on_validation_epoch_end(self, epoch_outputs):
    #     pass
