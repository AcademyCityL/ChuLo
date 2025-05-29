import os
from tools.timer import log_time_delta
import pandas as pd
import random
import numpy as np
from preprocess.bucketiterator import BucketIterator
from transformers import AutoTokenizer

from dataset.data_utils import TextDatasetBase
from dataset.textprocesser import Preprocesser
from sklearn.preprocessing import LabelEncoder
from customlayers.embedding import EmbeddingLayer
from torch.utils.data import Dataset, DataLoader
import json

# to do: pytorch style data loader
class WikiHop(TextDatasetBase):
    def __init__(self, data_path,device, batch_size=8,format = 'pointwise',train_ratio=1.0,remain_ori = False, vocab_inc = ["train"], max_sequence_length=512, \
        extra_tokens=['[UNK]'], processer = {},embedding  = {},\
        tokenizer_type='non_bert', tokenizer_name = 'whitespace',tokenizer_params = {}):
        super(WikiHop, self).__init__(extra_tokens = extra_tokens, tokenizer_type=tokenizer_type, tokenizer_name = tokenizer_name,\
            tokenizer_params = tokenizer_params, processer = processer, embedding = embedding)
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.format = format
        self.train_ratio = train_ratio
        self.label_encoder = LabelEncoder()
        ## load original file 
        self.ori_data = {}
        train,val,test = self.loadFile(os.path.join(data_path))
        self.remain_ori = remain_ori
        if self.remain_ori == True:
            self.ori_data = {'train':train,'val':val,'test':test} 
        ## preprocess data
        self.data = {'train': self.process(train,True), 'val':self.process(val),'test':self.process(test)}

        self.nclasses = len(self.label_encoder.classes_)
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'


        # construct vocab
        self.vocab_inc = vocab_inc
        all_corpus = []
        for split_name in self.vocab_inc:
            all_corpus += self.data[split_name]['x']['q']
            all_corpus += self.data[split_name]['x']['a']
            for ss in self.data[split_name]['x']['s']:
                all_corpus += ss
        self.construct_vocab(all_corpus)
        
        self.max_sequence_length = max_sequence_length

    def process(self,data, fit_label = False):
        # print(data['x']['q'])
        # print(data['x']['a'])
        # data['x']['q'] = self.processer.run(data['x']['q'])
        # data['x']['a'] = self.processer.run(data['x']['a'])
        # for i in range(len(data['x']['s'])):
        #     data['x']['s'][i] = data['x']['a']
        # print(data['x']['q'])
        # print(data['x']['a'])
        # return
        if fit_label == True:
            data['y'] = self.label_encoder.fit_transform(data['y'])
        else:
            data['y'] = self.label_encoder.transform(data['y'])
        return(data)

    @log_time_delta
    def remove_unanswered_questions(self,df):
        counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
        questions_have_correct = counter[counter>0].index 
        return df[df["question"].isin(questions_have_correct) ].reset_index()

    def loadFile(self, fpath):
        datas = dict()
        # The dataset only has dev data, the test data can only be used in the leaderboard
        file_name_list = ['train.json','dev.json']
        data_type_list = ['train','val']
        for file_name,data_type in zip(file_name_list,data_type_list): #'val'            
            data_file = os.path.join(fpath,file_name)
            # Opening JSON file
            f = open(data_file)
            # returns JSON object as
            # a dictionary
            raw_data = json.load(f)
            qid = []
            x = {'q':[],'a':[],'s':[]}
            y = []
            for one_row in raw_data:
                query = self.processer.run([one_row['query']])[0]
                candidates = self.processer.run(one_row['candidates'])
                supports = self.processer.run(one_row['supports'])
                answer = self.processer.run([one_row['answer']])[0]
                for candi in candidates:
                    qid.append(one_row['id'])
                    x['q'].append(query)
                    x['a'].append(candi)
                    x['s'].append(supports)
                    label = 1 if answer == candi else 0
                    y.append(label)
            
            datas[data_type] = {'qid':qid,'x':x,'y':y}
    
        used_length = int(self.train_ratio*len(datas['train']['x']['q']))
        datas['train']['qid'] = datas['train']['qid'][:used_length]
        datas['train']['x']['q'] = datas['train']['x']['q'][:used_length]
        datas['train']['x']['a'] = datas['train']['x']['a'][:used_length]
        datas['train']['x']['s'] = datas['train']['x']['s'][:used_length]
        datas['train']['y'] = datas['train']['y'][:used_length]
        return datas['train'],datas['val'],datas['val']     

    def pointwise_train(self,data):
        last_qidx = data['qid'][0]
        pos_qa = []
        neg_qa = []
        x = []
        y = []
        x_q = []
        for qidx, q,a,ss, label in zip(data['qid'],data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            qapair = q + ' [SEP] ' + a + ' [SEP] ' + ' '.join(ss)
            if last_qidx != qidx:
                if len(pos_qa) > 0:
                    neg_qa = neg_qa*int(len(pos_qa))
                    random.shuffle(neg_qa)
                    x.extend(pos_qa+neg_qa[:len(pos_qa)])
                    y.extend([1]*len(pos_qa)+[0]*len(pos_qa))
                    x_q.extend([qidx]*int(len(pos_qa)+len(pos_qa)))
                last_qidx = qidx
                pos_qa = []
                neg_qa = []
            if label == 1:
                pos_qa.append(qapair)
            else:
                neg_qa.append(qapair)

    def down_sampling_pointwise_train(self,data):
        last_qidx = data['qid'][0]
        pos_qa = []
        neg_qa = []
        x = []
        y = []
        x_q = []
        for qidx, q,a,ss, label in zip(data['qid'],data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            qapair = q + ' [SEP] ' + a + ' [SEP] ' + ' '.join(ss)
            if last_qidx != qidx:
                if len(pos_qa) > 0:
                    neg_qa = neg_qa*int(len(pos_qa))
                    random.shuffle(neg_qa)
                    x.extend(pos_qa+neg_qa[:len(pos_qa)])
                    y.extend([1]*len(pos_qa)+[0]*len(pos_qa))
                    x_q.extend([qidx]*int(len(pos_qa)+len(pos_qa)))
                last_qidx = qidx
                pos_qa = []
                neg_qa = []
            if label == 1:
                pos_qa.append(qapair)
            else:
                neg_qa.append(qapair)

        return x_q,x,y

    def down_sampling_pairwise_train(self,data):
        last_qidx = data['qid'][0]
        pos_qa = []
        neg_qa = []
        x_pos = []
        x_neg = []
        x_q = []
        for qidx, q,a,ss, label in zip(data['qid'],data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            qapair = q + ' [SEP] ' + a + ' [SEP] ' + ' '.join(ss)
            if last_qidx != qidx:
                if len(pos_qa) > 0:
                    neg_qa = neg_qa*int(len(pos_qa))
                    random.shuffle(neg_qa)
                    x_pos.extend(pos_qa)
                    x_neg.extend(neg_qa[:len(pos_qa)])
                    x_q.extend([qidx]*int(len(x_pos)))
                last_qidx = qidx
                pos_qa = []
                neg_qa = []
            if label == 1:
                pos_qa.append(qapair)
            else:
                neg_qa.append(qapair)
        y_pos = [1]*len(x_pos)
        y_neg = [0]*len(x_neg)
        return x_q,x_pos,x_neg,y_pos,y_neg

    def down_sampling_triplet_train(self,data):
        last_q = data['qid'][0]
        pos_a = []
        neg_a = []
        x_pos = []
        x_neg = []
        x_q = []
        for q, a, ss, label in zip(data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            a = a + ' [SEP] ' + ' '.join(ss)
            if last_q != q:
                if len(pos_a) > 0:
                    neg_a = neg_a*int(len(pos_a))
                    random.shuffle(neg_a)
                    x_pos.extend(pos_a)
                    x_neg.extend(neg_a[:len(pos_a)])
                    x_q.extend([q]*int(len(pos_a)))
                last_q = q
                pos_a = []
                neg_a = []
            if label == 1:
                pos_a.append(a)
            else:
                neg_a.append(a)
        y_pos = [1]*len(x_pos)
        y_neg = [0]*len(x_neg)
        return x_q,x_pos,x_neg,y_pos,y_neg

    def full_triplet_train(self,data):
        last_q = data['qid'][0]
        pos_a = []
        neg_a = []
        x_pos = []
        x_neg = []
        x_q = []
        for q, a, ss, label in zip(data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            a = a + ' [SEP] ' + ' '.join(ss)
            if last_q != q:
                if len(pos_a) > 0:
                    for a in pos_a:
                        x_pos.extend([a] * len(neg_a))
                    x_neg.extend(neg_a*len(pos_a))
                    x_q.extend([q]*int(len(pos_a))*int(len(neg_a)))
                last_q = q
                pos_a = []
                neg_a = []
            if label == 1:
                pos_a.append(a)
            else:
                neg_a.append(a)
        y_pos = [1]*len(x_pos)
        y_neg = [0]*len(x_neg)
        return x_q,x_pos,x_neg,y_pos,y_neg

    def sep_qadata(self,data):
        x_q = []
        x_a = []
        y = []
        for q, a, ss, label in zip(data['x']['q'],data['x']['a'],data['x']['s'],data['y']):
            a = a + ' [SEP] ' + ' '.join(ss)
            x_q.append(q)
            x_a.append(a)
            y.append(label)
        return x_q,x_a,y

    def get_train(self, shuffle=True):
        if self.format == 'pointwise':
            x_q,x,y = self.down_sampling_pointwise_train(self.data['train'])
            y = y
            # x_q,x = self.get_qapair(self.data['val'])
            # y = self.data['test']['y']
            data = (x_q,x,y)
        elif self.format == 'pairwise':
            x_q,x_pos,x_neg,y_pos,y_neg = self.down_sampling_pairwise_train(self.data['train'])
            data = (x_q,x_pos,x_neg,y_pos,y_neg)
        elif self.format == 'sampletriplet':
            x_q,x_pos,x_neg,y_pos,y_neg = self.down_sampling_triplet_train(self.data['train'])
            data = (x_q,x_pos,x_neg,y_pos,y_neg)
        elif self.format == 'fulltriplet':
            x_q,x_pos,x_neg,y_pos,y_neg = self.full_triplet_train(self.data['train'])
            data = (x_q,x_pos,x_neg,y_pos,y_neg)
        # print(x_q)
        # return
        return BucketIterator(data,batch_size=self.batch_size,shuffle=shuffle,format = self.format,vocab=self.token2id,\
            tokenizer_type = self.tokenizer_type,tokenizer = self.tokenizer,max_length=self.max_sequence_length,device=self.device)

    def get_qapair(self,data):
        x = []
        x_q = []
        for q, a, ss in zip(data['x']['q'],data['x']['a'],data['x']['s']):
            x.append(q + ' [SEP] '+ a + ' [SEP] ' + ' '.join(ss))
            x_q.append(q)
        return x_q,x

    def get_test(self, shuffle=False):
        if self.format == 'sampletriplet' or self.format == 'fulltriplet':
            x_q,x_a,y = self.sep_qadata(self.data['test'])
            data = (x_q,x_a,y)
        else:
            x_q,x = self.get_qapair(self.data['test'])
            y = self.data['test']['y']
            data = (x_q,x,y)
        return BucketIterator(data,batch_size=self.batch_size,shuffle=shuffle,test = True,format = self.format,vocab=self.token2id,\
            tokenizer_type = self.tokenizer_type,tokenizer = self.tokenizer,max_length=self.max_sequence_length,device=self.device)
    
    def get_val(self, shuffle=False):
        if self.format == 'sampletriplet' or self.format == 'fulltriplet':
            x_q,x_a,y = self.sep_qadata(self.data['val'])
            data = (x_q,x_a,y)
        else:
            x_q,x = self.get_qapair(self.data['val'])
            y = self.data['test']['y']
            data = (x_q,x,y)
        return BucketIterator(data,batch_size=self.batch_size,shuffle=shuffle,test = True,format = self.format,vocab=self.token2id,\
            tokenizer_type = self.tokenizer_type,tokenizer = self.tokenizer,max_length=self.max_sequence_length,device=self.device)