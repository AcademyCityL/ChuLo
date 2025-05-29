import os
from tools.timer import log_time_delta
import pandas as pd
import random
import numpy as np
# from preprocess.bucketiterator import BucketIterator
from transformers import AutoTokenizer

# from dataset.data_utils import TextDatasetBase
# from dataset.textprocesser import Preprocesser
from sklearn.preprocessing import LabelEncoder
from customlayers.embedding import EmbeddingLayer
from torch.utils.data import Dataset, DataLoader
import json
from nltk.tokenize import word_tokenize
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch

import bisect
import re

train_files = {
    'record_file': 'files/hotpotqa_train_record.pt',
    'eval_file': 'files/hotpotqa_train_eval.json',
}

val_files = {
    'record_file': 'files/hotpotqa_val_record.pt',
    'eval_file': 'files/hotpotqa_val_eval.json',
}

test_files = {
    'record_file': 'files/hotpotqa_test_record.pt',
    'eval_file': 'files/hotpotqa_test_eval.json',
}

word2idx_file = 'files/hotpotqa_word2idx.json'
idx2word_file = 'files/hotpotqa_idx2worx.json'
embedding_file = 'files/hotpotqa_embedding.pt'

# to do: pytorch style data loader
class HotpotQA(TextDatasetBase):
    def __init__(self, data_path,device, batch_size=8,format = 'pointwise',train_ratio=1.0, vocab_inc = ["train"], max_sequence_length=512, \
        extra_tokens=['[UNK]'], processer = {},embedding  = {}, distract_version = True,ignore_index=-100,sent_limit=1000,\
        tokenizer_type='non_bert', tokenizer_name = 'nltk',tokenizer_params = {}):
        super(HotpotQA, self).__init__(extra_tokens = extra_tokens, tokenizer_type=tokenizer_type, tokenizer_name = tokenizer_name,\
            tokenizer_params = tokenizer_params, processer = processer, embedding = embedding)
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.format = format
        self.train_ratio = train_ratio
        self.label_encoder = LabelEncoder()
        ## load original file 
        self.ori_data = {}
        self.distract_version = distract_version
        self.ignore_index = ignore_index
        self.sent_limit = sent_limit
        if os.path.isfile(train_files['record_file']):
            with open(train_files['eval_file'], "r") as fh:
                train_eval = json.load(fh)
            with open(val_files['eval_file'], "r") as fh:
                val_eval = json.load(fh)
            with open(test_files['eval_file'], "r") as fh:
                test_eval = json.load(fh)
            self.data = {'train': {'x': self.get_buckets(train_files['record_file']), 'y':train_eval}, 
            'val':{'x': self.get_buckets(val_files['record_file']), 'y':val_eval},\
            'test':{'x': self.get_buckets(test_files['record_file']), 'y':test_eval}
            }
            used_length = int(self.train_ratio*len(self.data['train']['x']))
            self.data['train']['x'] = self.data['train']['x'][:used_length]
            with open(word2idx_file, "r") as fh:
                self.token2id = json.load(fh)
            with open(idx2word_file, "r") as fh:
                self.id2token = json.load(fh)
            self.initial_embedding = torch.load(embedding_file)
        else:
            train,val,test = self.loadFile(os.path.join(data_path))
            self.data = {'train': train,'val':val,'test':test}
            # self.nclasses = len(self.label_encoder.classes_)
            # self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
            print("end loading file")
            # construct vocab
            self.vocab_inc = vocab_inc
            self.token2id, self.id2token = {},{}
            for split_name in self.vocab_inc:
                for i in range(len(self.data[split_name]['x'])):
                    self.add_tokens(self.data[split_name]['x'][i]['context_tokens'])
                    self.add_tokens(self.data[split_name]['x'][i]['ques_tokens'])
            self.add_tokens(self.extra_tokens)
            self.construct_vocab()
            print("start saving vocab and embedding")
            self.save(word2idx_file, self.token2id, message="word2idx")
            self.save(idx2word_file, self.id2token, message='idx2word')
            torch.save(self.initial_embedding, embedding_file)
            print("end saving vocab and embedding")
            ## preprocess data
            self.data = {'train': self.process(train,train_files), 'val':self.process(val,val_files),\
                'test':self.process(test,test_files)}
                # pickle.dump(datapoints, open(out_file, 'wb'), protocol=-1)
        self.max_sequence_length = max_sequence_length

    @log_time_delta
    def construct_vocab(self):
        if self.tokenizer_type == "non_bert":
            # init the embedding # todo multi-layer embedding
            # the weights will be overloaded by the model.load()
            self.initial_embedding =  EmbeddingLayer(self.embedding_params['initialization'], vocab=self.token2id,\
                 **self.embedding_params['kwargs'])
        elif self.tokenizer_type == "bert":
            self.token2id, self.id2token = {},{}
            for token, id in self.tokenizer.vocab.items():
                self.token2id[token] = id
                self.id2token[id] = token
            self.initial_embedding = "None----Using bert"

    def get_buckets(self, record_file):
        # datapoints = pickle.load(open(record_file, 'rb'))
        datapoints = torch.load(record_file)
        return [datapoints]

    def save(self, filename, obj, message=None):
        if message is not None:
            print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

    def process(self,data,config):
        examples = data['x']
        eval_examples = data['y']

        self.build_features(examples, config['record_file'], self.token2id)
        self.save(config['eval_file'], eval_examples, message='{} eval'.format(config['eval_file']))
        return data

    def find_nearest(self,a, target, test_func=lambda x: True):
        idx = bisect.bisect_left(a, target)
        if (0 <= idx < len(a)) and a[idx] == target:
            return target, 0
        elif idx == 0:
            return a[0], abs(a[0] - target)
        elif idx == len(a):
            return a[-1], abs(a[-1] - target)
        else:
            d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
            d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
            if d1 > d2:
                return a[idx-1], d2
            else:
                return a[idx], d1

    def fix_span(self, para, offsets, span):
        span = span.strip()
        parastr = "".join(para)
        assert span in parastr, '{}\t{}'.format(span, parastr)
        begins, ends = map(list, zip(*[y for x in offsets for y in x]))

        best_dist = 1e200
        best_indices = None

        if span == parastr:
            return parastr, (0, len(parastr)), 0

        for m in re.finditer(re.escape(span), parastr):
            begin_offset, end_offset = m.span()

            fixed_begin, d1 = self.find_nearest(begins, begin_offset, lambda x: x < end_offset)
            fixed_end, d2 = self.find_nearest(ends, end_offset, lambda x: x > begin_offset)

            if d1 + d2 < best_dist:
                best_dist = d1 + d2
                best_indices = (fixed_begin, fixed_end)
                if best_dist == 0:
                    break

        assert best_indices is not None
        return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist

    def word_tokenize(self,sent):
        doc = self.tokenizer.tokenize(sent)
        return doc


    def convert_idx(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            pre = current
            current = text.find(token, current)
            if current < 0:
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

    def prepro_sent(self, sent):
        return sent
        # return sent.replace("''", '" ').replace("``", '" ')

    def _process_article(self, article):
        paragraphs = article['context']
        # some articles in the fullwiki dev/test sets have zero paragraphs
        if len(paragraphs) == 0:
            paragraphs = [['some random title', 'some random stuff']]

        text_context, context_tokens, context_chars = '', [], []
        offsets = []
        flat_offsets = []
        start_end_facts = [] # (start_token_id, end_token_id, is_sup_fact=True/False)
        sent2title_ids = []

        def _process(sent, is_sup_fact, is_title=False):
            nonlocal text_context, context_tokens, context_chars, offsets, start_end_facts, flat_offsets
            N_chars = len(text_context)

            sent = sent
            sent_tokens = self.word_tokenize(sent)
            if is_title:
                sent = '<t> {} </t>'.format(sent)
                sent_tokens = ['<t>'] + sent_tokens + ['</t>']
            sent_chars = [list(token) for token in sent_tokens]
            sent_spans = self.convert_idx(sent, sent_tokens)

            sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
            N_tokens, my_N_tokens = len(context_tokens), len(sent_tokens)

            text_context += sent
            context_tokens.extend(sent_tokens)
            context_chars.extend(sent_chars)
            start_end_facts.append((N_tokens, N_tokens+my_N_tokens, is_sup_fact))
            offsets.append(sent_spans)
            flat_offsets.extend(sent_spans)

        if 'supporting_facts' in article:
            supporting_facts = article['supporting_facts']
            sp_set = set(list(map(tuple, article['supporting_facts'])))
        else:
            sp_set = set()
            supporting_facts = []

        sp_fact_cnt = 0
        for para in paragraphs:
            cur_title, cur_para = para[0], para[1]
            _process(self.prepro_sent(cur_title), False, is_title=True)
            sent2title_ids.append((cur_title, -1))
            for sent_id, sent in enumerate(cur_para):
                is_sup_fact = (cur_title, sent_id) in sp_set
                if is_sup_fact:
                    sp_fact_cnt += 1
                _process(self.prepro_sent(sent), is_sup_fact)
                sent2title_ids.append((cur_title, sent_id))

        if 'answer' in article:
            answer = article['answer'].strip()
            if answer.lower() == 'yes':
                best_indices = [-1, -1]
            elif answer.lower() == 'no':
                best_indices = [-2, -2]
            else:
                if article['answer'].strip() not in ''.join(text_context):
                    # in the fullwiki setting, the answer might not have been retrieved
                    # use (0, 1) so that we can proceed
                    best_indices = (0, 1)
                else:
                    _, best_indices, _ = self.fix_span(text_context, offsets, article['answer'])
                    answer_span = []
                    for idx, span in enumerate(flat_offsets):
                        if not (best_indices[1] <= span[0] or best_indices[0] >= span[1]):
                            answer_span.append(idx)
                    best_indices = (answer_span[0], answer_span[-1])
        else:
            # some random stuff
            answer = 'random'
            best_indices = (0, 1)

        ques_tokens = self.word_tokenize(self.prepro_sent(article['question']))
        ques_chars = [list(token) for token in ques_tokens]
        if best_indices[0] > len(context_tokens)-1 or best_indices[1] > len(context_tokens)-1:
            print("??????????????")
            print(len(context_tokens), best_indices)
        example = {'context_tokens': context_tokens,'context_chars': context_chars, 'ques_tokens': ques_tokens,\
             'ques_chars': ques_chars, 'y1s': [best_indices[0]], 'y2s': [best_indices[1]], 'id': article['_id'], \
                'start_end_facts': start_end_facts}
        eval_example = {'context': text_context, 'spans': flat_offsets, 'answer': [answer], 'id': article['_id'],
                'sent2title_ids': sent2title_ids, 'supporting_facts':supporting_facts}
        return example, eval_example
            
    def build_features(self, examples, out_file, word2idx_dict):
        para_limit, ques_limit = 0, 0
        for example in tqdm(examples):
            para_limit = max(para_limit, len(example['context_tokens']))
            ques_limit = max(ques_limit, len(example['ques_tokens']))


        def filter_func(example):
            return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

        print("Processing {} examples...".format(out_file))
        datapoints = []
        total = 0
        total_ = 0
        for example in tqdm(examples):
            total_ += 1

            if filter_func(example):
                continue

            total += 1

            context_idxs = np.ones(para_limit, dtype=np.int64) * self.get_token_id('[PAD]')
            ques_idxs = np.ones(ques_limit, dtype=np.int64) * self.get_token_id('[PAD]')

            def _get_word(word):
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in word2idx_dict:
                        return word2idx_dict[each]
                return 1

            context_idxs[:len(example['context_tokens'])] = [_get_word(token) for token in example['context_tokens']]
            ques_idxs[:len(example['ques_tokens'])] = [_get_word(token) for token in example['ques_tokens']]

            start, end = example["y1s"][-1], example["y2s"][-1]
            y1, y2 = start, end

            datapoints.append({'context_idxs': torch.from_numpy(context_idxs),
                'ques_idxs': torch.from_numpy(ques_idxs),
                'y1': y1,
                'y2': y2,
                'id': example['id'],
                'start_end_facts': example['start_end_facts']})
        print("Build {} / {} instances of features in total".format(total, total_))
        # pickle.dump(datapoints, open(out_file, 'wb'), protocol=-1)
        torch.save(datapoints, out_file)

    def loadFile(self, fpath):
        datas = dict()
        # The dataset only has dev data, the test data can only be used in the leaderboard
        if self.distract_version == True:
            val_file = 'hotpot_dev_distractor_v1.json'
        else:
            val_file = 'hotpot_dev_fullwiki_v1.json'
        file_name_list = ['hotpot_train_v1.1.json',val_file,'hotpot_test_fullwiki_v1.json']
        data_type_list = ['train','val','test']
        for file_name,data_type in zip(file_name_list,data_type_list): #'val'            
            data_file = os.path.join(fpath,file_name)
            # Opening JSON file
            f = open(data_file)
            # returns JSON object as
            # a dictionary
            raw_data = json.load(f)
            x = []
            y = {}
            for one_row in raw_data:
                example, eval_example = self._process_article(one_row)
                x.append(example)
                y[eval_example['id']] = eval_example
            f.close()
            datas[data_type] = {'x':x,'y':y}
    
        used_length = int(self.train_ratio*len(datas['train']['x']))
        datas['train']['x'] = datas['train']['x'][:used_length]
        # datas['train']['y'] = datas['train']['y'][:used_length]

        return datas['train'],datas['val'],datas['test']     

    def get_train(self, shuffle=True):
        # print(type(DataIterator))
        return DataIterator(self.data['train']['x'],self.data['train']['y'], self.batch_size, shuffle, self.sent_limit, self.get_token_id('[PAD]'))


    def get_test(self, shuffle=False):
        return DataIterator(self.data['val']['x'],self.data['val']['y'], self.batch_size, shuffle, self.sent_limit, self.get_token_id('[PAD]'))
    
    def get_val(self, shuffle=False):
        return DataIterator(self.data['test']['x'], self.data['test']['y'],self.batch_size, shuffle, self.sent_limit, self.get_token_id('[PAD]'))


class DataIterator(object):
    def __init__(self, buckets, eval_file,  bsz, shuffle, sent_limit, pad_id, ignore_index=-100):
        self.buckets = buckets
        self.eval_file = eval_file
        self.bsz = bsz
        # if para_limit is not None and ques_limit is not None:
        #     self.para_limit = para_limit
        #     self.ques_limit = ques_limit
        # else:
        # print(pad_id,type(pad_id))
        para_limit, ques_limit = 0, 0
        for bucket in buckets:
            for dp in bucket:
                para_limit = max(para_limit, dp['context_idxs'].size(0))
                ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
        # print(para_limit,ques_limit)
        self.para_limit, self.ques_limit = para_limit, ques_limit
        self.sent_limit = sent_limit

        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __iter__(self):
        
        context_idxs = torch.ones((self.bsz, self.para_limit),dtype=torch.long).cuda() * self.pad_id
        ques_idxs = torch.ones((self.bsz, self.ques_limit),dtype=torch.long).cuda() * self.pad_id
        y1 = torch.zeros((self.bsz),dtype=torch.long).cuda()
        y2 = torch.zeros((self.bsz),dtype=torch.long).cuda()
        q_type = torch.zeros((self.bsz),dtype=torch.long).cuda()
        start_mapping = torch.zeros((self.bsz, self.para_limit, self.sent_limit)).cuda()
        end_mapping = torch.zeros((self.bsz, self.para_limit, self.sent_limit)).cuda()
        all_mapping = torch.zeros((self.bsz, self.para_limit, self.sent_limit)).cuda()
        is_support = torch.zeros((self.bsz, self.sent_limit),dtype=torch.long).cuda()

        while True:
            if len(self.bkt_pool) == 0: break
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            ids = []

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: (x['context_idxs'] != self.pad_id).long().sum(), reverse=True)

            max_sent_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping]:
                mapping.zero_()
            is_support.fill_(self.ignore_index)

            for i in range(len(cur_batch)):
                context_idxs[i].copy_(cur_batch[i]['context_idxs'])
                ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
                if cur_batch[i]['y1'] >= 0:
                    y1[i] = cur_batch[i]['y1']
                    y2[i] = cur_batch[i]['y2']
                    q_type[i] = 0
                elif cur_batch[i]['y1'] == -1:
                    y1[i] = self.ignore_index
                    y2[i] = self.ignore_index
                    q_type[i] = 1
                elif cur_batch[i]['y1'] == -2:
                    y1[i] = self.ignore_index
                    y2[i] = self.ignore_index
                    q_type[i] = 2
                elif cur_batch[i]['y1'] == -3:
                    y1[i] = self.ignore_index
                    y2[i] = self.ignore_index
                    q_type[i] = 3
                else:
                    assert False
                ids.append(cur_batch[i]['id'])

                for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
                    if j >= self.sent_limit: break
                    if len(cur_sp_dp) == 3:
                        start, end, is_sp_flag = tuple(cur_sp_dp)
                    else:
                        start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
                    if start < end:
                        start_mapping[i, start, j] = 1
                        end_mapping[i, end-1, j] = 1
                        all_mapping[i, start:end, j] = 1
                        is_support[i, j] = int(is_sp_flag)

                max_sent_cnt = max(max_sent_cnt, len(cur_batch[i]['start_end_facts']))

            input_lengths = (context_idxs[:cur_bsz] != self.pad_id).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_q_len = int((ques_idxs[:cur_bsz] != self.pad_id).long().sum(dim=1).max())

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)
            yield ({'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_lens': input_lengths,
                'start_mapping': start_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'end_mapping': end_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'pad_id':self.pad_id},  
                {'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'eval_file': self.eval_file})
