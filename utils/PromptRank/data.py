#coding=utf-8
import re
import codecs
import json
import os
import sys
import logging
import requests
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

MAX_LEN = None
enable_filter = None
temp_en = None
temp_de = None

StanfordCoreNLP_path = './utils/stanford-corenlp-4.5.4'
# StanfordCoreNLP_path = '../stanford-corenlp-4.5.4'

def _request(self, annotators=None, data=None, *args, **kwargs):
    if sys.version_info.major >= 3:
        data = data.encode('utf-8')

    properties = {'annotators': annotators, 'outputFormat': 'json', 'timeout': '50000'}
    params = {'properties': str(properties), 'pipelineLanguage': self.lang}
    if 'pattern' in kwargs:
        params = {"pattern": kwargs['pattern'], 'properties': str(properties), 'pipelineLanguage': self.lang}

    logging.info(params)
    r = requests.post(self.url, params=params, data=data, headers={'Connection': 'close'})
    r_dict = json.loads(r.text)

    return r_dict

StanfordCoreNLP._request = _request

stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None


GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    cans_count = dict()
    
    np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
    keyphrase_candidate = []
    postags_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    # print("np_pos_tag_tokens ",np_pos_tag_tokens)
    # import sys
    # sys.exit()
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            postags = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            
            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1

            postags_candidate.append(postags)
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1
    # print("keyphrase_candidate ",keyphrase_candidate)
    if len(keyphrase_candidate) == 0:
        count = 0
        # avoid empty candidate
        for token in np_pos_tag_tokens:
            np = token[0]
            length = 1
            start_end = (count, count + length)
            count += length

            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1
            # postags_candidate.append(token._label)？
            keyphrase_candidate.append((np, start_end))
            if len(keyphrase_candidate) > 15:
                break

        else:
            count += 1
        print("avoid empty candidate, ",keyphrase_candidate)
        
    if enable_filter == True:
        i = 0
        while i < len(keyphrase_candidate):
            can, pos = keyphrase_candidate[i]
            #pos[0] > 50 and
            if can in cans_count.keys() and cans_count[can] == 1:
                keyphrase_candidate.pop(i)
                postags_candidate.pop(i)
                continue
            i += 1
    
    return keyphrase_candidate,postags_candidate

class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        try:
            self.tokens = en_model.word_tokenize(text)
            self.tokens_tagged = en_model.pos_tag(text)
            # print("self.tokens ",self.tokens)
        except:
            # print("text ",text)
            # print("----------------------------------")
            token_list = []
            token_tag_list = []
            word_list = text.split(' ')
            # print("word_list ",len(word_list))
            for i in range(0,len(word_list),10000):
                # print("local text -start: ",i)
                local_text = " ".join(word_list[i:i+10000])
                token_list += en_model.word_tokenize(local_text)
                token_tag_list += en_model.pos_tag(local_text)
            self.tokens = token_list
            self.tokens_tagged = token_tag_list
        # print("len of tokens: ", len(self.tokens),len(self.tokens_tagged))
        assert len(self.tokens) == len(self.tokens_tagged), "pos tagging error!"
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate, self.postags_candidate = extract_candidates(self.tokens_tagged, en_model)
        
class KPE_Dataset(Dataset):
    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0]
        en_input_mask = doc_pair[1]
        de_input_ids = doc_pair[2]
        dic = doc_pair[3]

        return [en_input_ids, en_input_mask, de_input_ids, dic]
    
def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new

def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels

def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc =abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text,database="Duc2001")
                data[fname]=text.strip("\n")
                # data[fname] = text
    return data,labels

def get_inspec_data(file_path="data/Inspec"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("%", '')
                text=clean_text(text)
                data[left]=text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text=text.replace("\n",' ')
                text=clean_text(text,database="Inspec")
                text=text.lower()
                label=text.split("; ")
                labels[left]=label
    return data,labels

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels

def remove (text):
    if len(text) == 1:
        return True
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False
    
def generate_doc_pairs(doc, candidates, idx,whole_doc=False):
    count = 0
    doc_pairs = []
    # print("len(doc) ",len(doc.split()),whole_doc,MAX_LEN)
    en_input =  tokenizer(doc, max_length=MAX_LEN, padding="max_length", return_overflowing_tokens = whole_doc,\
                          truncation=True, return_tensors="pt",return_attention_mask=True)
    of_en_input_ids = en_input["input_ids"]
    of_en_input_mask = en_input["attention_mask"]
    total_num = 0
    sub_idx = 0
    
    # print("of_en_input_ids ",len(of_en_input_ids),len(of_en_input_ids[0]))
    for i in range(len(of_en_input_ids)):
        for id, can_and_pos in enumerate(candidates):
            candidate = can_and_pos[0]
            # Remove stopwords in a candidate
            if i == len(of_en_input_ids) - 1 and id == len(candidates) - 1 and len(doc_pairs) == 0:
                print("len(doc_pairs) is still 0 at last one, not remove sp ",doc,candidates)
                pass
            else:
                if remove(candidate):
                    count +=1
                    continue
        
            de_input = temp_de + candidate + " ."
            de_input_ids = tokenizer(de_input, max_length=30, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
            de_input_ids[0, 0] = 0
            de_input_len = (de_input_ids[0] == tokenizer.eos_token_id).nonzero()[0].item() - 2
            
    #         for i in de_input_ids[0]:
    #             print(tokenizer.decode(i))
    #         print(de_input_len)
            
    #         x = tokenizer(temp_de, return_tensors="pt")["input_ids"]
    #         for i in x[0]:
    #             print(tokenizer.decode(i))
    #         exit(0)
            # print("sub_idx ",idx, sub_idx)
            dic = {"de_input_len":de_input_len, "candidate":candidate, "idx":idx, "sub_idx":sub_idx, \
                   "pos":can_and_pos[1][0]}
            
            doc_pairs.append([of_en_input_ids[i], of_en_input_mask[i], de_input_ids[0], dic])
            # print(tokenizer.decode(en_input_ids[0]))
            # print(tokenizer.decode(de_input_ids[0]))
            # print(candidate)
            # print(de_input_len)
            # print()
            # exit(0)
            total_num += 1
        sub_idx += 1
        # print("sub_idx ",sub_idx)
        # print("total_num ",total_num)
    return doc_pairs, count
        

def init(setting_dict):
    '''
    Init template, max length and tokenizer.
    '''

    global MAX_LEN, temp_en, temp_de, tokenizer, enable_filter
    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_filter = setting_dict["enable_filter"]

    tokenizer = AutoTokenizer.from_pretrained("t5-" + setting_dict["model"],model_max_length=MAX_LEN)

def data_process(setting_dict, dataset_dir, dataset_name):
    '''
    Core API in data.py which returns the dataset
    '''

    init(setting_dict)

    if dataset_name =="SemEval2017":
        data, referneces = get_semeval2017_data(dataset_dir + "/docsutf8", dataset_dir + "/keys")
    elif dataset_name == "DUC2001":
        data, referneces = get_duc2001_data(dataset_dir)
    elif dataset_name == "nus" :
        data, referneces = get_long_data(dataset_dir + "/nus_test.json")
    elif dataset_name == "krapivin":
        data, referneces = get_long_data(dataset_dir + "/krapivin_test.json")
    elif dataset_name == "kp20k":
        data, referneces = get_short_data(dataset_dir + "/kp20k_valid200_test.json")
    elif dataset_name == "SemEval2010":
        data, referneces = get_short_data(dataset_dir + "/semeval_test.json")
    else:
        data, referneces = get_inspec_data(dataset_dir)
    
    docs_pairs = []
    doc_list = []
    labels = []
    labels_stemed = []
    t_n = 0
    candidate_num = 0
    porter = nltk.PorterStemmer()

    for idx, (key, doc) in enumerate(data.items()):

        # Get stemmed labels and document segments
        labels.append([ref.replace(" \n", "") for ref in referneces[key]])
        labels_s = []
        for l in referneces[key]:
            tokens = l.split()
            labels_s.append(' '.join(porter.stem(t) for t in tokens))

        doc = ' '.join(doc.split()[:MAX_LEN])  
        labels_stemed.append(labels_s)
        doc_list.append(doc)
        
        # Statistic on empty docs
        empty_doc = 0
        try:
            text_obj = InputTextObj(en_model, doc)
        except:
            empty_doc += 1
            print("id: ", idx)
            print("doc: ", doc)

        # Generate candidates (lower)
        cans = text_obj.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            if enable_filter == True and len(can.split()) > 4:
                continue
            candidates.append([can.lower(), pos])
        candidate_num += len(candidates)
        
        # Generate docs_paris for constructing dataset 
        # doc = doc.lower()
        doc = temp_en + "\"" + doc + "\""
        doc_pairs, count = generate_doc_pairs(doc, candidates, idx)
        docs_pairs.extend(doc_pairs)
        t_n += count
    

    print("candidate_num: ", candidate_num)
    print("unmatched: ", t_n)
    dataset = KPE_Dataset(docs_pairs)
    print("examples: ", dataset.total_examples)

    en_model.close()
    return dataset, doc_list, labels, labels_stemed

# add for Sparse_Transformer
def data_process_all_kpcans_tags(setting_dict, samples):
    init(setting_dict)
    return_data = []
    for idx in range(len(samples)):
        doc = samples[idx]
        
        text_obj = InputTextObj(en_model, doc)

        cans = text_obj.keyphrase_candidate
        postags = text_obj.postags_candidate
        candidates = []
        postags_candidates = []
        for i in range(len(cans)):
            can, pos = cans[i]
        # for can, pos in cans:
            if enable_filter == True and len(can.split()) > 4:
                continue
            candidates.append([can.lower(), pos])
            postags_candidates.append(postags[i])
        candidate_num += len(candidates)
        return_data.append([doc,candidates,postags_candidates])
    return return_data

# modified for Sparse_Transformer
def data_process_st(setting_dict, samples, dataset_name,whole_doc=False):
    '''
    Core API in data.py which returns the dataset
    '''


    # tokenizer = T5Tokenizer.from_pretrained("t5-" + setting_dict["model"], model_max_length=setting_dict['max_len'])

    if "imdb" in dataset_name or 'mr' in dataset_name:
        setting_dict["temp_en"] = 'movie review:'
        setting_dict["temp_de"] = "this movie review mainly talks about "
    elif 'hyperpartisan' in dataset_name:
        setting_dict["temp_en"] = 'news:'
        setting_dict["temp_de"] = "this news mainly talks about "
    elif 'lun' in dataset_name:
        setting_dict["temp_en"] = 'news:'
        setting_dict["temp_de"] = "this news mainly talks about "
    elif 'ng20' in dataset_name:
        setting_dict["temp_en"] = 'news:'
        setting_dict["temp_de"] = "this news mainly talks about "
    elif 'r8' in dataset_name:
        setting_dict["temp_en"] = 'record:'
        setting_dict["temp_de"] = "this record mainly talks about "
    elif 'bbcn' in dataset_name:
        setting_dict["temp_en"] = 'news:'
        setting_dict["temp_de"] = "this news mainly talks about "
    elif 'bs' in dataset_name:
        setting_dict["temp_en"] = 'book summary:'
        setting_dict["temp_de"] = "this book summary mainly talks about "
    elif 'eurlex' in dataset_name:
        setting_dict["temp_en"] = 'legal document:'
        setting_dict["temp_de"] = "this legal document mainly talks about "
    elif 'gum' in dataset_name:
        setting_dict["temp_en"] = 'article:'
        setting_dict["temp_de"] = "this article mainly talks about "
    elif 'quac' in dataset_name:
        setting_dict["temp_en"] = 'article:'
        setting_dict["temp_de"] = "this article mainly talks about "
    elif 'conll' in dataset_name:
        setting_dict["temp_en"] = 'document:'
        setting_dict["temp_de"] = "this document mainly talks about "
    
    print("setting_dict ",setting_dict)
    print("dataset_name ",dataset_name)
    init(setting_dict)

    docs_pairs = []
    doc_list = []
    t_n = 0
    candidate_num = 0
    # print("len(samples) ",len(samples))
    skip = 0
    for idx in range(len(samples)):
        doc = samples[idx]
        doc_list.append(doc)
        
        text_obj = InputTextObj(en_model, doc)

        cans = text_obj.keyphrase_candidate
        candidates = []
        for can, pos in cans:
            if enable_filter == True and len(can.split()) > 4:
                continue
            candidates.append([can.lower(), pos])
        candidate_num += len(candidates)
        
        # Generate docs_paris for constructing dataset 
        # doc = doc.lower()
        doc = temp_en + "\"" + doc + "\""
        doc_pairs, count = generate_doc_pairs(doc, candidates, idx,whole_doc=whole_doc)
        docs_pairs.extend(doc_pairs)
        t_n += count
    

    # print("candidate_num: ", candidate_num)
    print("unmatched: ", t_n)
    dataset = KPE_Dataset(docs_pairs)
    # print("examples: ", dataset.total_examples)
    # print("len(dataset) ",len(dataset))
    # en_model.close()
    return dataset, doc_list


