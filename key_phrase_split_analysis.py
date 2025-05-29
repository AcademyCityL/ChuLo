import dataset
import torch
from tools.textprocesser import get_preprocesser,chunk_sentence,merge_chunks, ConstituencyParser
import os
import sys
from utils.PromptRank.main import get_key_phrases
import numpy as np
from tools.params import get_params
import argparse
import dataset.text_helper as th
import math
import pke
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

def use_pke_models(text,name='yake',topk=15):
    # initialize keyphrase extraction model, here TopicRank
    if name == 'yake':
        extractor = pke.unsupervised.YAKE()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=topk)
    kps, scores = [],[]
    if name == 'yake':
        for item in keyphrases:
            kps.append(item[0])
            scores.append(item[1]*-1)

    return kps,scores

def preprocess(da_object):
    # init g_parser
    # if textprocesser.g_parser is None:
    #     textprocesser.g_parser = ConstituencyParser(da_object.parser_type)
    all_labels = []
    all_sentences = []
    all_lessprocessed_sentences = {}
    da_object.processer = get_preprocesser(da_object.preprocesser_cfg)
    attn_mode = getattr(da_object,'attn_mode',{'name':'default','param1':0})
    start_split_index = 0
    data_config = da_object.global_config['DATA']
    splits =['train','val','test'] if data_config['only_val'] == False else ['val']
    print("Process splits: ",splits)
    if "split_idx" in data_config:
        max_split_index = 0
        for split in splits:
            if split not in da_object.datasets:
                continue
            dataset = da_object.datasets[split]
            if split != 'train':
                max_split_index += 1
            max_split_index += math.ceil(len(dataset.data) / data_config['split_size']) - 1
        print('When use split_size {}, the max_split_index is {}'.format(data_config['split_size'],max_split_index))    

    for split in splits:
        if split not in da_object.datasets:
            continue
        print("---------------  ",split)
        dataset = da_object.datasets[split]
        all_labels.extend(dataset.labels)
        dataset.data = [da_object.processer.process_one(x) for x in dataset.data]
        # only use berkeley parser need pre-truncatation
        # key_phrase_split: promptrank key_phrase_split2: YAKE
        assert attn_mode['name'] in ('key_phrase_split','key_phrase_split2'), 'Only support key_phrase_split now'
        
        if data_config['whole_doc'] == False:
            max_len = da_object.max_seq_len
        else:
            max_len = 'whole_doc'

        if "split_idx" in data_config:
            max_split_index = math.ceil(len(dataset.data) / data_config['split_size']) - 1
            cur_split_index = data_config['split_idx'] - start_split_index
            if cur_split_index > max_split_index:
                start_split_index += max_split_index + 1
                continue
            cache_file = 'results/cache/key_phrase_split/{}_{}_{}_{}_{}_kps.pt'.format(\
            da_object.global_config['DATA']['dataset_name'],attn_mode['name'],split,max_len,\
                data_config['split_idx'])
        else:
            cache_file = 'results/cache/key_phrase_split/{}_{}_{}_{}_kps.pt'.format(\
                da_object.global_config['DATA']['dataset_name'],attn_mode['name'],split,max_len)
        if os.path.exists(cache_file):
            print("key_phrase_split data {} exists, exist".format(cache_file))
            sys.exit(0)
        else:
            if data_config['whole_doc'] == False:
                all_sentence = [' '.join(one_sample[0].split(' ')[:max_len]) for one_sample in dataset]
            else:
                print("whole_doc is True, use the whole doc")
                all_sentence = [one_sample[0] for one_sample in dataset]
            if "split_idx" in data_config:
                all_sentence = all_sentence[cur_split_index * data_config['split_size']:\
                                            (cur_split_index + 1) * data_config['split_size']]
            # all_sentence = [one_sample[0] for one_sample in dataset]
            # all_sentence = all_sentence[:3]
            topk = 15
            # no matter what the whole_doc is, max_len is set as da_object.max_seq_len for the use of tokenizer
            if attn_mode['name'] == 'key_phrase_split':
                all_kp,all_score = get_key_phrases(all_sentence,dataset.file_path,max_len = da_object.max_seq_len,\
                                               topk=topk,whole_doc=data_config['whole_doc'])
            elif attn_mode['name'] == 'key_phrase_split2':
                all_kp = []
                all_score = []
                for sentence in tqdm(all_sentence,desc='Use YAKE to do kp extraction'):
                    kps, scores = use_pke_models(sentence,name='yake',topk=topk)
                    all_kp.append(kps)
                    all_score.append(scores)

            max_chunk_num = 0
            chunk_num_stastics = []
            for i in range(len(all_sentence)):
                kps = all_kp[i]
                # all_sentence[i] = '[LOC] '+all_sentence[i]
                # for kp in kps:
                #     '''
                #     ERROR! The all_sentence data is error because we can't simply replace, for
                #     example, 'sa' is the kay phrase but there is a word 'same'. The right processing method
                #     is in the pre_tokenize.py
                #     '''
                #     all_sentence[i] = all_sentence[i].replace(kp, kp+' [LOC]')
                # print("chunk num: ",chunk_num)
                chunk_num = all_sentence[i].count('[LOC]')
                chunk_num_stastics.append(chunk_num)
                max_chunk_num = max(max_chunk_num,chunk_num)
            cache = {'all_kp':all_kp,'all_score':all_score,'all_sentence':all_sentence,\
                     'max_chunk_num':max_chunk_num,'chunk_num_stastics':chunk_num_stastics}
            torch.save(cache,cache_file)
            print("key_phrase_split data saved")
            print("key_phrase_split statistics when max_len is {}: avg: {}, max: {}, min: {}, std:{}".\
                  format(max_len,np.average(chunk_num_stastics),np.max(chunk_num_stastics),\
                         np.min(chunk_num_stastics),np.std(chunk_num_stastics)))
            if "split_idx" in data_config:
                break
            # print(cache)
            
th.preprocess = preprocess

def prepare_envs():
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/stat/'):
        os.mkdir('results/stat/')
    if not os.path.exists('results/cache/'):
        os.mkdir('results/cache/')
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
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--split_size', type=int, default=512)
    parser.add_argument('--not_split', action='store_true',default=False)
    parser.add_argument('--whole_doc', action='store_true',default=False)
    parser.add_argument('--only_cans', action='store_true',default=False)
    parser.add_argument('--only_val', action='store_true',default=False)
    args = parser.parse_args()
    config_file = args.config
    split_idx = args.split_idx
    split_size = args.split_size
    not_split = args.not_split
    whole_doc = args.whole_doc
    only_cans = args.only_cans
    only_val = args.only_val
    config = get_params(config_file)
    data_config = config['DATA']
    print("not_split ",not_split)
    if not_split == False:
        data_config['split_idx'] = split_idx
        data_config['split_size'] = split_size
    data_config['whole_doc'] = whole_doc
    data_config['only_cans'] = only_cans
    data_config['only_val'] = only_val
    data = dataset.get_data(data_config,'gpu',config)
    # sys.exit(0)
    preprocess(data)