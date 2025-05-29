import pandas as pd
import os
import numpy as np
import json
import math
import typing
from typing import List,Dict,Any
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer
import nltk
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, Linear
import math
import datasets
import dataset
from tools.params import get_params
import argparse
from tqdm import tqdm


def prepare_envs():
    if not os.path.exists('results/stat/'):
        os.mkdir('results/stat/')
    if not os.path.exists('results/cache/'):
        os.mkdir('results/cache/')
    if not os.path.exists('results/cache/tokenized_results/'):
        os.mkdir('results/cache/tokenized_results/')
    if not os.path.exists('results/cache/vocabs/'):
        os.mkdir('results/cache/vocabs/')

def tokenize_and_align_labels(tokenizer, word_lists, token_label_lists, label_encoder):
    tokenized_inputs = tokenizer(word_lists, truncation=False, padding = False, is_split_into_words=True)

    labels = []
    for i, one_sent_label_list in enumerate(token_label_lists):
        one_sent_label_list = label_encoder.transform(one_sent_label_list)
        token_label_type = len(one_sent_label_list.shape)
        if token_label_type == 1:
            pad_token_label = -100
        elif token_label_type == 2:
            pad_token_label = [-100] * len(one_sent_label_list[0])

        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(pad_token_label)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(one_sent_label_list[word_idx])
            else:
                label_ids.append(pad_token_label)
            previous_word_idx = word_idx
        labels.append(label_ids)
    return {'o_input_ids':tokenized_inputs['input_ids'], 'o_attention_mask':tokenized_inputs['attention_mask'],'o_token_labels':labels}

def get_qatask_decoder_input(inputs):
    # the input ids are [cls] question tokens [sep] context tokens [sep] 
    # we use longformer's self attention pattern, i.e., add global attention to the question tokens and add sliding window attention to the context tokens.
    # so we need to modify the attention mask, tobe, 0: masked tokens, 1: local (sliding window) tokens, 2: global attention tokens
    merged_attention_masks = []
    for attention_mask, context_start in zip(inputs['attention_mask'], inputs['context_start_positions']):
        # print(type(attention_mask),type(context_start))
        attention_mask = torch.tensor(attention_mask)
        attention_mask[:context_start] = 2
        merged_attention_masks.append(attention_mask.tolist())
    return {'o_input_ids':inputs['input_ids'], 'o_attention_mask':merged_attention_masks,'o_context_start_positions':inputs['context_start_positions'],'o_start_positions':inputs["start_positions"], 'o_end_positions':inputs["end_positions"], "o_offset_mapping":inputs["o_offset_mapping"]}



def preprocess_function(tokenizer, questions, contexts, ans_list_list):
    questions = [q.strip() for q in questions]
    inputs = tokenizer(
        questions,
        contexts,

        padding=False,
        truncation=False,
        return_attention_mask = True,
        return_offsets_mapping=True,
        return_special_tokens_mask = True,
        return_token_type_ids = True,
        return_tensors = None,
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []
    context_start_positions = []

    for i, offset in enumerate(offset_mapping):
        answer_turple = ans_list_list[i][0] # For QuAC dataset, its train split has only one answer per question, but multiple answers in val split. Here we only process train split case, for val split, we use evaluate script 
        start_char = answer_turple[1]
        end_char = start_char + len(answer_turple[0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context (sequence_id = 0 is question tokens)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        context_start_positions.append(idx)
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["context_start_positions"] = context_start_positions
    inputs["o_offset_mapping"] = offset_mapping
    return inputs

if __name__=="__main__":
    prepare_envs()
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config', default='ERROR')
    args = parser.parse_args()
    config_file = args.config
    data_name = config_file.split('_')[0].split('/')[-1]
    config = get_params(config_file)
    data_config = config['DATA']
    tokenizer_name = data_config['tokenizer_type'] + '_' + data_config['tokenizer_name'].replace('/','_')
    print("data_config ",data_config)
    daobj = dataset.get_data(data_config,'gpu',config,pre_cache=False)
    stat = []
    bs = 512
    cache_prefix = {'hp':'hyperpartisan','imdb':'imdb','mr':'mr','lun':'lun','ng20':'ng20','r8':'r8','bbcn':'bbcn','bs':'bs','bs-pair':'bs', 'eurlex':'eurlex','eurlex-inverse':'eurlex','gum':'gum','quac':'quac','conll':'conll'}   
    # LOC_ID = daobj.tokenizer.get_vocab()['[LOC]']
    # PAD_ID = daobj.tokenizer.get_vocab()['[PAD]']  
    attn_mode = getattr(daobj,'attn_mode',{'name':'default','param1':0})
    assert attn_mode['name'] in ('key_phrase_split','key_phrase_split2','key_phrase_chunk_rep','key_phrase_chunk_rep2'), 'Only support key_phrase_split and key_phrase_chunk_rep now'
    if attn_mode['name'] in 'key_phrase_split':
        mid_name = 'key_phrase_chunk_rep'
    elif attn_mode['name'] == 'key_phrase_split2':
        mid_name = 'key_phrase_chunk_rep2'
    else:
        mid_name = attn_mode['name']
    
    if attn_mode['name'] in 'key_phrase_chunk_rep':
        f_mid_name = 'key_phrase_split'
    elif attn_mode['name'] == 'key_phrase_chunk_rep2':
        f_mid_name = 'key_phrase_split2'
    else:
        f_mid_name =  attn_mode['name']
    
    max_len_name = 'whole_doc' 
    # dataset_name = daobj.global_config['DATA']['dataset_name']
    # if dataset_name == 'eurlex':
    #     if daobj.global_config['DATA'].get('inverse', False) == True:
    #         dataset_name = 'eurlex_inverse'
    # elif dataset_name == 'bs':
    #     if daobj.global_config['DATA'].get('pair', False) == True:
    #         dataset_name = 'bs-pair'
    for split in ['train','val','test']:
        if split not in daobj.datasets:
            continue
        all_idx = []
        all_targets = []
        all_sentences = []
        all_word_list = []
        all_sentences_loc = []
        all_kp_scores = []
        all_kps = []
        dsobj = daobj.datasets[split]
        f = 'results/cache/key_phrase_split/{}_{}_{}_whole_doc_kps.pt'.format(\
            cache_prefix[data_name],f_mid_name,split)
        cache = torch.load(f)
        token_wise_task = getattr(daobj, "token_wise_task", None)
        token_wise_task_qa = getattr(daobj, "token_wise_task_qa", None)
        if token_wise_task == True:
            for sentence,label, word_list, idx in daobj.datasets[split]:
                all_idx.append(idx)
                all_targets.append(label)
                all_sentences.append(sentence)
                all_word_list.append(word_list)
        elif token_wise_task_qa == True:
            all_questions = []
            all_dialogue_ids = []
            all_dif_idx = {}
            dislogue_id_set = set()
            for sentence, question, ans_list, dislogue_id, idx in daobj.datasets[split]:
                all_idx.append(idx)
                all_targets.append(ans_list)
                all_sentences.append(sentence)
                all_questions.append(question)
                all_dialogue_ids.append(dislogue_id)
                all_dif_idx[idx] = len(set(all_dialogue_ids))-1
        else:
            for sentence,label,idx in daobj.datasets[split]:
                all_idx.append(idx)
                all_targets.append(label)
                all_sentences.append(sentence)

        if data_name == 'bs-pair':
            new_cache_kp_scores = []
            new_cache_kps = []
            for i in range(0, len(cache['all_kp']) - 1, 2):
                kp_set = set(cache['all_kp'][i] + cache['all_kp'][i+1])
                kps_tmp = cache['all_kp'][i] + cache['all_kp'][i+1]
                kpscore_tmp = cache['all_score'][i] + cache['all_score'][i+1]
                kps_add = []
                kpscore_add = []
                for kp, score in zip(kps_tmp,kpscore_tmp):
                    if kp in kp_set:
                        kps_add.append(kp)
                        kpscore_add.append(score)
                        kp_set.remove(kp)
                        
                new_cache_kp_scores.append(kpscore_add)
                new_cache_kps.append(kps_add)
            cache['all_score'] = new_cache_kp_scores
            cache['all_kp'] = new_cache_kps
            print("len(all_sentences) ",len(all_sentences),len(new_cache_kp_scores),len(new_cache_kps))
            assert len(new_cache_kp_scores)==len(all_sentences),"ERROR pre tokenized when bs-pair!"

        # print("cache['all_kp'] ",len(cache['all_kp']),all_dif_idx[83567])
        if token_wise_task_qa == True:
            for i in range(len(all_sentences)):
                ori_idx = all_dif_idx[i]
                # print("cache['all_kp'] ",len(cache['all_kp']),all_dif_idx[83567])
                if len(cache['all_kp'][ori_idx]) == 0:
                    # occurs in MR dataset
                    all_kp_scores.append([-0.1])
                    all_kps.append([all_sentences[i].split(' ')[0]])
                else:   
                    all_kp_scores.append(cache['all_score'][ori_idx])
                    all_kps.append(cache['all_kp'][ori_idx])
        else:
            for i in range(len(all_sentences)):
                if len(cache['all_kp'][i]) == 0:
                    # occurs in MR dataset
                    all_kp_scores.append([-0.1])
                    all_kps.append([all_sentences[i].split(' ')[0]])
                else:   
                    all_kp_scores.append(cache['all_score'][i])
                    all_kps.append(cache['all_kp'][i])
        
       

        original_cache = {'input_ids':[],'attention_mask':[],'token_type_ids':[],'special_tokens_mask':[],\
                           'offset_mapping':[]}
        tokenized_cache = {}
        tokenized_cache.update(original_cache)
        all_kp_token_type_ids = [] # 1 for key phrase tokens, 0 for non key phrase tokens
        all_kp_token_scores = [] # score for key phrase tokens, 0 for non key phrase tokens
        if token_wise_task == True or token_wise_task_qa == True:
            if token_wise_task == True:
                add_dict = {'o_input_ids':[], 'o_attention_mask':[],'o_token_labels':[]}
                tokenized_cache.update(add_dict)
                for i in tqdm(range(0,len(all_sentences),bs),desc='Construct decoder input for token wise task gum'):
                    word_lists = all_word_list[i:i+bs]
                    token_label_lists = all_targets[i:i+bs]
                    inputs_for_token_wise_task = tokenize_and_align_labels(daobj.tokenizer, word_lists, token_label_lists, daobj.label_encoder)
                    for k,v in add_dict.items():
                        tokenized_cache[k].extend(inputs_for_token_wise_task[k])
            elif token_wise_task_qa == True:
                add_dict = {'o_input_ids':[], 'o_attention_mask':[],'o_context_start_positions':[],'o_start_positions':[],'o_end_positions':[],'o_offset_mapping':[]}
                tokenized_cache.update(add_dict)
                for i in tqdm(range(0,len(all_sentences),bs),desc='Construct decoder input for token wise task quac'):
                    contexts = all_sentences[i:i+bs]
                    ans_list_list = all_targets[i:i+bs]
                    questions = all_questions[i:i+bs]
                    inputs_for_token_wise_task = preprocess_function(daobj.tokenizer,questions,contexts,ans_list_list)
                    # convert start_positions and end_positions to token labels
                    inputs_for_token_wise_task = get_qatask_decoder_input(inputs_for_token_wise_task)

                    for k,v in add_dict.items():
                        tokenized_cache[k].extend(inputs_for_token_wise_task[k])
                
        for i in tqdm(range(0,len(all_sentences),bs),desc='Construct kp_token_type_ids'):
            inputs_ori = daobj.tokenizer(
                    all_sentences[i:i+bs],
                    padding=False,
                    truncation=False,
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = None,
                )
            

            bt_loc = False
            for l in tqdm(range(len(inputs_ori['input_ids'])),desc='Process one batch'):
                iids_ori = inputs_ori['input_ids'][l]
                kps = all_kps[i+l][:10]
                # print(len(kps))
                # print(kps)
                kp_scores = all_kp_scores[i+l][:10]
                # print(len(kp_scores))
                # print(kp_scores)
                # import sys
                # sys.exit()
                kps_input = daobj.tokenizer(
                    kps,
                    padding=False,
                    truncation=False,
                    return_attention_mask = False,
                    return_offsets_mapping= False,
                    return_special_tokens_mask = False,
                    return_token_type_ids = False,
                    return_tensors = None,
                )
                # add two spaces to avoid sub-word replacement
                iids_ori_str = ' ' + ' '.join([str(c) for c in iids_ori]) + ' '
                # print("iids_ori ",iids_ori)
                for m in range(len(kps_input['input_ids'])):
                    kp_ids = kps_input['input_ids'][m]
                # for kp_ids, score in zip(kps_input['input_ids'],kp_scores):
                    kp_ids_str = ' ' + ' '.join([str(c) for c in kp_ids[1:-1]]) + ' ' # no need [CLS] and [SEP]
                    to_rep = ' ' + ' '.join(['-10{}'.format(m)] * (len(kp_ids)-2)) + ' '
                    # print("kp_ids_str ",kp_ids_str)
                    # print("to_rep ",to_rep)
                    iids_ori_str = iids_ori_str.replace(kp_ids_str, to_rep)
                # print("iids_ori_str ",iids_ori_str)
                iids_kp = [int(c) for c in iids_ori_str[1:-1].split(' ')]
                kp_token_type_ids = []
                kp_token_scores = []
                for c in iids_kp:
                    if c < 0:
                        kp_token_type_ids.append(1)
                        # print("int(str(c)[3:] ",int(str(c)[3:]))
                        kp_token_scores.append(kp_scores[int(str(c)[3:])])
                    else:
                        kp_token_type_ids.append(0)
                        kp_token_scores.append(0)
                if len(kp_token_type_ids) != len(iids_ori):
                    print("all_sentences ",all_sentences[i+l])
                    print("kps",cache['all_kp'][i+l])
                    print("iids_ori ",iids_ori)
                    
                assert len(kp_token_type_ids) == len(iids_ori), "kp_token_type_ids error! {} and {}".\
                    format(len(kp_token_type_ids),len(iids_ori))
                all_kp_token_type_ids.append(kp_token_type_ids)
                all_kp_token_scores.append(kp_token_scores)
            for k,v in original_cache.items():
                tokenized_cache[k].extend(inputs_ori[k])
        tokenized_cache['kp_token_scores'] = all_kp_token_scores
        tokenized_cache['kp_token_type_ids'] = all_kp_token_type_ids
        tokenized_cache['sentences'] = all_sentences
        f_tokenized_cache = []
        for i in range(len(tokenized_cache['sentences'])):
            one_sample_cache = {}
            for k,v in tokenized_cache.items():
                one_sample_cache[k] = v[i]
            f_tokenized_cache.append(one_sample_cache)
        torch.save(f_tokenized_cache,'results/cache/tokenized_results/{}_{}_{}_{}_{}_top10.pt'.format(daobj.global_config['DATA']['dataset_name'],mid_name,split,tokenizer_name,max_len_name))

    