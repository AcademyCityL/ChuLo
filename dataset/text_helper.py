import torch
from tools.textprocesser import get_preprocesser,chunk_sentence,merge_chunks, ConstituencyParser
import tools.textprocesser as textprocesser
from torch.utils.data import DataLoader
from tools.tokenizer import get_tokenizer
import os
import pandas as pd
import copy
from transformers import T5Tokenizer
import nltk
# from utils.PromptRank.main import get_key_phrases
import numpy as np
import random
from pre_tokenize import tokenize_and_align_labels,preprocess_function,get_qatask_decoder_input
'''
The following methods can only help with customized method. The pytorch-lightning method can't be
moved here
'''

chunk_statistics = []

def cut_and_pad(ds_object,sentences,vocab = None):
    used_vocab = ds_object.token2id if vocab is None else vocab
    if ds_object.max_seq_len > 0:
        max_len = min(max([len(sentence) for sentence in sentences]),ds_object.max_seq_len)
    elif ds_object.max_seq_len == -1:
        max_len = max([len(sentence) for sentence in sentences])
    new_sentences = []
    masks = []
    for sentence in sentences:
        if len(sentence) > max_len:
            new_sentence = sentence[:max_len]
            masks.append([1]*max_len)
        else:
            new_sentence = sentence + [used_vocab['[PAD]']] * (max_len - len(sentence)) 
            masks.append([1]*len(sentence)+[0]*(max_len - len(sentence)))
        new_sentences.append(new_sentence)
    return(new_sentences, masks)

def pad_and_align_for_kpcr_with_sentemb(inputs,max_len,attn_mode,tokenizer,token_wise_task=None,token_wise_task_qa = None):
    '''
    Remove cls and sep token, will add back after getting chunk representations
    chunk within each sentence
    '''
    # print("start pad_and_align_for_kpcr")
    chunk_len = attn_mode['param1']
    # max_token_len = max_len * chunk_len
    all_kp_token_weights = []
    new_inputs = {'sentences':[],'input_ids':[],'attention_mask':[],'token_type_ids':[],'special_tokens_mask':[],'offset_mapping':[],'kp_token_type_ids':[],'kp_token_scores':[],'kp_token_weights':[],'map_ids':[],'sent_map_ids':[],'sentence_textrank_scores':[]}

    # for decoder
    if token_wise_task == True:
        keep_fields = ['o_input_ids', 'o_attention_mask','o_token_labels']
        pad_and_align_for_token_wise_task(inputs, tokenizer, False)
    elif token_wise_task_qa == True:
        keep_fields = ['o_input_ids', 'o_attention_mask','o_context_start_positions','o_start_positions','o_end_positions','o_questions','o_dialogue_ids','o_sentences','o_offset_mapping']
        pad_and_align_for_token_wise_task(inputs, tokenizer, True)

    

    for i in range(len(inputs['input_ids'])):

            
        for k,v in inputs.items():
            if k not in ('sentences','textrank_data'): # remove CLS and SEP
                inputs[k][i] = inputs[k][i][1:-1]

        if attn_mode['param2'] == 'fixed_weights':
            weights_cfg = attn_mode['param3'] #[non key phrase token weight, key phrase token weight]
            weights_cfg = [float(c) for c in weights_cfg.split('_')]
            one_kp_token_weights = []
            for type_id in inputs['kp_token_type_ids'][i]:
                if type_id == 0: # non kp tokens
                    one_kp_token_weights.append(weights_cfg[0])
                elif type_id == 1: # kp tokens
                    one_kp_token_weights.append(weights_cfg[1])
                else:
                    assert False, 'assign weights error'
            all_kp_token_weights.append(one_kp_token_weights)
            # print("one_kp_token_weights ",one_kp_token_weights[0])
        elif attn_mode['param2'] == 'average_weights':
            all_kp_token_weights.append([1]*len(inputs['kp_token_type_ids'][i]))
        original_ordered_sentences_data = sorted(inputs['textrank_data'][i],key=lambda x: x[0])
        seq_length_after_chunk = 0
        trauncate = False
        new_inputs['sent_map_ids'].append([])
        for j in range(len(original_ordered_sentences_data)): # j is the sent_id
            if trauncate == True:
                break
            sent_start, sent_end = original_ordered_sentences_data[j][3]
            sentence_importance = 1-original_ordered_sentences_data[j][2]# change it from lower is better to higher is better
            # because we removed cls and sep in the begingning, so the start and end neet to reduce 1
            sent_start = sent_start - 1
            sent_end = sent_end - 1
            # print("ori len ",len(inputs['input_ids'][i]))
            seq_length_after_chunk += 1 # each sentence has a sent_emb
            if seq_length_after_chunk >= max_len: # if the last one is sent_emb only without chunk emb, drop it
                trauncate = True
                break
            for l in range(0,len(inputs['input_ids'][i][sent_start:sent_end]),chunk_len):
                # check and truncate
                seq_length_after_chunk += 1
                if seq_length_after_chunk > max_len:
                    trauncate = True
                    break
                new_inputs['sentences'].append(inputs['sentences'][i])
                ids = inputs['input_ids'][i][sent_start:sent_end][l:l+chunk_len]
                new_inputs['input_ids'].append(inputs['input_ids'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['attention_mask'].append(inputs['attention_mask'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['token_type_ids'].append(inputs['token_type_ids'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['special_tokens_mask'].append(inputs['special_tokens_mask'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['offset_mapping'].append(inputs['offset_mapping'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['kp_token_type_ids'].append(inputs['kp_token_type_ids'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['kp_token_scores'].append(inputs['kp_token_scores'][i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['kp_token_weights'].append(all_kp_token_weights[i][sent_start:sent_end][l:l+chunk_len])
                new_inputs['map_ids'].append(i)
                new_inputs['sent_map_ids'][-1].append(j)
            new_inputs['map_ids'].append(i) # this one is used for the sent emb inserted later
                # print(" print(len(ids)) ",len(new_inputs['input_ids'][-1]))
            # check and pad
            extra_token_num = len(inputs['input_ids'][i][sent_start:sent_end]) % chunk_len
            # print("extra_token_num ",len(inputs['input_ids'][i][sent_start:sent_end]),extra_token_num)
            pre_len = len(new_inputs['input_ids'][-1])
            if extra_token_num > 0 and pre_len < chunk_len: # when truncate happened, pre_len may == chunk_len
                pad_num = chunk_len - extra_token_num
                new_inputs['input_ids'][-1].extend([tokenizer.pad_token_id]*pad_num)
                new_inputs['attention_mask'][-1].extend([0]*pad_num)
                new_inputs['token_type_ids'][-1].extend([0]*pad_num)
                new_inputs['special_tokens_mask'][-1].extend([1]*pad_num)
                new_inputs['offset_mapping'][-1].extend([[0,0]]*pad_num)
                new_inputs['kp_token_type_ids'][-1].extend([-1]*pad_num)
                new_inputs['kp_token_scores'][-1].extend([0]*pad_num)
                new_inputs['kp_token_weights'][-1].extend([0]*pad_num)
            new_inputs['sentence_textrank_scores'].append(sentence_importance)
            # if len(new_inputs['input_ids'][-1]) in (12,17):
            #     print("extra_token_num ",pre_len, extra_token_num, pad_num, len(new_inputs['input_ids'][-1]) , len(inputs['input_ids'][i][sent_start:sent_end]))
            # print("len(ids) ",len(new_inputs['input_ids'][-1]))
            # print('---------')
    if token_wise_task == True or token_wise_task_qa == True:
        for k in keep_fields:
            new_inputs[k] = inputs[k]
    return new_inputs

def pad_and_align_for_token_wise_task(inputs, tokenizer, forquac = False):
    # note: use left padding for sliding window attn
    o_max_len = 0
    if forquac == False:
        left = True
        token_label_type = len(torch.tensor(inputs['o_token_labels'][0]).shape)
    else:
        left = False
    for i in range(len(inputs['o_input_ids'])):
        if o_max_len < len(inputs['o_input_ids'][i]):
            o_max_len = len(inputs['o_input_ids'][i])

    if left == True:
        for i in range(len(inputs['o_input_ids'])):
            if len(inputs['o_input_ids'][i]) < o_max_len:
                pad_num = o_max_len - len(inputs['o_input_ids'][i])

                inputs['o_input_ids'][i] = [tokenizer.pad_token_id]*pad_num + inputs['o_input_ids'][i]
                inputs['o_attention_mask'][i] = [0]*pad_num + inputs['o_attention_mask'][i]

                if token_label_type == 1:
                    pad_token_label = -100
                    inputs['o_token_labels'][i] = [pad_token_label]*pad_num + inputs['o_token_labels'][i]
                elif token_label_type == 2:
                    pad_token_label = [-100] * len(inputs['o_token_labels'][i][0])
                    inputs['o_token_labels'][i] = [pad_token_label]*pad_num + inputs['o_token_labels'][i]
    elif left == False:
        for i in range(len(inputs['o_input_ids'])):
            if len(inputs['o_input_ids'][i]) < o_max_len:
                pad_num = o_max_len - len(inputs['o_input_ids'][i])

                inputs['o_input_ids'][i] = inputs['o_input_ids'][i] + [tokenizer.pad_token_id]*pad_num
                inputs['o_attention_mask'][i] = inputs['o_attention_mask'][i] + [0]*pad_num
    return inputs

    

def pad_and_align_for_kpcr(inputs,max_len,attn_mode,tokenizer,token_wise_task=None,token_wise_task_qa = None):
    '''
    Remove cls and sep token, will add back after getting chunk representations
    '''
    # print("start pad_and_align_for_kpcr")
    chunk_len = attn_mode['param1']
    max_token_len = max_len * chunk_len
    all_kp_token_weights = []
    new_inputs = {'sentences':[],'input_ids':[], 'attention_mask':[], 'token_type_ids':[],'special_tokens_mask':[],
                  'offset_mapping':[],'kp_token_type_ids':[],'kp_token_scores':[],'kp_token_weights':[],
                  'map_ids':[]}
            
    # for decoder
    if token_wise_task == True:
        keep_fields = ['o_input_ids', 'o_attention_mask','o_token_labels']
        pad_and_align_for_token_wise_task(inputs, tokenizer, False)
    elif token_wise_task_qa == True:
        keep_fields = ['o_input_ids', 'o_attention_mask','o_context_start_positions','o_start_positions','o_end_positions','o_questions','o_dialogue_ids','o_sentences','o_offset_mapping']
        pad_and_align_for_token_wise_task(inputs, tokenizer, True)

    for i in range(len(inputs['input_ids'])):
        for k,v in inputs.items():
            if k != 'sentences' and 'o_' not in k: # remove CLS and SEP
                inputs[k][i] = inputs[k][i][1:-1]

        if len(inputs['input_ids'][i]) > max_token_len:
            inputs['input_ids'][i] = inputs['input_ids'][i][:max_token_len]
        else:
            extra_token_num = len(inputs['input_ids'][i]) % chunk_len
            if extra_token_num > 0:
                pad_num = chunk_len - extra_token_num
                inputs['input_ids'][i].extend([tokenizer.pad_token_id]*pad_num)
                inputs['attention_mask'][i].extend([0]*pad_num)
                inputs['token_type_ids'][i].extend([0]*pad_num)
                inputs['special_tokens_mask'][i].extend([1]*pad_num)
                inputs['offset_mapping'][i].extend([[0,0]]*pad_num)
                inputs['kp_token_type_ids'][i].extend([-1]*pad_num)
                inputs['kp_token_scores'][i].extend([0]*pad_num)
        if len(inputs['input_ids'][i]) % attn_mode['param1'] !=0:
                print(len(inputs['input_ids'][i]),chunk_len)
                print(extra_token_num,pad_num)
                import sys
                sys.exit()
        if attn_mode['param2'] == 'fixed_weights':
            weights_cfg = attn_mode['param3'] #[non key phrase token weight, key phrase token weight]
            weights_cfg = [float(c) for c in weights_cfg.split('_')]
            one_kp_token_weights = []
            for type_id in inputs['kp_token_type_ids'][i]:
                if type_id == 0: # non kp tokens
                    one_kp_token_weights.append(weights_cfg[0])
                elif type_id == 1: # kp tokens
                    one_kp_token_weights.append(weights_cfg[1])
                elif type_id == -1: # pad tokens
                    one_kp_token_weights.append(0)
                else:
                    assert False, 'assign weights error'
            all_kp_token_weights.append(one_kp_token_weights)
            # print("one_kp_token_weights ",one_kp_token_weights[0])
        elif attn_mode['param2'] == 'average_weights':
            all_kp_token_weights.append([1]*len(inputs['kp_token_type_ids'][i]))
        elif attn_mode['param2'] == 'ori_scores_1':
            # higher score, higher weight
            # the non-key-phrase tokens'weight = exp(min score)
            # all scores are lower than 0
            np_kp_scores = np.array(inputs['kp_token_scores'][i])
            np_kp_token_type_ids =np.array(inputs['kp_token_type_ids'][i])
            assert np.max(np_kp_scores) <= 0, print("error kp score!!!!")
            min_score = np.min(np_kp_scores)
            # print("np_kp_token_type_ids ",min_score, np_kp_token_type_ids)
            # print("np_kp_scores0000 ",np_kp_scores)
            np_kp_scores[np_kp_token_type_ids==0] = min_score
            # print("np_kp_scores1111 ",np_kp_scores)
            np_kp_scores[np_kp_token_type_ids!=-1] = np.exp(1-np_kp_scores/min_score)[np_kp_token_type_ids!=-1]
            # print("np_kp_scores ",np_kp_scores)
            # print("np_kp_scores2222 ",np_kp_scores)
            all_kp_token_weights.append(np_kp_scores.tolist())



        # re-batch
        for j in range(0,len(inputs['input_ids'][i]),chunk_len):
            new_inputs['sentences'].append(inputs['sentences'][i])
            new_inputs['input_ids'].append(inputs['input_ids'][i][j:j+chunk_len])
            new_inputs['attention_mask'].append(inputs['attention_mask'][i][j:j+chunk_len])
            new_inputs['token_type_ids'].append(inputs['token_type_ids'][i][j:j+chunk_len])
            new_inputs['special_tokens_mask'].append(inputs['special_tokens_mask'][i][j:j+chunk_len])
            new_inputs['offset_mapping'].append(inputs['offset_mapping'][i][j:j+chunk_len])
            new_inputs['kp_token_type_ids'].append(inputs['kp_token_type_ids'][i][j:j+chunk_len])
            new_inputs['kp_token_scores'].append(inputs['kp_token_scores'][i][j:j+chunk_len])
            b = all_kp_token_weights[i][j:j+chunk_len]
            # if torch.tensor(b).sum() == 0:
            #     print("all_kp_token_weights00000 ",all_kp_token_weights[i])
            #     print("torch.all_kp_token_weights00000 chunk ",b)
            #     print("inputs['kp_token_scores'][i] ",inputs['kp_token_scores'][i])
            #     print("kp_token_type_ids ",inputs['kp_token_type_ids'][i])
            #     print("kp_token_type_ids chunk ",inputs['kp_token_type_ids'][i][j:j+chunk_len])
            new_inputs['kp_token_weights'].append(all_kp_token_weights[i][j:j+chunk_len])
            new_inputs['map_ids'].append(i)
    # print("end pad_and_align_for_kpcr")
    if token_wise_task == True or token_wise_task_qa == True:
        for k in keep_fields:
            new_inputs[k] = inputs[k]
    return new_inputs

def move_the_fisrt_some_tokens_to_the_end(inputs,max_len=512):
    for i in range(len(inputs['input_ids'])):
        if len(inputs['input_ids'][i]) > max_len:
            for k,v in inputs.items():
                if k != 'sentences': # remove CLS and SEP
                    inputs[k][i] = [inputs[k][i][0]] + inputs[k][i][1 + max_len:-1] + inputs[k][i][1:1+max_len] + [inputs[k][i][-1]]
    return  inputs

def insert_random_tokens_randomly(inputs,max_len,attn_mode,tokenizer,cached_tokenized_results):
    pass

def insert_random_tokens_randomly_surrond(inputs,ds_object,attn_mode,idxs):
    assert 'param1' in attn_mode, "only support the one has param1"
    real_max_len = ds_object.max_seq_len*attn_mode['param1']
    expended_cache = {}
    if hasattr(ds_object,'expended_cache'):
        expended_cache = ds_object.expended_cache
    else:
        setattr(ds_object,'expended_cache',expended_cache)
    for idx, one_input in zip(idxs,inputs):
        if expended_cache.get(idx, None):
            insert_way = random.randint(0,2)
            if insert_way == 0: # Insert in front of
                pass
            elif insert_way == 1: # Insert behind
                pass
            elif insert_way ==2: # insert surrond
                pass

def collate_fn_bert(ds_object,examples,not_label = False, tokenizer = None,da_object=None):
    # print("start collate_fn_bert 1")
    targets = []
    sentences = []
    chunked_sentences = []
    chunked_sents = []
    wordpos2chunks = []
    max_chunks = 0
    # print('collate_fn_non_bert', ds_object.cache_tokenize.keys())
    idxs = []
    attn_mode = getattr(ds_object,'attn_mode',{'name':'default','param1':0})
    sub_task = attn_mode.get('sub_task',None)
    tokenizer = tokenizer if tokenizer is not None else ds_object.tokenizer
    token_wise_task = getattr(ds_object, "token_wise_task", None)
    token_wise_task_qa = getattr(ds_object, "token_wise_task_qa", None)
    # print("getattr(ds_object,'cached_tokenized_results',False) ",getattr(ds_object,'cached_tokenized_results',False)==False)
    if getattr(ds_object,'cached_tokenized_results',False) and attn_mode['name'] not in ("key_phrase_split",'key_phrase_split2'):
        cached_tokenized_results = ds_object.cached_tokenized_results
        if attn_mode.get('sent_emb',None) == 'all':
            inputs = {'input_ids':[],'attention_mask':[],'token_type_ids':[],'special_tokens_mask':[],'offset_mapping':[],'sentences':[],'textrank_data':[]}
        else:
            inputs = {'input_ids':[],'attention_mask':[],'token_type_ids':[],'special_tokens_mask':[],'offset_mapping':[],'sentences':[]}
        # if attn_mode['name'] == 'key_phrase_split':
        #     inputs.update({'kps':[],'kp_scores':[],'max_chunk_len':[]})
        if attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            inputs.update({'kp_token_scores':[],'kp_token_type_ids':[]})
            #no need to align the inputs again because we preprocess all of the examples in one batch
        # print("cached_tokenized_results ",cached_tokenized_results[0].keys())
        if token_wise_task == True:
            inputs.update({'o_input_ids':[],'o_attention_mask':[],'o_token_labels':[]})
            for sentence,label, word_list, idx in examples:
                idxs.append(idx)
                targets.append(label)
                sentences.append(sentence)
                for k,v in inputs.items():
                    # print(cached_tokenized_results[idx].keys())
                    v.append(cached_tokenized_results[idx][k])
        elif token_wise_task_qa == True:
            inputs.update({'o_input_ids':[], 'o_attention_mask':[],'o_context_start_positions':[],'o_start_positions':[],'o_end_positions':[],'o_offset_mapping':[]})

            all_questions = []
            all_dialogue_ids = []
            # print("cached_tokenized_results ",cached_tokenized_results[0].keys(), inputs.keys())
            for sentence, question, ans_list, dislogue_id, idx in examples:
                idxs.append(idx)
                targets.append(ans_list)
                sentences.append(sentence)
                all_questions.append(question)
                all_dialogue_ids.append(dislogue_id)

                for k,v in inputs.items():
                    # print(cached_tokenized_results[idx].keys())
                    v.append(cached_tokenized_results[idx][k])
            inputs['o_questions'] = all_questions
            inputs['o_dialogue_ids'] = all_dialogue_ids
            inputs['o_sentences'] = sentences
        else:
            for sentence,label,idx in examples:
                # multilabel case,drop the sample of which the target is [], like Booksummary dataset
                if hasattr(label,'__len__') and len(label) > 0:
                    idxs.append(idx)
                    targets.append(label)
                    sentences.append(sentence)
                    for k,v in inputs.items():
                        v.append(cached_tokenized_results[idx][k])
                else:
                    idxs.append(idx)
                    targets.append(label)
                    sentences.append(sentence)
                    for k,v in inputs.items():
                        # print(cached_tokenized_results[idx].keys())
                        v.append(cached_tokenized_results[idx][k])
        # print("before transform, targets[0] ",targets[0])
        # print("can go here  ? ",sub_task)
        if sub_task == 'move_the_fisrt_some_tokens_to_the_end':
            # don't use, don't make sense
            # print("move_the_fisrt_some_tokens_to_the_end ===========")
            inputs = move_the_fisrt_some_tokens_to_the_end(inputs,ds_object.max_seq_len)
            # print("=========")
            # print(tokenizer.decode(inputs['input_ids'][0]))
            # print('\n\n')
            # print(sentences[0])
        elif sub_task == 'insert_random_tokens_in_front_of':
            pass
        elif sub_task == 'insert_random_tokens_behind':
            pass
        elif sub_task == 'insert_random_tokens_randomly':
            inputs = insert_random_tokens_randomly(inputs,ds_object.max_seq_len,attn_mode,ds_object.tokenizer,cached_tokenized_results)
        elif sub_task == 'insert_random_tokens_randomly_surrond':
            inputs = insert_random_tokens_randomly_surrond(inputs,ds_object,attn_mode,idxs)
        if attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            # print(inputs.keys())
            # import sys
            # sys.exit()
            if attn_mode.get('sent_emb', None) == 'all':
                inputs = pad_and_align_for_kpcr_with_sentemb(inputs,ds_object.max_seq_len,attn_mode,ds_object.tokenizer,token_wise_task, token_wise_task_qa)
            else:
                inputs = pad_and_align_for_kpcr(inputs,ds_object.max_seq_len,attn_mode,ds_object.tokenizer,token_wise_task, token_wise_task_qa)
            # print("???????pad and align")
        # print("start collate_fn_bert 2")
        try:
            inputs['input_ids'] = torch.tensor(inputs['input_ids'])
        except:
            # print(inputs['input_ids'])
            print(attn_mode['name'])
            for i in inputs['input_ids']:
                print(len(i))
            assert False, "Error when use tokenized cache data"
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
        inputs['special_tokens_mask'] = torch.tensor(inputs['special_tokens_mask'])
        inputs['offset_mapping'] = torch.tensor(inputs['offset_mapping'])
        if token_wise_task == True:
            if attn_mode['name'] == 'default': # for lonformer case
                pad_and_align_for_token_wise_task(inputs, tokenizer, False)
            inputs['o_input_ids'] = torch.tensor(inputs['o_input_ids'])
            inputs['o_attention_mask'] = torch.tensor(inputs['o_attention_mask'])
            inputs['o_token_labels'] = torch.tensor(inputs['o_token_labels'])
        elif token_wise_task_qa == True:
            inputs['o_input_ids'] = torch.tensor(inputs['o_input_ids'])
            inputs['o_attention_mask'] = torch.tensor(inputs['o_attention_mask'])
            inputs['o_context_start_positions'] = torch.tensor(inputs['o_context_start_positions'])
            inputs['o_start_positions'] = torch.tensor(inputs['o_start_positions'])
            inputs['o_end_positions'] = torch.tensor(inputs['o_end_positions'])
            inputs['o_offset_mapping'] = inputs['o_offset_mapping']

        if attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            inputs['kp_token_type_ids'] = torch.tensor(inputs['kp_token_type_ids'])
            inputs['kp_token_scores'] = torch.tensor(inputs['kp_token_scores'])
            inputs['kp_token_weights'] = torch.tensor(inputs['kp_token_weights'])
            inputs['map_ids'] = torch.tensor(inputs['map_ids'])
    
        sentences = inputs['sentences']
    else:
        # because we cache the tokenization results, so we need to set the 'padding' to 'max_length' to simplify the usage of cache.
        # It can be optimized in the future
        if token_wise_task == True:
            word_lists = []
            # need to add o_token_labels as the above
            for sentence,label, word_list, idx in examples:
                idxs.append(idx)
                targets.append(label)
                sentences.append(sentence)
                word_lists.append(word_list)
            
            inputs_for_token_wise_task = tokenize_and_align_labels(tokenizer, word_lists, targets, da_object.label_encoder)
        elif token_wise_task_qa == True:
            all_questions = []
            all_dialogue_ids = []
            for sentence, question, ans_list, dislogue_id, idx in examples:
                idxs.append(idx)
                targets.append(ans_list)
                sentences.append(sentence)
                all_questions.append(question)
                all_dialogue_ids.append(dislogue_id)
                
            inputs_for_token_wise_task_qa = preprocess_function(tokenizer,all_questions,sentences,targets)
            inputs_for_token_wise_task_qa = get_qatask_decoder_input(inputs_for_token_wise_task_qa)
        else:
            for sentence,label,idx in examples:
                idxs.append(idx)
                targets.append(label)
                sentences.append(' '.join(sentence.split(' ')[:ds_object.max_seq_len]))
        # print('attn_mode',attn_mode)
        if attn_mode['name'] == 'default':
            # normal full attention
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
        elif attn_mode['name'] == 'default_without_chunk_and_pad':
            # normal full attention
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=False,
                    padding=False,
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
        elif attn_mode['name'] == 'fixed_token_length_wo_loc':
            # fixed token length without LOC token
            # only used with param2=all_imps
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
            input_ids, attention_mask, token_type_ids, special_tokens_mask, offset_mapping = inputs['input_ids'].tolist(), \
                inputs['attention_mask'].tolist(), inputs['token_type_ids'].tolist(), inputs['special_tokens_mask'].tolist(), \
                    inputs['offset_mapping'].tolist()
            last_block_len = (len(input_ids[0]) - 2) % attn_mode['param1'] # minus 2 because of [CLS] and [SEP]
            # Because we sue padding='max_length', so we normally don't need to pad the last block, e.g., 
            # the max_length is 512
            if last_block_len != 0:
                for i in range(len(input_ids)):
                    for index in range(len(input_ids[i]) - 1,len(input_ids[i]) - 1 + attn_mode['param1']-last_block_len):
                        input_ids[i].insert(index,tokenizer.get_vocab()['[PAD]'])
                        attention_mask[i].insert(index,0)
                        token_type_ids[i].insert(index,0)
                        special_tokens_mask[i].insert(index,1) 
                        offset_mapping[i].insert(index,[0,0])
            inputs['input_ids'] = torch.tensor(input_ids)
            inputs['attention_mask'] = torch.tensor(attention_mask)
            inputs['token_type_ids'] = torch.tensor(token_type_ids)
            inputs['special_tokens_mask'] = torch.tensor(special_tokens_mask)
            inputs['offset_mapping'] = torch.tensor(offset_mapping)
        elif attn_mode['name'] == 'fixed_token_length':
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
            input_ids, attention_mask, token_type_ids, special_tokens_mask, offset_mapping = inputs['input_ids'].tolist(), \
                inputs['attention_mask'].tolist(), inputs['token_type_ids'].tolist(), inputs['special_tokens_mask'].tolist(), \
                    inputs['offset_mapping'].tolist()
            # print('insert_spe_tokens ',insert_spe_tokens,last_block_len)
            max_insert_spe_tokens = (ds_object.max_seq_len - 2) // (attn_mode['param1'] - 1)
            real_max_len = (max_insert_spe_tokens+1) * attn_mode['param1'] + 2
            for i in range(len(input_ids)):
                sep_len = sum(attention_mask[i])
                insert_spe_tokens = (sep_len - 2) // (attn_mode['param1'] - 1)
                last_block_len = (sep_len - 2) % (attn_mode['param1'] - 1)
                for index in range(1, sep_len - 1 +insert_spe_tokens, attn_mode['param1']):
                    input_ids[i].insert(index,tokenizer.get_vocab()['[LOC]'])
                    attention_mask[i].insert(index,attention_mask[i][index])
                    token_type_ids[i].insert(index,0)
                    special_tokens_mask[i].insert(index,1)
                    offset_mapping[i].insert(index,[0,0])
                # pad tokens in the last local block
                # print('len(input_ids[i])',len(input_ids[i]))
                add_last = 0
                if last_block_len != 0:
                    for index in range(sep_len + insert_spe_tokens,(insert_spe_tokens+1)*attn_mode['param1'] + 1):
                        input_ids[i].insert(index,tokenizer.get_vocab()['[PAD]'])
                        attention_mask[i].insert(index,0)
                        token_type_ids[i].insert(index,0)
                        special_tokens_mask[i].insert(index,1) 
                        offset_mapping[i].insert(index,[0,0])
                        add_last += 1
                for index in range(sep_len + insert_spe_tokens + add_last + 1, real_max_len, attn_mode['param1']):
                    input_ids[i].insert(index,tokenizer.get_vocab()['[LOC]'])
                    attention_mask[i].insert(index,attention_mask[i][index])
                    token_type_ids[i].insert(index,0)
                    special_tokens_mask[i].insert(index,1)
                    offset_mapping[i].insert(index,[0,0])
                # final padding
                for index in range(len(input_ids[i]),real_max_len):
                    input_ids[i].insert(index,tokenizer.get_vocab()['[PAD]'])
                    attention_mask[i].insert(index,0)
                    token_type_ids[i].insert(index,0)
                    special_tokens_mask[i].insert(index,1) 
                    offset_mapping[i].insert(index,[0,0])
            inputs['input_ids'] = torch.tensor(input_ids)
            inputs['attention_mask'] = torch.tensor(attention_mask)
            inputs['token_type_ids'] = torch.tensor(token_type_ids)
            inputs['special_tokens_mask'] = torch.tensor(special_tokens_mask)
            inputs['offset_mapping'] = torch.tensor(offset_mapping)
        elif attn_mode['name'] == 'sentence_split':
            for i in range(len(sentences)):
                sentence_list = nltk.sent_tokenize(sentences[i])
                sentences[i] = ' [LOC] '.join(sentence_list)
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
        elif attn_mode['name'] == 'constituency_parsing':
            for i in range(len(sentences)):
                one_chunked_sentence,tree,ori_chunks = chunk_sentence(sentences[i],['sbar', 'np', 'vp', 'pp', 'adjp', 'advp']) 
                processed_s = '[LOC]'
                for chunk in one_chunked_sentence:
                    processed_s = processed_s + ' ' + ' '.join(chunk)
                sentences[i] = processed_s   
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
            # need to pad each chunk for alignment
            LOC_ID = tokenizer.get_vocab()['[LOC]']
            PAD_ID = tokenizer.get_vocab()['[PAD]']
            max_chunk_len = 0
            for ids in input_ids:
                last_loc_pos = 0
                for i in range(len(ids)):
                    token_id = ids[i]
                    if token_id == LOC_ID:
                        max_chunk_len = max(i - last_loc_pos,max_chunk_len)
                        last_loc_pos = i
                max_chunk_len = max(max_chunk_len,len(ids) - last_loc_pos)

            for i in range(len(input_ids)):
                last_loc_index = 1
                for index in range(2,len(input_ids[i])):
                    if input_ids[i][index] == LOC_ID:
                        cur_loc_len = index - last_loc_index
                        last_loc_index = index
                        input_ids[i] = input_ids[i][:index] + [PAD_ID]*(max_chunk_len - cur_loc_len) + input_ids[i][:index]
                        input_ids[i].insert(index,tokenizer.get_vocab()['[LOC]'])
                        attention_mask[i].insert(index,attention_mask[i][index])
                        token_type_ids[i].insert(index,0)
                        special_tokens_mask[i].insert(index,1)
                        offset_mapping[i].insert(index,[0,0])
                # pad tokens in the last local block
                # print('len(input_ids[i])',len(input_ids[i]))
                if last_block_len != 0:
                    for index in range(len(input_ids[i]) - 1,(insert_spe_tokens+1)*attn_mode['param1'] + 1):
                        input_ids[i].insert(index,tokenizer.get_vocab()['[PAD]'])
                        attention_mask[i].insert(index,0)
                        token_type_ids[i].insert(index,0)
                        special_tokens_mask[i].insert(index,0) # avoid attention computation in the model, actually it should be 1
                        offset_mapping[i].insert(index,[0,0])
                # print('len(input_ids[i])',len(input_ids[i]))

        elif attn_mode['name'] in ('key_phrase_split','key_phrase_split_wo_loc','key_phrase_split2'): 
            assert attn_mode['name'] != 'key_phrase_split_wo_loc', 'Don\'t support without loc'
            global chunk_statistics
            add_loc = 0
            max_chunk_num = 0
            chunk_num_stat = []
            if getattr(ds_object,'cached_tokenized_results',False):
                all_kp = []
                all_score = []
                chunk_num = 0
                for i in range(len(sentences)):
                    index = idxs[i]
                    all_kp.append(ds_object.cached_tokenized_results['all_kp'][index])
                    all_score.append(ds_object.cached_tokenized_results['all_score'][index])
                    sentences[i] = ds_object.cached_tokenized_results['all_sentence'][index]
                    chunk_num =  ds_object.cached_tokenized_results['chunk_num_stastics'][index]
                    # print("chunk num: ",chunk_num)
                    chunk_num_stat.append(chunk_num)
                    max_chunk_num = max(max_chunk_num,chunk_num)
                    add_loc += chunk_num
            else:
                assert False, "Should do this in preprocess"
                pass
                # all_kp,all_score = get_key_phrases(sentences,ds_object.file_path,max_len=ds_object.max_seq_len)
                # for i in range(len(sentences)):
                #     kps = all_kp[i]
                #     sentences[i] = ' '.join(sentences[i].split(' ')[:ds_object.max_seq_len])
                #     sentences[i] = '[LOC] '+sentences[i]
                #     for kp in kps:
                #         sentences[i] = sentences[i].replace(kp, kp+' [LOC]')
                #     chunk_num = sentences[i].count('[LOC]')
                #     add_loc += chunk_num
                #     # print("chunk num: ",chunk_num)
                #     max_chunk_num = max(max_chunk_num,chunk_num)
            # because in the fixed_token_length we insert [LOC] after truncating to 512
            fair_compare = add_loc//len(sentences)
            inputs = tokenizer(
                    sentences,
                    max_length=ds_object.max_seq_len + fair_compare,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask = True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask = True,
                    return_token_type_ids = True,
                    return_tensors = 'pt',
                )
            input_ids, attention_mask, token_type_ids, special_tokens_mask, offset_mapping = inputs['input_ids'].tolist(), \
                inputs['attention_mask'].tolist(), inputs['token_type_ids'].tolist(), inputs['special_tokens_mask'].tolist(), \
                    inputs['offset_mapping'].tolist()
            # need to pad each chunk for alignment
            LOC_ID = tokenizer.get_vocab()['[LOC]']
            PAD_ID = tokenizer.get_vocab()['[PAD]']
            SEP_ID = tokenizer.get_vocab()['[SEP]']
            # print("LOC_ID: {}, PAD_ID: {}".format(LOC_ID,PAD_ID))
            max_chunk_len = 0
            chunk_len_stat = []
            for ids in input_ids:
                last_loc_pos = 0
                local_max_chunk_len = 0
                for i in range(len(ids)):
                    token_id = ids[i]
                    if token_id == LOC_ID or token_id == SEP_ID:
                        chunk_len = i - last_loc_pos
                        local_max_chunk_len = max(chunk_len,local_max_chunk_len)
                        max_chunk_len = max(chunk_len,max_chunk_len)
                        last_loc_pos = i
                chunk_len_stat.append(local_max_chunk_len)
                # max_chunk_len = max(max_chunk_len,len(ids) - last_loc_pos)
            print("(chunk num,max chunk len): ",*zip(chunk_num_stat,chunk_len_stat))
            token_len_stat = []
            for chunk_num, chunk_len in zip(chunk_num_stat,chunk_len_stat):
                token_len_stat.append(chunk_num*chunk_len + 2)
            print("token len stat: ", pd.DataFrame(token_len_stat).describe())
            print("global max_chunk_num: {}, max_chunk_len: {} ".format(max_chunk_num,max_chunk_len))
            # print("real max length: ",max_chunk_len*max_chunk_num+2)
            for i in range(len(input_ids)):
                # print("--------------------------")
                # print("Before: ",len(input_ids[i]))
                # print(input_ids[i])
                last_loc_index = 1
                iid,am,tti,stm,om = input_ids[i][:last_loc_index],attention_mask[i][:last_loc_index],\
                    token_type_ids[i][:last_loc_index],special_tokens_mask[i][:last_loc_index],offset_mapping[i][:last_loc_index]
                # print("original len ",len(input_ids[i]),fair_compare,max_chunk_num,max_chunk_len)
                # print("original iid ",input_ids[i])
                for index in range(2,len(input_ids[i])): #[CLS][LOC][W1]...
                    if input_ids[i][index] in (LOC_ID, SEP_ID):
                        cur_loc_len = index - last_loc_index
                        pad_len = max_chunk_len - cur_loc_len
                        iid += input_ids[i][last_loc_index:index] + [PAD_ID]*(pad_len)
                        am += attention_mask[i][last_loc_index:index] + [0]*(pad_len)
                        tti += token_type_ids[i][last_loc_index:index] + [0]*(pad_len)
                        stm += special_tokens_mask[i][last_loc_index:index] + [1]*(pad_len)
                        om += offset_mapping[i][last_loc_index:index] + [[0,0]]*(pad_len)
                        last_loc_index = index
                pad_len = max_chunk_len*max_chunk_num+2 - len(iid) - 1
                # print("before pad ",len(iid),pad_len)
                # print("len(iid): {}, last pad len: {}".format(len(iid), pad_len))
                iid += ([input_ids[i][last_loc_index]] + [PAD_ID]*(pad_len))
                am += ([attention_mask[i][last_loc_index]] + [0]*(pad_len))
                tti += ([token_type_ids[i][last_loc_index]] + [0]*(pad_len))
                stm += ([special_tokens_mask[i][last_loc_index]] + [1]*(pad_len))
                om += ([offset_mapping[i][last_loc_index]] + [[0,0]]*(pad_len))
                # print("len(iid) ",LOC_ID,PAD_ID,SEP_ID,len(iid),iid)
                # print("am",am)
                # print("am reshape",torch.tensor(am)[1:-1].reshape(-1,max_chunk_len))
                input_ids[i] = iid
                attention_mask[i] = am
                token_type_ids[i] = tti
                special_tokens_mask[i] = stm
                offset_mapping[i] = om
                # print('len(input_ids[i])',len(input_ids[i]))
            # import sys
            # sys.exit()
                chunk_statistics.append(len(iid))
            inputs['input_ids'] = torch.tensor(input_ids)
            inputs['attention_mask'] = torch.tensor(attention_mask)
            inputs['token_type_ids'] = torch.tensor(token_type_ids)
            inputs['special_tokens_mask'] = torch.tensor(special_tokens_mask)
            inputs['offset_mapping'] = torch.tensor(offset_mapping)
            inputs['kps'] = all_kp
            inputs['kp_scores'] = all_score 
            inputs['max_chunk_len'] = max_chunk_len
        elif attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            pass
        if token_wise_task == True:
            inputs.update(inputs_for_token_wise_task)
        elif token_wise_task_qa == True:
            inputs.update(inputs_for_token_wise_task_qa)
            inputs['o_questions'] = all_questions
            inputs['o_dialogue_ids'] = all_dialogue_ids
            inputs['o_sentences'] = sentences
    if not_label == False:
        # print("targets 1 ",len(targets))
        if token_wise_task == True:
            tmp_targets = []
            for token_labels in targets:
                tmp_targets.append(ds_object.label_encoder.transform(token_labels))
            targets = tmp_targets
        elif token_wise_task_qa == True:
            pass
        else:
            targets = ds_object.label_encoder.transform(targets)
        # print("after transform, targets[0] ",targets[0])
            targets = torch.tensor(targets)
        # print("targets 2 ",targets.shape)
    batch = {
        'targets': targets,
        'sentences': sentences,
    }

    batch.update(inputs)

    if attn_mode['name'] not in ['default','fixed_token_length','fixed_token_length_wo_loc']:
        # In this case, we need to align the local blocks
        # TODO
        # batch['offset_mapping'] = inputs['offset_mapping']
        pass
    # print(batch['targets'])
    # print("start collate_fn_bert 3")
    return batch

def extract_candidates(tokens_tagged):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate

def collate_seperate_pair_fn_bert(ds_object,examples):
    targets = []
    ids1 = []
    ids2 = []
    sentences1 = []
    sentences2 = []
    # print('collate_fn_non_bert', ds_object.cache_tokenize.keys())
    count_label = {0:0,1:0}
    for sample,label,idx in examples:
        count_label[label] += 1
        targets.append(1-label)
        sentences1.append(sample['s1'])
        sentences2.append(sample['s2'])
        # some datasets may don't have id
        ids1.append(sample.get('id1',len(ids1)))
        ids2.append(sample.get('id2',len(ids2)))

    inputs1 = {}
    inputs1.update(ds_object.tokenizer(
            sentences1,
            max_length=ds_object.max_seq_len,
            truncation=True,
            padding=True,
            return_attention_mask = True,
            return_offsets_mapping=True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            return_tensors = 'pt',
        ))
    inputs2 = {}
    inputs2.update(ds_object.tokenizer(
            sentences2,
            max_length=ds_object.max_seq_len,
            truncation=True,
            padding=True,
            return_attention_mask = True,
            return_offsets_mapping=True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            return_tensors = 'pt',
        ))
    # print( sentences1)
    # print( sentences2)
    # print(inputs)
    targets = ds_object.label_encoder.transform(targets)
    # print(torch.tensor(targets).sum())
    batch = {
        'targets': torch.tensor(targets),
        'sentences1': sentences1,
        'sentences2': sentences2,
        'ids1': ids1,
        'ids2': ids2,
        'inputs1':inputs1,
        'inputs2':inputs2,
    }
    # print(batch['targets'])
    return batch

def collate_pair_fn_bert(ds_object,examples):
    targets = []
    ids1 = []
    ids2 = []
    sentences1 = []
    sentences2 = []
    # print('collate_fn_non_bert', ds_object.cache_tokenize.keys())
    for sample,label,idx in examples:
        targets.append(label)
        sentences1.append(sample['s1'])
        sentences2.append(sample['s2'])
        # some datasets may don't have id
        ids1.append(sample.get('id1',len(ids1)))
        ids2.append(sample.get('id2',len(ids2)))

    inputs = ds_object.tokenizer(
            sentences1,
            sentences2,
            max_length=ds_object.max_seq_len,
            truncation="only_second",
            padding=True,
            return_attention_mask = True,
            return_offsets_mapping=True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            return_tensors = 'pt',
        )
    # print( sentences1)
    # print( sentences2)
    # print(inputs)
    targets = ds_object.label_encoder.transform(targets)
    # print(torch.tensor(targets).sum())
    batch = {
        'targets': torch.tensor(targets),
        'sentences1': sentences1,
        'sentences2': sentences2,
        'ids1': ids1,
        'ids2': ids2,
    }
    batch.update(inputs)
    # print(batch['targets'])
    return batch

def collate_seperate_pair_fn_non_bert(ds_object,examples):
    pass


'''
The following methods can only help with customized method. The pytorch-lightning method can't be
moved here
'''

# def check_and_load_chunked_sentences(da_object):
#     file_path = os.path.join(da_object.data_path,f'chunking_data_{da_object.tokenizer_name}_ver_{da_object.chunking}.pt')
#     if os.path.exists(file_path):
#         print(f'Loading chunked sentences... chunking_data_{da_object.tokenizer_name}_ver_{da_object.chunking}.pt')
#         save_data = torch.load(file_path)
#         save_data = {one_row['sentence']:one_row for one_row in save_data}
#         return save_data
#     else:
#         return None

def construct_chunk_vocab(da_object,loaded_chunk_data):
    chunk_vocab = {"[PAD]":0,"[CLS]":1,"[UNK]":2}
    c = 3
    for sentence, chunk_data in loaded_chunk_data.items():
        for chunk in chunk_data['final_chunked_sentence']:
            chunk_key = " ".join(chunk)
            if chunk_key not in chunk_vocab:
                chunk_vocab[chunk_key] = len(chunk_vocab)
                c += 1
    '''
    use copy.deepcopy for some unk bugs.
    '''
    setattr(da_object,"chunk_vocab",chunk_vocab)
    for split,dataset in da_object.datasets.items():
        setattr(dataset,"chunk_vocab",chunk_vocab)
    print("construct_chunk_vocab ",c)
   

def preprocess(da_object):
    # init g_parser
    # if textprocesser.g_parser is None:
    #     textprocesser.g_parser = ConstituencyParser(da_object.parser_type)
    all_labels = []
    all_sentences = []
    all_lessprocessed_sentences = {}
    da_object.processer = get_preprocesser(da_object.preprocesser_cfg)
    print("da_object.processer ",da_object.processer,da_object.preprocesser_cfg)
    attn_mode = getattr(da_object,'attn_mode',{'name':'default','param1':0})
    token_wise_task = getattr(da_object, "token_wise_task", None)
    token_wise_task_qa = getattr(da_object, "token_wise_task_qa", None)
    if token_wise_task== True:
        # in this case, we need labelencoder first
        for split,dataset in da_object.datasets.items():
             all_labels.extend(dataset.labels)
        # print("all_labels", all_labels)
        all_labels = [token_label for labels in all_labels for token_label in labels]
        set_all_labels = [token_one_label for token_labels in all_labels for token_one_label in token_labels]
        print("all_labels ",set(set_all_labels))
        da_object.label_encoder.fit(all_labels)
        da_object.nclasses = len(da_object.label_encoder.classes_)
        all_labels = []

    for split,dataset in da_object.datasets.items():
        all_labels.extend(dataset.labels)
        dataset.data = [da_object.processer.process_one(x) for x in dataset.data]

        tokenizer_name = da_object.tokenizer_type + '_' + da_object.tokenizer_name.replace('/','_')
        # only use berkeley parser need pre-truncatation
        if attn_mode['name'] in ('fixed_token_length','fixed_token_length_wo_loc'):
            name = attn_mode['name'] + '_' + str(attn_mode['param1'])
        elif attn_mode['name'] in ('key_phrase_split','key_phrase_split2'):
            name = attn_mode['name']
        elif attn_mode['name'] == 'default':
            name = attn_mode['name']
        elif attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            name = attn_mode['name']

        if attn_mode['name'] in ('key_phrase_split','key_phrase_split2'):
            cache_file = 'results/cache/key_phrase_split/{}_{}_{}_{}_{}_kps.pt'.format(\
                da_object.global_config['DATA']['dataset_name'],name,split,tokenizer_name,da_object.max_seq_len)
        elif attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            # cache_file = 'results/cache/tokenized_results/{}_{}_{}_{}_{}.pt'.format(\
            cache_file = 'results/cache/tokenized_results/{}_{}_{}_{}_{}_top10.pt'.format(\
                da_object.global_config['DATA']['dataset_name'],name,split,tokenizer_name,'whole_doc')
            # 512 means nothing in this case, just the name 
            # print("cache_file  ",cache_file)
        else:
            cache_file = 'results/cache/tokenized_results/{}_{}_{}_{}_{}.pt'.format(da_object.global_config['DATA']\
                                                    ['dataset_name'],name,split,tokenizer_name,da_object.max_seq_len)
        if os.path.exists(cache_file):
            print("tokenized data exists, loading ",cache_file)
            setattr(dataset,"cached_tokenized_results",torch.load(cache_file))
        elif da_object.pre_cache == True:
            print("tokenized data doesn't exists ",cache_file)
            all_samples = [one_sample for one_sample in dataset]
            batch = collate_fn_bert(dataset,all_samples,not_label=True,tokenizer=da_object.tokenizer,da_object=da_object)
            catched_tokenized_results = []
            
            for i in range(len(all_samples)):
                save_data = {
                    'sentences':batch['sentences'][i],
                    'input_ids':batch['input_ids'][i].tolist(),
                    'attention_mask':batch['attention_mask'][i].tolist(),
                    'offset_mapping':batch['offset_mapping'][i].tolist(),
                    'special_tokens_mask':batch['special_tokens_mask'][i].tolist(),
                    'token_type_ids':batch['token_type_ids'][i].tolist(),
                }
                if attn_mode['name'] in ('key_phrase_split','key_phrase_split2'):
                    save_data['kps'] = batch['kps'][i]
                    save_data['kp_scores'] = batch['kp_scores'][i]
                if token_wise_task == True:
                    save_data['o_input_ids'] = batch['o_input_ids'][i]
                    save_data['o_attention_mask'] = batch['o_attention_mask'][i]
                    save_data['o_token_labels'] = batch['o_token_labels'][i]
                if token_wise_task_qa == True:
                    save_data['o_input_ids'] = batch['o_input_ids'][i]
                    save_data['o_attention_mask'] = batch['o_attention_mask'][i]
                    save_data['o_context_start_positions'] = batch['o_context_start_positions'][i]
                    save_data['o_start_positions'] = batch['o_start_positions'][i]
                    save_data['o_end_positions'] = batch['o_end_positions'][i]
                    save_data['o_questions'] = batch['o_questions']
                    save_data['o_dialogue_ids'] = batch['o_dialogue_ids']
                    save_data['o_sentences'] = batch['o_sentences']
                    save_data['o_offset_mapping'] = batch['o_offset_mapping']
                    
                catched_tokenized_results.append(save_data)

            torch.save(catched_tokenized_results,cache_file)
            setattr(dataset,"cached_tokenized_results",catched_tokenized_results)
            print("tokenized data saved")
        else:
            print("pre_cache is False, don't compute the cache data")
    
        all_sentences.extend(dataset.data)

    if token_wise_task== True:
        all_labels = [token_label for labels in all_labels for token_label in labels]
        set_all_labels = [token_one_label for token_labels in all_labels for token_one_label in token_labels]
        print("all_labels ",set(set_all_labels))
    if token_wise_task_qa == True:
        pass
    else:
        da_object.label_encoder.fit(all_labels)
        da_object.nclasses = len(da_object.label_encoder.classes_)
        print("da_object.nclasses ",da_object.nclasses)
    da_object.construct_vocab(all_sentences)
    # print('da_object.cache_tokenize  preprocess',da_object.cache_tokenize.keys())
    for split,dataset in da_object.datasets.items():
        dataset.set_lable_encoder(da_object.label_encoder)
        dataset.set_vocab(da_object.token2id, da_object.id2token)
        dataset.set_cache_tokenize(da_object.cache_tokenize)

def preprocess_pair(da_object):
    all_labels = []
    all_sentences = []
    da_object.processer = get_preprocesser(**da_object.preprocesser_cfg)
    for split,dataset in da_object.datasets.items():
        all_labels.extend(dataset.labels)
        if da_object.tokenizer_type == 'non_bert':
            for sample in dataset.data:
                sample['s1'] = da_object.processer.process_one(sample['s1'])
                sample['s2'] = da_object.processer.process_one(sample['s2'])
                all_sentences.append(sample['s1'])
                all_sentences.append(sample['s2'])
        
    da_object.label_encoder.fit(all_labels)
    da_object.nclasses = len(da_object.label_encoder.classes_)
    da_object.construct_vocab(all_sentences)
    # print('da_object.cache_tokenize  preprocess',da_object.cache_tokenize.keys())
    for split,dataset in da_object.datasets.items():
        dataset.set_lable_encoder(da_object.label_encoder)
        dataset.set_vocab(da_object.token2id, da_object.id2token)
        dataset.set_cache_tokenize(da_object.cache_tokenize)

def construct_vocab(da_object,all_corpus):
    # to do 
    # re-write this function
    da_object.cache_tokenize = {} # only used for none bert tokenizer
    da_object.token2id, da_object.id2token = {},{}
    # print('vocab_path ',da_object.vocab_name)
    # if os.path.exists(da_object.vocab_name):
    #     data= torch.load(da_object.vocab_name)
    #     da_object.cache_tokenize = data['cache_tokenize']
    #     da_object.token2id = data['token2id']
    #     da_object.id2token =  data['id2token']
    #     print('da_object.token2id  ',len(da_object.token2id),len(da_object.cache_tokenize))
    #     return 
    # check whether construct_vocab distributed
    # due to the running time of constructing discocat vocab of R8
    # So this is only used for R8
    only_construct_vocab = False
    if da_object.tokenizer_type == "bert":
        for token, id in da_object.tokenizer.vocab.items():
            da_object.token2id[token] = id
            da_object.id2token[id] = token
        print('da_object.token2id  ',len(da_object.token2id))
        # da_object.initial_embedding = "None----Using bert"
    else:
        assert False, 'Only support bert tokenizer'
    # if not only_construct_vocab:
    tokenizer_name = da_object.tokenizer_name.replace('/','_')
    vocab_name = '{}_{}_{}'.format(da_object.global_config['DATA']['dataset_name'],\
                                   da_object.tokenizer_type, tokenizer_name)
    torch.save({'token2id':da_object.token2id,'id2token':da_object.id2token, \
            'cache_tokenize':da_object.cache_tokenize},'results/cache/vocabs/{}'.format(vocab_name))
    # else:
    #     filename = da_object.vocab_name[:-3] + '_'+da_object.vocab_split + '.pt'
    #     torch.save({'token2id':da_object.token2id,'id2token':da_object.id2token, \
    #                     'cache_tokenize':da_object.cache_tokenize},filename)
    #     import sys
    #     sys.exit(0)


def add_tokens(da_object,tokens):
    # print("tokens ", tokens)
    for token in tokens:
        if token not in da_object.token2id:
            # print(token)
            da_object.token2id[token] = len(da_object.token2id)
            # print('token ',token,len(da_object.token2id))
            da_object.id2token[len(da_object.id2token)] = token

def init_tokenizer(da_object):
    if da_object.tokenizer_type == 'non_bert':
        da_object.tokenizer = get_tokenizer(da_object.tokenizer_type,da_object.tokenizer_name,da_object.tokenizer_params)
    elif da_object.tokenizer_type == 'bert':
        da_object.tokenizer_real_name = 'results/cache/tokenizers/{}_{}/'.format(da_object.global_config['DATA']['dataset_name']\
                                                                         ,da_object.tokenizer_name.replace('/','_'))
        if os.path.exists(da_object.tokenizer_real_name):
            '''
            https://huggingface.co/course/chapter6/2
            huggingface said the tokenizer training is determinstic, but in my experiments it will produce different tokenizer
            with the same corpus, so I use the saved pretrained tokenizer
            '''
            print("use previous trained bert-like tokenizer ",da_object.tokenizer_real_name)
            da_object.tokenizer = get_tokenizer('bert',da_object.tokenizer_real_name)
        else:
            if da_object.tokenizer_name in ('blank_en', 'wordpiece'):
                corpus = []
                for split,dataset in da_object.datasets.items():
                    for raw_data in dataset.data:
                        corpus.append(raw_data)
                da_object.tokenizer = get_tokenizer(da_object.tokenizer_type,da_object.tokenizer_name,da_object.tokenizer_params,\
                    corpus = corpus)
            else:
                da_object.tokenizer = get_tokenizer(da_object.tokenizer_type,da_object.tokenizer_name,da_object.tokenizer_params)
            da_object.tokenizer.save_pretrained(da_object.tokenizer_real_name)