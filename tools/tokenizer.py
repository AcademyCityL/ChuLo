# -*- coding: utf-8 -*-
from tokenizers import decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer 
import spacy

from collections import defaultdict

def help_special_token(tokenzier, tokens):
    pass


class Nltk_tokenizer:
    def __init__(self, word_tokenize, params={}):
        self.tokenizer = word_tokenize
        self.params = params
        self.add_sep = self.params.get('add_sep',False)
        self.add_cls = self.params.get('add_cls',False)

    def tokenize(self,text):
        sep = ' [SEP] '
        if self.add_sep == False:
            text = text.replace(' [SEP] ', ' ')
            sep = ' '
        sentences = text.split(' [SEP] ')
        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenizer(sentence) 
            tokens = ' '.join(tokens) 
            all_tokens.append(tokens)
        text_tokens = sep.join(all_tokens).split()
        if self.add_cls == True:
            text_tokens = ['[CLS]'] + text_tokens
        return text_tokens

    def set_vocab(self, vocab):
        self.vocab = vocab


class Nltk_tweet_tokenizer:
    def __init__(self, word_tokenize, params={}):
        self.tokenizer = word_tokenize
        self.params = params
        self.add_sep = self.params.get('add_sep',False)
        self.add_cls = self.params.get('add_cls',False)

    def tokenize(self,text):
        sep = ' [SEP] '
        if self.add_sep == False:
            text = text.replace(' [SEP] ', ' ')
            sep = ' '
        sentences = text.split(' [SEP] ')
        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            tokens = ' '.join(tokens) 
            all_tokens.append(tokens)
        text_tokens = sep.join(all_tokens).split()
        if self.add_cls == True:
            text_tokens = ['[CLS]'] + text_tokens
        return text_tokens

    def set_vocab(self, vocab):
        self.vocab = vocab

class Spacy_space_en:
    def __init__(self, params={}):
        self.tokenizer = spacy.blank("en")
        self.params = params
        self.add_sep = self.params.get('add_sep',False)
        self.add_cls = self.params.get('add_cls',False)

    def tokenize(self,text):
        sep = ' [SEP] '
        if self.add_sep == False:
            text = text.replace(' [SEP] ', ' ')
            sep = ' '
        sentences = text.split(' [SEP] ')
        all_tokens = []
        for sentence in sentences:
            doc = self.tokenizer(sentence)
            tokens = [token.text for token in doc]
            tokens = ' '.join(tokens) 
            all_tokens.append(tokens)
        text_tokens = sep.join(all_tokens).split()
        if self.add_cls == True:
            text_tokens = ['[CLS]'] + text_tokens
        return text_tokens

class WhiteSpace:
    def __init__(self, params={}):
        self.tokenizer = WhitespaceTokenizer
        self.params = params
        self.add_sep = self.params.get('add_sep',False)
        self.add_cls = self.params.get('add_cls',False)

    def tokenize(self,text):
        sep = ' [SEP] '
        if self.add_sep == False:
            text = text.replace(' [SEP] ', ' ')
            sep = ' '
        sentences = text.split(' [SEP] ')
        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence) 
            tokens = ' '.join(tokens) 
            all_tokens.append(tokens)
        text_tokens = sep.join(all_tokens).split()
        if self.add_cls == True:
            text_tokens = ['[CLS]'] + text_tokens
        return text_tokens

def get_tokenizer(tokenizer_type='non_bert', tokenizer_name = 'blank_en',tokenizer_params = {}, **kwargs):
    if tokenizer_type == 'non_bert':
        if tokenizer_name == 'whitespace':
                return WhiteSpace(params = tokenizer_params)
        elif tokenizer_name == 'nltk':
            return Nltk_tokenizer(word_tokenize,params = tokenizer_params)
        elif tokenizer_name == 'nltk_tweet':
            return Nltk_tweet_tokenizer(TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True),params = tokenizer_params)
        elif tokenizer_name == 'spacy_blank_en':
            return Spacy_space_en(params = tokenizer_params)
        elif tokenizer_name == 'discocat':
            return Discocat_tokenizer(params = tokenizer_params)
    elif tokenizer_type == 'bert':
        if tokenizer_name == 'blank_en':
            return get_en_space_tokenizer(kwargs['corpus'],**tokenizer_params)
        elif tokenizer_name == 'wordpiece':
            return get_wordpiece_tokenizer(kwargs['corpus'],**tokenizer_params)
        else:
            # print(tokenizer_name , tokenizer_params)
            return AutoTokenizer.from_pretrained(tokenizer_name,**tokenizer_params)

def get_en_space_tokenizer(corpus=[],clean=False,lowercase=False):
    # corpus: list[string], ["l kslcka as","sd, asdq, sss"]
    # the tokenizer can replace nltk.word_tokenize or spacy.space('en')
    # https://huggingface.co/docs/tokenizers/v0.13.2/en/api/normalizers
    ct = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    ct.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.BertNormalizer(clean_text=clean,handle_chinese_chars=True,\
                                                         lowercase=lowercase )]#normalizers.Lowercase()]
    )
    ct.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(behavior = 'isolated' )]
    )
    # ct = AutoTokenizer(tokenizer_object = ct)
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]','[BOS]', '[EOS]', '[MASK]']
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
    ct.train_from_iterator(corpus,trainer = trainer) # [te] should include all corpus,can't train again later to increase vocab
    ct.post_processor = processors.BertProcessing(('[SEP]', ct.token_to_id('[SEP]')),('[CLS]', ct.token_to_id('[CLS]')))
    return PreTrainedTokenizerFast(tokenizer_object=ct,bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]',sep_token='[SEP]',\
        pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
    # the follow is the code to get original input string, b = PreTrainedTokenizerFast or AutoTokenizer,
    # b([te]) = BatchEncoding
    #    length = len(b.convert_ids_to_tokens(bte['input_ids'],skip_special_tokens=True))
    #    sp = b([te]).token_to_chars(length)
    #    print(te[0:sp[1]])

def get_wordpiece_tokenizer(corpus=[],clean=False,lowercase=False, add_spe_tokens=[]):
    # corpus: list[string], ["l kslcka as","sd, asdq, sss"]
    # the tokenizer can replace bert_base_cased
    # https://huggingface.co/docs/tokenizers/v0.13.2/en/api/normalizers
    ct = Tokenizer(models.WordPiece(unk_token='[UNK]'))
    ct.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.BertNormalizer(clean_text=clean,handle_chinese_chars=True,\
            lowercase=lowercase )]#normalizers.Lowercase()]
    )
    ct.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(behavior = 'isolated' )]
    )
    # ct = AutoTokenizer(tokenizer_object = ct)
    special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]','[BOS]', '[EOS]', '[MASK]'] + add_spe_tokens
    trainer = trainers.WordPieceTrainer(special_tokens=special_tokens)
    #( vocab_size = 30000 min_frequency = 0 show_progress = True special_tokens = []\
    # limit_alphabet = None initial_alphabet = [] continuing_subword_prefix = '##' end_of_word_suffix = None )
    ct.train_from_iterator(corpus,trainer = trainer) # [te] should include all corpus,can't train again later to increase vocab
    ct.post_processor = processors.BertProcessing(('[SEP]', ct.token_to_id('[SEP]')),('[CLS]', ct.token_to_id('[CLS]')))
    return PreTrainedTokenizerFast(tokenizer_object=ct,bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]',sep_token='[SEP]',\
        pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
    # the follow is the code to get original input string, b = PreTrainedTokenizerFast or AutoTokenizer,
    # b([te]) = BatchEncoding
    #    length = len(b.convert_ids_to_tokens(bte['input_ids'],skip_special_tokens=True))
    #    sp = b([te]).token_to_chars(length)
    #    print(te[0:sp[1]])
