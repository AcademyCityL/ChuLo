# -*- coding: utf-8 -*-
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from nltk.stem import SnowballStemmer
import re, string
import sys
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer 
import benepar
benepar.download('benepar_en3')
import copy
from tokenizers import normalizers,pre_tokenizers
import stanza
import re
g_parser = None

class ConstituencyParser(object):
    def __init__(self, parser_type='berkeley'):
        self.parser_type = parser_type
        if self.parser_type == 'berkeley':
            self.parser = benepar.Parser("benepar_en3")
        elif self.parser_type == 'stanford':
            self.parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',tokenize_no_ssplit=True)
            # self.reg = re.compile(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])')
    
    def parse(self, sentence):
        if self.parser_type == 'berkeley':
            return self.parser.parse(sentence)
        elif self.parser_type == 'stanford':
            # sentence = self.reg.sub('_', sentence) # to avoid sentence split
            # sentence = sentence.replace('.', '_')
            doc = self.parser(sentence)
            return nltk.tree.tree.Tree.fromstring(str(doc.sentences[0].constituency))
        
 
def get_preprocesser(cfg):
    real_cfg = copy.deepcopy(cfg)
    name = real_cfg.pop('name', 'normal')
    # print(":name ",name=='bs')
    if name == 'normal':
        return Preprocesser(**real_cfg)
    elif name == 'twitter':
        return TwitterPreprocesser(**real_cfg)
    elif name == 'imdb':
        return IMDBPreprocesser(**real_cfg)
    elif name == 'BertGCN':
        return BertGCNPreprocesser(**real_cfg)
    elif name == 'LongformerHp':
        return LongformerHpPreprocesser(**real_cfg)
    elif name == 'TextGCN':
        return TextGCNPreprocesser(**real_cfg)
    elif name == 'bs':
        # print("Noneprocesser ")
        return Noneprocesser(**real_cfg)
    elif name == 'eurlex':
        return Noneprocesser(**real_cfg)
    elif name == 'None':
        return Noneprocesser(**real_cfg)

class LongformerHpPreprocesser(object):
    FLAGS = re.MULTILINE | re.DOTALL
    def __init__(self,**kwargs):
        '''
        Use the same preprocess as https://github.com/allenai/longformer/blob/master/scripts/hp_preprocess.py
        But add lowercase
        paper:Longformer: The Long-Document Transformer
        '''
        
    def re_sub(self, pattern, repl, text, flags=None):
        if flags is None:
            return re.sub(pattern, repl, text, flags=self.FLAGS)
        else:
            return re.sub(pattern, repl, text, flags=(self.FLAGS | flags))


    def clean_txt(self,text):

        text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"&#160;", "", text)

        # Remove URL
        text = self.re_sub(r"(http)\S+", "", text)
        text = self.re_sub(r"(www)\S+", "", text)
        text = self.re_sub(r"(href)\S+", "", text)
        # Remove multiple spaces
        text = self.re_sub(r"[ \s\t\n]+", " ", text)

        # remove repetition
        text = self.re_sub(r"([!?.]){2,}", r"\1", text)
        text = self.re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

        return text.strip().lower()

    def process_one(self,text,**kwargs):
        return self.clean_txt(text)

class TextGCNPreprocesser(object):
    '''
    Use the same preprocess as https://github.com/yao8839836/text_gcn/blob/962223652e9bb164ac2d83cd09fc7b8845ce860b/utils.py#L281
    TextGCN
    '''
    def process_one(self,text,**kwargs):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

class BertGCNPreprocesser(object):
    '''
    Use the same preprocess as https://github.com/ZeroRin/BertGCN/blob/main/utils/utils.py
    paper: https://arxiv.org/abs/2105.05727 
    BertGCN: Transductive Text Classification by Combining GCN and BERT
    '''
    def __init__(self,**kwargs):
        pass
    def process_one(self,text,**kwargs):
        # print("text ",text)
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()
    
class IMDBPreprocesser(object):
    '''
    Use the same preprocess as https://github.com/SanghunYun/UDA_pytorch/blob/master/utils/tokenization.py
    '''
    def __init__(self,**kwargs):
        self.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.BertNormalizer(clean_text=True,handle_chinese_chars=True,\
            lowercase=True,)]) # BertNormalizer also include stripaccent
        self.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation(behavior = 'isolated' )])
    def process_one(self,text,**kwargs):
        text = self.normalizer.normalize_str(text)
        text = self.pre_tokenizer.pre_tokenize_str(text)
        text = ' '.join([i[0] for i in text])
        return text

class Noneprocesser(object):
    def __init__(self,**kwargs):
        pass
    def process_one(self,text,**kwargs):
        return text

class Preprocesser(object):
    def __init__(self, remove_punctuation=False, stem=False, lower=False, \
        stopword=False,sentence_split=False):
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer=SnowballStemmer('english')
        self.valid_lang = ['en', 'cn'] 

        #remove punctuation
        self.remove_punctuation = remove_punctuation

        # sentence split
        # Split the sentences, then each document will use [SEP] to concatenate the sentences. 
        # So the tokenizer must process these [SEP]
        # But if use transformer-tokenizer, it doesn't need to insert SEP and remove punctuations
        self.sentence_split = sentence_split
        
        
        # word segmentation
        self.word_seg_enable = True
        self.word_seg_lang = 'en'
        
        # word stemming
        self.stem = stem
        
        # word lowercase
        self.lower = lower
          
        # stop word removal
        self.stopword = stopword

    
    def get_default_tokenizer(self):
        return word_tokenize

    def process_one(self,text,remove_punc = True,stem=True,lower=True,stopword=True,special_punc=None):
        
        if self.sentence_split == True:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = [text]
        ouput_text = []
        for sentence in sentences:

            if self.remove_punctuation == True and remove_punc == True:
                sentence = self.punct_remove(sentence)
            

            tokenizer = self.get_default_tokenizer()
            sentence = tokenizer(sentence)
            if self.stem == True and stem == True:
                sentence = self.word_stem(sentence)

            if self.lower and lower == True:
                sentence = self.word_lower(sentence)
                
            if self.stopword and stopword == True:
                sentence = self.stopword_remove(sentence)
            
            sentence = " ".join(sentence)
            ouput_text.append(sentence)
        ouput_text = " [SEP] ".join(ouput_text)
        return ouput_text
    
                
    def run(self,sentences):
        
        output = []
#        print('preprocess begins.')
        for sentence in sentences:
            output.append(self.process_one(sentence))
#        print('preprocess ends.')
        return output
    
    def is_stopword(self,word):
        return word in self.stopwords_set

    def stopword_remove(self,sentence,return_ids = False):
        if return_ids == True:
            tokenizer = self.get_default_tokenizer()
            sentence = tokenizer(sentence)
            ret_sentence = []
            ret_ids = []
            for i in range(len(sentence)):
                if sentence[i] not in self.stopwords_set:
                    ret_sentence.append(sentence[i])
                    ret_ids.append(i)
            ret_sentence = " ".join(ret_sentence)
            return ret_sentence, ret_ids
        else:
            sentence = [w for w in sentence if w not in self.stopwords_set]
        return sentence
    
    def punct_remove(self,sentence):
        # print(sentence)
        sentence = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', sentence)
        return sentence
    
    def word_lower(self,sentence):
        sentence = [w.lower() for w in sentence]
        return sentence
    
    def word_seg_en(self,sentence):
        sentence= word_tokenize(sentence) 
        # show the progress of word segmentation with tqdm
        '''docs_seg = []
        print('docs size', len(docs))
        for i in tqdm(range(len(docs))):
            docs_seg.append(word_tokenize(docs[i]))'''
        return sentence
    
#     def word_seg_cn(self,sentence):
#         sentence = list(jieba.cut(sentence))
# #        docs = [list(jieba.cut(sent)) for sent in docs]
#         return sentence

    def stem_one_word(self,word):
        return self.stemmer.stem(word) 
                                   
    def word_stem(self,sentence):
        sentence = [self.stemmer.stem(w) for w in sentence]
        return sentence
    
    def check_need_remove_sw(self,token_list):
        return True
    
    def process_one_word(self,word,need_remove_sw):
        word = self.stem_one_word(word)
        if need_remove_sw and self.is_stopword(word):
            return True
        return word

def load_stopwords():
    stopwords = set()
    for file_name in ['files/ravikiranj_stopwords.txt','files/snowball_stopwords.txt','files/web_confs_stopwords.txt']:
        with open(file_name, 'r') as f:
            for line in f:
                line=line.strip('\n')
                word  = line.split(' ')[0]
                if word not in ['|', ' ','', '#']:
                    stopwords.add(word)
    return stopwords



class TwitterPreprocesser():
    def __init__(self, stock_market = True,old_style = True,sentence_split = False,\
                  hyperlik = True,hashtag = True,punc = True,stopword = True,stem = True):
        self.stemmer = PorterStemmer()
        self.stopwords_english = load_stopwords()
        
        self.stock_market = stock_market
        self.old_style = old_style
        self.sentence_split = sentence_split
        self.hyperlik = hyperlik
        self.hashtag = hashtag
        self.punc = punc
        self.stopword = stopword
        self.stem = stem

    def get_default_tokenizer(self):
        return TweetTokenizer(preserve_case=False, strip_handles=True,
                                    reduce_len=True)
    
    def check_need_remove_sw(self,token_list):
        '''
        ONly used in chunk and align
        '''
        tmp = []
        for word in token_list:
            if self.is_stopword(word) == False:
                tmp.append(word)
        if len(tmp) == 0:
            # print("tmp ",tmp)
            return False
        return True

    def process_one_word(self,word,need_remove_sw):
        if need_remove_sw and self.is_stopword(word):
            return True
        word = self.stem_one_word(word)
        # print("stem ",word)
        return word
    
    def spe_punct_remove(self,sentence):
        # print(sentence)
        sentence = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', sentence)
        return sentence

    def process_one(self,text,stock_market = True,old_style = True,\
                  hyperlik = True,hashtag = True,punc = True,stopword = True,stem = True,special_punc = False):
        '''
        remove accents, like 'mehra sachi aha that s alright ❤ ️ we love you too a'
        There is a special character in the right of heart, it's neither a whitespace or blank, but its length
        is 1, and it can be printed like a whitespace
        '''
        f = normalizers.BertNormalizer(clean_text=True,handle_chinese_chars=True,\
            lowercase=True ) # hao yong
        text = f.normalize_str(text) 
        if self.sentence_split == True:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = [text]
        ouput_text = []
        record = []
        for tweet in sentences:
            record.append({})
            record[-1]['ori'] = tweet
            # remove stock market tickers like $GE
            if self.stock_market == True and stock_market == True:
                tweet = re.sub(r'\$\w*', '', tweet)
                record[-1]['stock_market'] = tweet
            
            # remove old style retweet text "RT"
            if self.old_style == True and old_style == True:
                tweet = re.sub(r'^RT[\s]+', '', tweet)
                record[-1]['old_style'] = tweet
            
            # remove hyperlinks
            if self.hyperlik == True and hyperlik == True:
                #[^\n\r] equals to . , [\s] equals to whitespace
                tweet = re.sub(r'https?:\/\/[^\n\r\s]*[\r\n]*', '', tweet)
                record[-1]['hyperlik'] = tweet
            
            # remove hashtags
            # only removing the hash # sign from the word
            if self.hashtag == True and hashtag == True:
                tweet = re.sub(r'#', '', tweet)
                record[-1]['hashtag'] = tweet

            # remove special punctuation
            if special_punc == True:
                tweet = self.spe_punct_remove(tweet)
                record[-1]['special_punc'] = tweet
    
            # tokenize tweets
            tokenizer = self.get_default_tokenizer()
            tweet_tokens = tokenizer.tokenize(tweet)
            record[-1]['tokenize'] = tweet_tokens
            if self.punc == True and punc == True:
                tmp = []
                for word in tweet_tokens:
                    if word not in string.punctuation:
                        tmp.append(word)
                tweet_tokens = tmp
                record[-1]['punc'] = tweet_tokens

            if self.stopword == True and stopword == True:
                tmp = []
                for word in tweet_tokens:
                    if self.is_stopword(word) == False:
                        tmp.append(word)
                tweet_tokens = tmp
                record[-1]['stopword'] = tweet_tokens
                if len(tweet_tokens) == 0:
                    tweet_tokens = record[-1]['punc']

            if self.stem == True and stem == True:
                tweet_tokens = [self.stem_one_word(word) for word in tweet_tokens]
                record[-1]['stem'] = tweet_tokens

            tweet_tokens = ' '.join(tweet_tokens)
            ouput_text.append(tweet_tokens)
        ouput_text = " [SEP] ".join(ouput_text)
        if len(ouput_text) == 0:
            print(record)
            print(ouput_text)

        return ouput_text
    
    def is_stopword(self,word):
        return word in self.stopwords_english
    
    def stem_one_word(self,word):
        return self.stemmer.stem(word) 

def process_tweet(text,add_sep= False,stock_market = True,old_style = True,\
                  hyperlik = True,hashtag = True,punc = True,sw = True,stem = True,):
    '''
    dropped, use TwitterPreprocesser.process_one instead
    '''
    if add_sep == True:
        sentences = nltk.sent_tokenize(text)
    else:
        sentences = [text]
    ouput_text = []
    for tweet in sentences:
        stemmer = PorterStemmer()
        stopwords_english = load_stopwords()
        
        # remove stock market tickers like $GE
        if stock_market == True:
            tweet = re.sub(r'\$\w*', '', tweet)
        
        # remove old style retweet text "RT"
        if old_style == True:
            tweet = re.sub(r'^RT[\s]+', '', tweet)
        
        # remove hyperlinks
        if hyperlik == True:
            tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        
        # remove hashtags
        # only removing the hash # sign from the word
        if hashtag == True:
            tweet = re.sub(r'#', '', tweet)
        
        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (((word not in stopwords_english) or sw == False) and  # remove stopwords
                    ((word not in string.punctuation) or punc == False)):  # remove punctuation
                # tweets_clean.append(word)
                if stem == True:
                    stem_word = stemmer.stem(word)  # stemming word
                else:
                    stem_word = word
                tweets_clean.append(stem_word)

        tweets_clean = ' '.join(tweets_clean)
        ouput_text.append(tweets_clean)
    ouput_text = " [SEP] ".join(ouput_text)
    return ouput_text

def check_self(tree):
    self_label = tree.label()
    for pos in tree.treepositions():
        if len(pos) == 0: # top node
            continue
        if not isinstance(tree[pos],str):
            # print(tree[pos].label(),self_label)
            if tree[pos].label() == self_label:
                return False
    return True

def check_subtree(chunks,tree_pos):
    str_pos = str(tree_pos)
    for pos_str in chunks:
        if pos_str + ',' == str_pos[1:len(pos_str)+2]: # avoid 1, 1 in 1, 10, 1 or 0, 1 in 1, 10, 1
            return False
    return True

def merge_chunks(word_chunks,chunk_types,sentence=None,except_phs=[]):
    # first merge the error splited words
    if sentence is not None:
        word_idx = 0
        word_list = sentence.split()
        new_word_chunks = []
        new_chunk_types = []
        tmp_chunk = []
        tmp_word = ''
        tmp_type = None
        for j in range(len(word_chunks)):
            chunk = word_chunks[j]
            chunk_type = chunk_types[j]
        # for chunk, chunk_type in zip(word_chunks,chunk_types):
            if len(tmp_word) == 0:
                # not in merging progress
                tmp_type = chunk_type
            for i in range(len(chunk)):
                word = chunk[i]
                if word == word_list[word_idx]:
                    tmp_chunk.append(word)
                    word_idx += 1
                elif len(tmp_word) == 0 and word != word_list[word_idx]:
                    # For stanford parser, the special words like ':)' will be transformed to ':-RRB-', then results in the failure
                    # To avoid it, we need to check whether the word is a sub-word
                    if word not in word_list[word_idx]:
                        tmp_chunk.append(word_list[word_idx])
                        word_idx += 1
                    else:
                        # start merging subwords, and all the splited subwords are merged into the chunk containing the first subword
                        tmp_word += word
                elif len(tmp_word) > 0:
                    # merging progress
                    tmp_word += word
                    if tmp_word == word_list[word_idx]:
                        # end merging
                        tmp_chunk.append(tmp_word)
                        tmp_word = ''
                        word_idx += 1
            # arter going through a chunk, if the merging is finished, then add this chunk
            if len(tmp_word) == 0:
                new_word_chunks.append(tmp_chunk)
                new_chunk_types.append(tmp_type)
                tmp_chunk = []
                tmp_type = None
            elif j == len(word_chunks) - 1:
                # This aims to deal with the case that stanford parser may modify the word strangely, like 
                # 'in the building homeboyzradio djbashkenya' will be parsed as 'in the building homeboyzradio djbashkeny'
                # 'a' is droped
                tmp_chunk.append(word_list[word_idx])
                tmp_word = ''
                new_word_chunks.append(tmp_chunk)
                new_chunk_types.append(tmp_type)
                tmp_chunk = []
                tmp_type = None
            else:
                AssertionError("The merging is not finished!")

        # print("word_chunks     ",word_chunks)
        # print("new_word_chunks ",new_word_chunks)
        # print("tmp_chunk       ",tmp_chunk)
        # print("tmp_word        ",tmp_word)
        word_chunks = new_word_chunks
        chunk_types = new_chunk_types
    new_word_chunks = []
    tmp_chunk = []
    for i,ii in zip(word_chunks,chunk_types):
        if len(i) > 1 or ii in except_phs:
            if len(tmp_chunk) > 0:
                new_word_chunks.append(tmp_chunk)
            new_word_chunks.append(i)
            tmp_chunk = []
        else:
            tmp_chunk += i
    if len(tmp_chunk) > 0:
        new_word_chunks.append(tmp_chunk)
    return new_word_chunks

# def chunk_sentence(sentence):
#     '''
#     1. Search the tree from top to bottom, each subtree is a chunk if there is not another label the same as the root of the subtree.
#     2. Merge the adjacent chunks (not include single NP word) only have one word
#     '''

#     global g_parser
#     # print("============================")
#     # print("parse sentence: ",sentence)
#     tree = g_parser.parse(sentence)
#     chunks = []
#     chunk_types = []
#     str_check = set([])
#     for pos in tree.treepositions():
#         ch = tree[pos]
#         # print(str(pos))
#         if len(pos) <=1 : # top node
#             continue
#         if isinstance(ch,str): # leaf node
#             continue
#         if check_subtree(str_check,pos):
#             # print(pos)
#             if check_self(ch):
#                 chunks.append(ch)
#                 chunk_types.append(ch.label())
#                 str_check.add(str(pos)[1:-1])

#     word_chunks = []
#     if len(chunks) == 0:
#         # single word case
#         chunks.append(tree[0])
#         chunk_types.append(tree[0].label())
#     for i in chunks:
#         word_chunks.append(i.leaves())
    
#     '''
#     merge one-word chunks and error splited words, like :) in tweet will be splited as two words : and )
#     which should be merged as one word
#     '''
#     try:
#         new_word_chunks = merge_chunks(word_chunks,chunk_types,sentence,[])
#     except:
#         print("ERROR merge_chunks")
#         print("tree ",tree.pretty_print())
#         print('word_chunks',word_chunks)
#         print("sentence ",sentence)
#         assert False, "..."
#     # print("=====================")
#     # print(sentence)
#     # print(new_word_chunks)
#     # print(tree.pretty_print())
#     if sum([len(chunk) for chunk in new_word_chunks]) != len(sentence.split()):
#         print("ERROR CHUNKING")
#         print("tree ",tree.pretty_print())
#         print('word_chunks',word_chunks)
#         print("new_word_chunks ",new_word_chunks)
#         print("sentence ",sentence)
#         assert False, "..."
#     return new_word_chunks,tree,word_chunks

def check_self_not_have(tree,labels):
    self_label = tree.label()
    for pos in tree.treepositions():
        if len(pos) == 0: # top node
            continue
        if not isinstance(tree[pos],str):
            # print(tree[pos].label(),self_label)
            if tree[pos].label() in labels:
                return False
    return True
   
def chunk_sentence(sentence,consider_tags=None):
    '''
    1. Search the tree from bottom to top.
    And only search minimal phrase chunks. 
    '''
    consider_tags = consider_tags or set('ADJP','ADVP','FRAG', 'INTJ', 'NAC','NP','NX','PP','PRN','PRT','QP',\
                         'RRC','UCP','VP','WHADJP','WHAVP','WHNP','WHPP','X')
    global g_parser
    tree = g_parser.parse(sentence)
    chunks = []
    chunk_types = []
    str_check = set([])
    for pos in tree.treepositions():
        ch = tree[pos]
        # print(str(pos))
        if len(pos) <=1: # top node
            continue
        if isinstance(ch,str): # leaf node
            continue
        if check_subtree(str_check,pos):
            # print(pos)
            if (ch.label() in consider_tags and check_self(ch)): 
                #or check_self_not_have(ch,('NP','PP')):
                chunks.append(ch)
                chunk_types.append(ch.label())
                str_check.add(str(pos)[1:-1])
    word_chunks = []
    if len(chunks) == 0:
        # single word case
        chunks.append(tree[0])
        chunk_types.append(tree[0].label())
    for i in chunks:
        word_chunks.append(i.leaves())
    
    # merge one-word chunks except NP
    new_word_chunks = merge_chunks(word_chunks,chunk_types,sentence,[])
    
    # print("=====================")
    # print(sentence)
    # print(new_word_chunks)
    # print(tree.pretty_print())

    if sum([len(chunk) for chunk in new_word_chunks]) != len(sentence.split()):
        print("ERROR CHUNKING")
        print(tree.pretty_print())
        print(new_word_chunks)
        assert False, "..."
    return new_word_chunks,tree,word_chunks