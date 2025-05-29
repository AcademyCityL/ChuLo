# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np
import gensim.downloader
gensim.downloader.BASE_DIR = 'files/'
import gensim
import os
import pickle
from transformers import AutoConfig, AutoModel
import copy
import torch.nn.functional as F

'''
临时复制代码
'''
class RelativePosition(nn.Module):
    def __init__(self, d_model, max_k=4):
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k # max relative position
        self.pe = nn.Parameter(torch.Tensor(max_k * 2 + 1, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, length_q,length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_k, self.max_k)
        final_mat = distance_mat_clipped + self.max_k
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.pe[final_mat].cuda() # shape: [length_q, length_k, d_model]
        return embeddings
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        '''
        Copy from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print('............  ',pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1),:]
        return self.dropout(x)

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        #
        return torch.cat((x_rope, x_pass), dim=-1)

class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_head: int, bias: bool):
        super().__init__()

        self.d_model = d_model

        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, n_heads * d_head, bias=bias)
        # Number of n_heads
        self.n_heads = n_heads
        # Number of dimensions in vectors in each head
        self.d_head = d_head

    def forward(self, x: torch.Tensor):
        # Input has shape `[batch_size, seq_len, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the n_heads.
        batch_size, seq_len, d_model = x.shape
        assert d_model==self.d_model, "self.d_model != d_model"

        # Linear transform
        x = self.linear(x)

        # Split last dimension into n_heads
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Output has shape `[batch_size, n_heads, seq_len, d_head]`
        return x
    

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, dropout = 0.1, bias = True, pe_type = 'absolute_sin'):
        """
        * `n_heads` is the number of n_heads.
        * `d_model` is the number of features in the `query`, [`key` and `value`] vectors.
        """

        super().__init__()

        self.d_model = d_model

        # Number of features per head
        self.d_head = d_model // n_heads
        # Number of n_heads
        self.n_heads = n_heads

        assert self.d_model == self.n_heads * self.d_head, "self.d_model != self.n_heads * self.d_head"

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, n_heads, self.d_head, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, n_heads, self.d_head, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, n_heads, self.d_head, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Scaling factor before the softmax
        self.scale = math.sqrt(self.d_head)

        # code for special pe
        self.pe_type = pe_type
        if self.pe_type == 'relative_pos':
            self.rel_pe_k = RelativePosition(self.d_head)
            self.rel_pe_v = RelativePosition(self.d_head)

    def scaled_dot_product_score(self, q, k,rel=None):
        # print("scaled_dot_product ")    
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / self.scale
        if rel is not None:
            bs, n_heads, len_q, head_dim = q.shape
            _, _, len_k, _ = k.shape
            # print('q shape',q.shape)
            q2 = q.permute(2,0,1,3).reshape(len_q, bs*n_heads, head_dim)
            rel_weight = torch.matmul(q2, rel[0].transpose(1, 2)).transpose(0, 1)
            rel_weight = rel_weight.contiguous().view(bs, n_heads, len_q, len_k)/self.scale
            attn_logits += rel_weight

        # if mask is not None:
        #     # print('attn_logits ',attn_logits.shape, mask.shape) 
        #     attn_logits = attn_logits.masked_fill(mask == 0, torch.inf)
        return attn_logits
        # attention = F.softmax(attn_logits, dim=-1)
        # values = torch.matmul(attention, v)
        # if rel is not None:
        #     rel_weight = attention.permute(2, 0, 1, 3).contiguous().reshape(len_q, bs*n_heads, len_k)
        #     rel_weight = torch.matmul(rel_weight, rel[1]).transpose(0, 1)
        #     rel_weight = rel_weight.contiguous().view(bs, n_heads, len_q, head_dim)
        #     values += rel_weight
        # return values, attention

    def get_scores(self, query, key, rel=None):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        return self.scaled_dot_product_score(query,key,rel)

    def forward(self,query,key,value,mask= None):
      
        # `query`, `key` and `value`  have shape `[batch_size, seq_len, d_model]`


        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[batch_size, n_heads, seq_len, d_head]`.
        # print("query ",query.shape,key.shape,value.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        batch_size, n_heads, len_q, d_head = query.shape

        # Compute attention scores $Q K^\top$.
        if self.pe_type == 'relative_pos':
            len_k = key.shape[2]
            rel = [self.rel_pe_k(len_k,len_k),self.rel_pe_v(len_k,len_k)]
        else:
            rel = None
        # This gives a tensor of shape `[batch_size, n_heads, seq_len, seq_len]`.
        # print("query ",query.shape,key.shape)
        scores = self.get_scores(query, key, rel)

        # Apply mask
        # print("scores ",scores.shape,mask.shape,scores)
        if mask is not None:
            if len(mask.shape) == 2:
                real_mask = mask.unsqueeze(1).unsqueeze(2)
            else:
                real_mask = mask
            scores = scores.masked_fill(real_mask == 0, float("-inf"))
            # print("scores 2",scores)

        attn = F.softmax(scores,dim=-1)
        # print("scores 3",attn)
        # Apply dropout
        if self.training:
            attn = self.dropout(attn)
        
        attn_output = torch.matmul(attn,value)

        if self.pe_type == 'relative_pos':
            rel_weight = attn.permute(2, 0, 1, 3).contiguous().reshape(len_q, batch_size*n_heads, len_k)
            rel_weight = torch.matmul(rel_weight, rel[1]).transpose(0, 1)
            rel_weight = rel_weight.contiguous().view(batch_size, n_heads, len_q, d_head)
            attn_output += rel_weight

        # Concatenate multiple n_heads
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, len_q, n_heads*d_head)

        # Output layer
        return self.output(attn_output),attn

# def set_pad_emb(embs):
#     if '[PAD]' in embs:

class EmbeddingLayer(nn.Module):
    PRECISION2DTYPE = {
        32: torch.float,
        64: torch.double,
        16: torch.half
    }
    def __init__(self, initialization, vocab=None, precision=32, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.initialization = initialization
        self.dtype = self.PRECISION2DTYPE[precision]
        if initialization == 'gensim':
            self.embedding = GensimEmbedding(vocab,self.dtype,**kwargs)
        elif initialization == 'discocat':
            self.embedding = DiscocatEmbedding(vocab,self.dtype,**kwargs)
        elif initialization == 'random':
            self.embedding = RandomEmbedding(vocab,self.dtype,**kwargs)
        elif initialization == 'chunk':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'chunk_pretrain':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'pretrain':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'chunk_pretrain_emb':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'chunk_pretrain_emb_wdrop':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'chunk_pretrain_emb_only':
            self.embedding = PreAndChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)
        elif initialization == 'attn_chunk':
            self.embedding = AttnChunkEmbedding(vocab,self.dtype,subtype=initialization,**kwargs)

    def forward(self, input,*args,**kwargs):
        output = self.embedding(input,*args,**kwargs)
        return output

class AttnChunkEmbedding(nn.Module):
    def __init__(self,vocab, dtype=torch.float, subtype='cls_chunk_pretrain', model_name='bert-base-uncased',dim=512, \
                 freeze=False, pad_token='[PAD]',cls_token = '[CLS]',sep_token = '[SEP]'):
        super(AttnChunkEmbedding, self).__init__()
        self.dtype = dtype
        self.subtype = subtype
        self.dim = dim
        self.model_name = model_name
        self.freeze = freeze
        # print(self.name)
        # self.dtype = dtype
        self.vocab = vocab
        self.initial_emb()
        print("len vocab ",len(vocab), vocab[self.cls_token])
    
    def inittial_emb(self):
        if self.subtype in ('chunk_pretrain','pretrain'):
            self.pretrain_model = AutoModel.from_pretrained(self.model_name)
            for param in self.pretrain_model.parameters():
                param.requires_grad = not self.freeze



class PreAndChunkEmbedding(nn.Module):
    def __init__(self,vocab, dtype=torch.float, subtype='chunk', model_name='bert-base-uncased',dim=512, \
                 freeze=False, way='sum',learnable_weight = False, init_weights='',pretrain_emb=None, norm=True, pad_token='[PAD]',cls_token = '[CLS]',sep_token = '[SEP]',sent_emb = None,sent_imp = None,dropout=None, use_random_weights = False):
        super(PreAndChunkEmbedding, self).__init__()
        self.dtype = dtype
        self.subtype = subtype
        self.dim = dim
        self.model_name = model_name
        self.freeze = freeze
        # print(self.name)
        # self.dtype = dtype
        self.vocab = vocab
        self.way = way
        self.learnable_weight = learnable_weight
        self.init_weights = init_weights
        self.norm = norm
        self.pretrain_emb = pretrain_emb
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.sent_emb = sent_emb
        self.sent_imp = sent_imp
        self.dropout = dropout
        self.use_random_weights = use_random_weights
        self.initial_emb()
        print("len vocab ",len(vocab), vocab[self.cls_token])

    def initial_emb(self):
        if self.subtype == 'chunk':
            self.embs = nn.Embedding(num_embeddings = len(self.vocab),embedding_dim = self.dim,\
            dtype=self.dtype,padding_idx=self.vocab[self.pad_token])
        elif self.subtype in ('chunk_pretrain','pretrain'):
            self.pretrain_model = AutoModel.from_pretrained(self.model_name)
            for param in self.pretrain_model.parameters():
                param.requires_grad = not self.freeze
        elif self.subtype == 'chunk_pretrain_emb':
            pass
        elif self.subtype == 'chunk_pretrain_emb_wdrop':
            self.dropout = nn.Dropout(self.dropout)
        elif self.subtype == 'chunk_pretrain_emb_only':
            pretrain_model = AutoModel.from_pretrained(self.model_name)
            self.pretrain_emb = pretrain_model.embeddings

        self.emb_dim = self.dim
        if self.learnable_weight in ('init_kp_weight_randomly','init_kp_weight_randomly2'):
            # 0 for pad, 1 for non kp token, 2 for kp token
            self.kp_bias_emb = nn.Embedding(num_embeddings = 3,embedding_dim = 1,\
            dtype=self.dtype, padding_idx=0)
            #init_weights[0] is for non kp tokens, init_weights[1] is for kp tokens
            self.init_weights = [float(i) for i in self.init_weights.split('_')] 
        elif self.learnable_weight == 'init_specific_kp_weight':
            self.init_weights = [float(i) for i in self.init_weights.split('_')] 
            emb_tensor = torch.tensor([[0.],[self.init_weights[0]],[self.init_weights[1]]])
            self.kp_bias_emb = nn.Embedding.from_pretrained(emb_tensor, freeze=False,padding_idx=0)
            #init_weights[0] is for non kp tokens, init_weights[1] is for kp tokens
        
        if self.sent_imp in ('elu_gate','textrank_elu_gate'):
            self.sent_imp_w = torch.nn.Linear(self.emb_dim, 1, bias=True, device=None, dtype=None)

        if self.way == 'attn_cls':
            self.attn_emb = MultiHeadAttention(8, 768, dropout = 0.1, bias = True, pe_type = "None")
            self.attn_cls = nn.Embedding(1, 768)
        elif self.way == 'attn_cls_cat_sum':
            self.attn_emb = MultiHeadAttention(8, 768, dropout = 0.1, bias = True, pe_type = "None")
            self.attn_cls = nn.Embedding(1, 768)
            self.cat_reduce = torch.nn.Linear(self.emb_dim*2, self.emb_dim, bias=True, device=None, dtype=None)
        
        if self.use_random_weights == True:
            weight_emb = torch.rand((len(self.vocab), 1), requires_grad=False)
            self.weight_emb = nn.Embedding.from_pretrained(weight_emb,freeze=True)

    def forward(self, input_ids, attention_mask=None, kp_token_weights=None, map_ids=None, sent_map_ids=None,sentence_textrank_scores=None,**kwargs):
        # print("input_ids ",input_ids.shape)
        # print("...  len ",len(input_ids),len(map_ids),len(sent_map_ids))
        if self.subtype == 'pretrain':
            # print("input_ids ",input_ids.shape)
            output = self.pretrain_model(input_ids=input_ids,attention_mask=attention_mask)
            output = output['last_hidden_state']
            # print("output emb ",output.shape)
            return output, attention_mask
        
        map_len = torch.unique(map_ids, sorted=True, return_inverse=False, return_counts=True, dim=None)[1]
        ret_tensor = torch.zeros((map_len.shape[0], torch.max(map_len)+2, self.emb_dim)).cuda() # +2 for CLS and SEP
        ret_mask = torch.zeros((map_len.shape[0], torch.max(map_len)+2)).cuda() # +2 for CLS and SEP
        cls_tensor = torch.tensor([self.vocab[self.cls_token]]*map_len.shape[0]).cuda()
        sep_tensor = torch.tensor([self.vocab[self.sep_token]]*map_len.shape[0]).cuda()
        # print("sep_tensor ",sep_tensor.shape, input_ids.shape, input_ids[0], self.pretrain_emb)

        if self.subtype == 'chunk_pretrain':
            output = self.pretrain_model(input_ids=input_ids,attention_mask=attention_mask)
            output = output['last_hidden_state']
            cls_tensor = self.pretrain_model.get_input_embeddings()(cls_tensor)
            sep_tensor = self.pretrain_model.get_input_embeddings()(sep_tensor)
        elif self.subtype in ('chunk_pretrain_emb','chunk_pretrain_emb_only','chunk_pretrain_emb_wdrop'):
            # the reference code 
            # Here we only get the original word embeddings, leave the rest of processing of the BERT in the call of BERT forward function
            output = self.pretrain_emb.word_embeddings(input_ids)
            cls_tensor = self.pretrain_emb.word_embeddings(cls_tensor)
            sep_tensor = self.pretrain_emb.word_embeddings(sep_tensor)
            if self.subtype == 'chunk_pretrain_emb_wdrop':
                output = self.dropout(output)
                cls_tensor = self.dropout(cls_tensor)
                sep_tensor = self.dropout(sep_tensor)
            # print("sep_tensor 222 ",sep_tensor.shape)
        elif self.subtype == 'chunk':
            output = self.embs(input_ids)
            cls_tensor = self.embs(cls_tensor)
            sep_tensor = self.embs(sep_tensor)
        
        if self.learnable_weight in ('init_specific_kp_weight','init_kp_weight_randomly','init_kp_weight_randomly2'):
            # print("kp_token_weights[0] 1 ",kp_token_weights.shape,kp_token_weights[0])
            # print("self.init_weights ",self.init_weights,kp_token_weights[0]==self.init_weights[0])
            real_kp_token_ids = torch.zeros(kp_token_weights.shape, dtype = torch.long).cuda()
            real_kp_token_ids[kp_token_weights == self.init_weights[0]] = 1
            real_kp_token_ids[kp_token_weights == self.init_weights[1]] = 2
            # print("real_kp_token_ids ",real_kp_token_ids)
            real_kp_token_weights = self.kp_bias_emb(real_kp_token_ids).squeeze(2)
            # print("self.kp_bias_emb ",self.kp_bias_emb.weight,self.kp_bias_emb.weight.requires_grad)
            if self.learnable_weight in ('init_kp_weight_randomly2'):
                real_kp_token_weights = real_kp_token_weights*real_kp_token_weights
        elif self.use_random_weights == True:
            real_kp_token_weights = self.weight_emb(input_ids).reshape(input_ids.shape)
            # print(input_ids.shape, real_kp_token_weights.shape)
        else:
            real_kp_token_weights = kp_token_weights

        if self.norm == True:
            # before_real_kp_token_weights = real_kp_token_weights.detach()
            # sum_real_kp_token_weights = torch.sum(real_kp_token_weights,dim=1).unsqueeze(1)
   
            real_kp_token_weights = real_kp_token_weights/(torch.sum(real_kp_token_weights,dim=1).unsqueeze(1))
            # if torch.isnan(real_kp_token_weights).any():
            #     print("real_kp_token_weights shape ",real_kp_token_weights.shape,sum_real_kp_token_weights.shape)
            #     index = torch.where(torch.isnan(real_kp_token_weights)==True)
            #     print("sum_real_kp_token_weights ",sum_real_kp_token_weights[index[0][0]])
            #     print("before_real_kp_token_weights before sum ",before_real_kp_token_weights[index[0][0]])
        real_kp_token_weights = real_kp_token_weights.unsqueeze(-1)
        # print('self.way ',self.way)
        if self.way == 'sum':
            weighted_output = (output*real_kp_token_weights).sum(dim=1)
        elif self.way == 'elu_sum':
            weighted_output = (torch.nn.functional.elu(output*real_kp_token_weights)).sum(dim=1)
        elif self.way == 'mean':
            weighted_output = (output*real_kp_token_weights).mean(dim=1)
        elif self.way == "attn_cls":
            attn_cls_emb = self.attn_cls(torch.tensor([0]*output.shape[0]).cuda())
            __input = torch.cat([attn_cls_emb.view(output.shape[0],1,-1),output],dim=1)
            output,_ = self.attn_emb(__input,__input,__input)
            weighted_output = output[:,0,:].reshape(output.shape[0],-1)
        elif self.way == 'attn_cls_cat_sum':
            attn_cls_emb = self.attn_cls(torch.tensor([0]*output.shape[0]).cuda())
            __input = torch.cat([attn_cls_emb.view(output.shape[0],1,-1),output],dim=1)
            attn_output,_ = self.attn_emb(__input,__input,__input)
            weighted_output_1 = attn_output[:,0,:].reshape(output.shape[0],-1)
            weighted_output_2 = (output*real_kp_token_weights).sum(dim=1)
            weighted_output = self.cat_reduce(torch.cat([weighted_output_1,weighted_output_2],dim=1))


        # print("weighted_output  ",weighted_output.shape)

        cum_len = 0  
        last_cum_len = 0
        # print("sent_map_ids ",sent_map_ids,self.sent_emb)
        if sent_map_ids is not None and self.sent_emb is not None:
            if self.sent_emb in ('sep','mean'):
                if self.sent_emb == 'mean':
                    mean_kp_token_weights = torch.ones(real_kp_token_weights.shape).cuda()
                    mean_kp_token_weights[real_kp_token_weights==0] = 0
                    mean_kp_token_weights_sum = mean_kp_token_weights.sum(dim=1)
                    output_for_mean_sent_emb = (output*mean_kp_token_weights).sum(dim=1)
                    # print("output_for_mean_sent_emb ",output_for_mean_sent_emb.shape)
                for i, l in enumerate(map_len): 
                    # print("sent_map_ids[i] ",len(map_len),l,len(sent_map_ids[i]),sent_map_ids[i])
                    last_sent_id = sent_map_ids[i][0]
                    inserted_sent_emb = 0
                    last_insert_pos = 1
                    for chunk_id, sent_id in enumerate(sent_map_ids[i]):
                        # print("cum_len ",last_sent_id,sent_id,inserted_sent_emb, cum_len)
                        ret_tensor[i, 1 + inserted_sent_emb + chunk_id] = weighted_output[cum_len] 
                        cum_len += 1
                        # print("sent_id ",last_sent_id,sent_id)
                        if sent_id != last_sent_id:
                            last_sent_id = sent_id
                            inserted_sent_emb += 1
                            insert_pos = 1 + inserted_sent_emb + chunk_id
                            # print("sent_emb ",self.sent_emb)
                            if self.sent_emb == 'sep':
                                sent_emb = self.pretrain_emb.word_embeddings(torch.tensor(self.vocab[self.sep_token]).cuda())
                            else:
                                sent_emb = output_for_mean_sent_emb[last_cum_len:cum_len,:]
                                # print("sent_emb ",sent_emb.shape)
                                sent_emb = sent_emb.sum(dim=0)/mean_kp_token_weights_sum[last_cum_len:cum_len].sum()  
                                # print(sent_emb.shape)
                            if self.sent_imp == 'elu_gate': 
                                importamce = torch.nn.functional.elu(self.sent_imp_w(sent_emb))
                                sent_emb = importamce*sent_emb
                            elif self.sent_imp == 'textrank_elu_gate':
                                sent_emb = sentence_textrank_scores[inserted_sent_emb-1]*sent_emb
                                importamce = torch.nn.functional.elu(self.sent_imp_w(sent_emb))
                                sent_emb = importamce*sent_emb
                            ret_tensor[i, insert_pos] = sent_emb
                            last_cum_len = cum_len
                            last_insert_pos = insert_pos
                    inserted_sent_emb += 1
                    insert_pos = 1 + inserted_sent_emb + chunk_id
                    if self.sent_emb == 'sep':
                        sent_emb = self.pretrain_emb.word_embeddings(torch.tensor(self.vocab[self.sep_token]).cuda())
                    else:
                        sent_emb = output_for_mean_sent_emb[last_cum_len:cum_len].sum(dim=0)/mean_kp_token_weights_sum[last_cum_len:cum_len].sum()  
                    if self.sent_imp == 'elu_gate': 
                        importamce = torch.nn.functional.elu(self.sent_imp_w(sent_emb))
                        sent_emb = importamce*sent_emb
                    elif self.sent_imp == 'textrank_elu_gate':
                        sent_emb = sentence_textrank_scores[inserted_sent_emb-1]*sent_emb
                        importamce = torch.nn.functional.elu(self.sent_imp_w(sent_emb))
                        sent_emb = importamce*sent_emb
                    ret_tensor[i, insert_pos] = sent_emb
                    last_insert_pos = insert_pos
                    # print("inserted_sent_emb ",inserted_sent_emb)
                    ret_tensor[i,l+1] = sep_tensor[i]
                    ret_tensor[i,0] = cls_tensor[i]
                    ret_mask[i, :l+1] = 1
                    ret_mask[i, l+1] = 1
        else:
            for i, l in enumerate(map_len): 
                ret_tensor[i, 1:l+1] = weighted_output[cum_len: cum_len+l] 
                ret_tensor[i,l+1] = sep_tensor[i]
                ret_tensor[i,0] = cls_tensor[i]
                ret_mask[i, :l+1] = 1
                ret_mask[i, l+1] = 1
                cum_len += l 
        # print("ret_tensor ",ret_tensor.shape)
        # concat cls and sep embedding
        return ret_tensor, ret_mask
 

class PreEmbedding(nn.Module):
    def __init__(self,vocab, model_name='bert-base-uncased', dim=512, freeze=False):
        super(PreEmbedding, self).__init__()
        self.dim = dim
        self.model_name = model_name
        self.freeze = freeze
        # print(self.name)
        # self.dtype = dtype
        self.vocab = vocab
        self.initial_emb(self.model_name)

    def initial_emb(self,model_name):
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        print("self.freeze ",self.freeze)
        if self.freeze in [True,False]:
            for param in self.pretrain_model.parameters():
                param.requires_grad = not self.freeze
        self.emb_dim = self.dim

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # print("input_ids ",input_ids.shape)
        output = self.pretrain_model(input_ids=input_ids,attention_mask=attention_mask)
        output = output['last_hidden_state']
        # print("output emb ",output.shape)
        return output, attention_mask

class ChunkPreEmbedding(nn.Module):
    def __init__(self,vocab, model_name='bert-base-uncased',dim=512, freeze=False, way='sum',learnable_weight = False, \
                 init_weights=''):
        super(ChunkPreEmbedding, self).__init__()
        self.dim = dim
        self.model_name = model_name
        self.freeze = freeze
        # print(self.name)
        # self.dtype = dtype
        self.vocab = vocab
        self.way = way
        self.learnable_weight = learnable_weight
        self.init_weights = init_weights
        self.initial_emb(self.model_name)
        print("len vocab ",len(vocab), vocab['[CLS]'])

    def initial_emb(self,model_name):
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        if self.freeze in [True,False]:
            for param in self.pretrain_model.parameters():
                param.requires_grad = not self.freeze
        self.emb_dim = self.dim
        if self.learnable_weight == 'init_kp_weight':
            # 0 for pad, 1 for non kp token, 2 for kp token
            self.kp_bias_emb = nn.Embedding(num_embeddings = 3,embedding_dim = 1,\
            dtype=self.dtype, padding_idx=0)
            #init_weights[0] is for non kp tokens, init_weights[1] is for kp tokens
            self.init_weights = [float(i) for i in self.init_weights.split('_')]  

    def forward(self, input_ids, attention_mask=None, kp_token_weights=None, map_ids=None, **kwargs):
        # print("input_ids ",input_ids.shape)
        output = self.pretrain_model(input_ids=input_ids,attention_mask=attention_mask)
        output = output['last_hidden_state']
        # print("output emb ",output.shape)
        if self.learnable_weight == 'init_kp_weight':
            real_kp_token_ids = torch.zeros(kp_token_weights.shape, dtype = torch.long)
            real_kp_token_ids[kp_token_weights == self.init_weights[0]] == 1
            real_kp_token_ids[kp_token_weights == self.init_weights[1]] == 2
            kp_token_weights = self.kp_bias_emb(real_kp_token_ids)

        kp_token_weights = kp_token_weights/(torch.sum(kp_token_weights,dim=1).unsqueeze(1))
        kp_token_weights = kp_token_weights.unsqueeze(-1)
        # print('self.way ',self.way)
        if self.way == 'sum':
            output = (output*kp_token_weights).sum(dim=1)
        elif self.way == 'mean':
            output = (output*kp_token_weights).mean(dim=1)
        # print("output mean ",output.shape)
        map_len = torch.unique(map_ids, sorted=True, return_inverse=False, return_counts=True, dim=None)[1]
        ret_tensor = torch.zeros((map_len.shape[0], torch.max(map_len)+2, output.shape[-1])).cuda() # +2 for CLS and SEP
        ret_mask = torch.zeros((map_len.shape[0], torch.max(map_len)+2)).cuda() # +2 for CLS and SEP
        cls_tensor = torch.tensor([self.vocab['[CLS]']]*map_len.shape[0]).cuda()
        cls_tensor = self.pretrain_model.get_input_embeddings()(cls_tensor)
        sep_tensor = torch.tensor([self.vocab['[SEP]']]*map_len.shape[0]).cuda()
        sep_tensor = self.pretrain_model.get_input_embeddings()(sep_tensor)
        cum_len = 0  
        for i, l in enumerate(map_len): 
            ret_tensor[i, 1:l+1] = output[cum_len: cum_len+l] 
            ret_tensor[i,l+1] = sep_tensor[i]
            ret_tensor[i,0] = cls_tensor[i]
            ret_mask[i, :l+1] = 1
            ret_mask[i, l+1] = 1
            cum_len += l 
        
        # concat cls and sep embedding
        return ret_tensor, ret_mask

class ChunkEmbedding(nn.Module):
    def __init__(self,vocab, dtype=torch.float, dim=512, freeze=False, way='sum', learnable_weight = False, \
                 init_weights=''):
        super(ChunkEmbedding, self).__init__()
        self.dim = dim
        self.freeze = freeze
        # print(self.name)
        self.dtype = dtype
        self.initial_emb(vocab)
        self.vocab = vocab
        self.way = way
        self.learnable_weight = learnable_weight
        self.init_weights = init_weights

    def initial_emb(self,vocab):
        self.embs = nn.Embedding(num_embeddings = len(vocab),embedding_dim = self.dim,\
            dtype=self.dtype,padding_idx=vocab['[PAD]'])
        self.emb_dim = self.dim
        if self.learnable_weight == 'init_kp_weight':
            # 0 for pad, 1 for non kp token, 2 for kp token
            self.kp_bias_emb = nn.Embedding(num_embeddings = 3,embedding_dim = 1,\
            dtype=self.dtype, padding_idx=0)
            #init_weights[0] is for non kp tokens, init_weights[1] is for kp tokens
            self.init_weights = [float(i) for i in self.init_weights.split('_')]  

    def forward(self, input_ids,kp_token_weights=None,map_ids=None, **kwargs):
        # print("input_ids ",input_ids.shape)
        output = self.embs(input_ids)
        # print("output emb ",output.shape)
        if self.learnable_weight == 'init_kp_weight':
            real_kp_token_ids = torch.zeros(kp_token_weights.shape, dtype = torch.long)
            real_kp_token_ids[kp_token_weights == self.init_weights[0]] == 1
            real_kp_token_ids[kp_token_weights == self.init_weights[1]] == 2
            kp_token_weights = self.kp_bias_emb(real_kp_token_ids)

        kp_token_weights = kp_token_weights/(torch.sum(kp_token_weights,dim=1).unsqueeze(1))
        kp_token_weights = kp_token_weights.unsqueeze(-1)
        if self.way == 'sum':
            output = (output*kp_token_weights).sum(dim=1)
        elif self.way == 'mean':
            output = (output*kp_token_weights).mean(dim=1)
        # print("output mean ",output.shape)
        map_len = torch.unique(map_ids, sorted=True, return_inverse=False, return_counts=True, dim=None)[1]
        ret_tensor = torch.zeros((map_len.shape[0], torch.max(map_len)+2, output.shape[-1])).cuda() # +2 for CLS and SEP
        ret_mask = torch.zeros((map_len.shape[0], torch.max(map_len)+2)).cuda() # +2 for CLS and SEP
        cls_tensor = torch.tensor([self.vocab['[CLS]']]*map_len.shape[0]).cuda()
        cls_tensor = self.embs(cls_tensor)
        sep_tensor = torch.tensor([self.vocab['[SEP]']]*map_len.shape[0]).cuda()
        sep_tensor = self.embs(sep_tensor)
        cum_len = 0  
        for i, l in enumerate(map_len): 
            ret_tensor[i, 1:l+1] = output[cum_len: cum_len+l] 
            ret_tensor[i,l+1] = sep_tensor[i]
            ret_tensor[i,0] = cls_tensor[i]
            ret_mask[i, :l+1] = 1
            ret_mask[i, l+1] = 1
            cum_len += l 
        
        # concat cls and sep embedding
        return ret_tensor, ret_mask

class RandomEmbedding(nn.Module):
    def __init__(self,vocab, dtype=torch.float, dim=512, freeze=False):
        super(RandomEmbedding, self).__init__()
        self.dim = dim
        self.freeze = freeze
        # print(self.name)
        self.dtype = dtype
        self.initial_emb(vocab)

    def initial_emb(self,vocab):
        self.embs = nn.Embedding(num_embeddings = len(vocab),embedding_dim = self.dim,\
            dtype=self.dtype,padding_idx=vocab['[PAD]'])
        self.emb_dim = self.dim


    def forward(self, input):
        output = self.embs(input)
        return output


class DiscocatEmbedding(nn.Module):
# ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 
# 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 
# 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 
# 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 
# '__testing_word2vec-matrix-synopsis']
    emb_path = 'results/analysis_discocat/'
    def __init__(self,vocab, dtype=torch.float, name='offenseval_all', suffix='without_w2v', mode='cat', freeze=True):
        '''
        mode: abs: module of the complex; cat: concatenate real part and image part
        '''
        super(DiscocatEmbedding, self).__init__()
        self.name = name
        self.freeze = freeze
        self.add_tokens = []
        self.d_embs = {}
        self.suffix = suffix
        self.mode = mode
        self.dtype = dtype
        if self.mode == 'abs':
            self.vector_size = 256 
        elif self.mode == 'cat':
            self.vector_size = 512 
        self.c = 0
        self.initial_emb(vocab)

    def initial_emb(self,vocab):
        file_path = os.path.join(self.emb_path,self.name+'_embs.pickle')
        print('emb file_path',file_path)
        with open(file_path, 'rb') as f:
            d_embs = pickle.load(f)
            self.d_embs = d_embs['{}_{}'.format(self.name,self.suffix)]
        extra_embs = {}
        # print('complete load file')
        for box_name, embedding in self.d_embs.items():
            words = box_name.split('_')[0]
            words = words.split()
            for word in words:
                if word not in extra_embs:
                    extra_embs[word] = [embedding]
                else:
                    extra_embs[word].append(embedding)
                # print(word, embedding)
        for word, embeddings in extra_embs.items():
            extra_embs[word] = np.array(embeddings).mean(axis = 0).tolist()
        self.d_embs.update(extra_embs)
        print('self.d_embs.keys()',self.d_embs.keys())
        ids = list(vocab.values())
        ids.sort()
        id2token = {}
        embs = []
        print('vocab', len(vocab), vocab)
        # the vocab contains two kinds of tokens, the first is word_garmmar, come from the sentences that 
        # sucessfully transformed from string to diagram. The second is word only, come from the sentences
        # that failed to be transformed to diagram or bending noun. The sencod token will use the average 
        # embedding of all its grammar embeddings or unk. For example, love_n = [1,2], love_v = [3,4], and 
        # love = [2,3]
        for word,id in vocab.items():
            id2token[id] = word
        # print(ids)
        for id in ids:
            word = id2token[id]
            embs.append(self.get_discocat_emb(word))
        # print(embs[0])
        print('self.c  ',self.c)
        embs = np.array(embs)
        embs[vocab['[PAD]']] = 0
        self.embs = nn.Embedding.from_pretrained(torch.tensor(embs,dtype=self.dtype),freeze=self.freeze,padding_idx=vocab['[PAD]'])
        self.emb_dim = self.embs.embedding_dim


    def get_discocat_emb(self,word):
        try:
            if self.mode == 'abs':
                return np.abs(np.array(self.d_embs[word]))
            elif self.mode == 'cat':
                nparray = np.array(self.d_embs[word])
                return np.concatenate((np.real(nparray),np.imag(nparray)))
        except:
            self.c +=1
            return np.random.uniform(-0.5,+0.5,self.vector_size)


    def forward(self, input):
        output = self.embs(input)
        return output
        
class GensimEmbedding(nn.Module):
# ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 
# 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 
# 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 
# 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 
# '__testing_word2vec-matrix-synopsis']
    def __init__(self,vocab, dtype=torch.float, name='glove-wiki-gigaword-50', freeze=True):
        super(GensimEmbedding, self).__init__()
        self.name = name
        self.freeze = freeze
        self.add_tokens = []
        # print(self.name)
        self.dtype = dtype
        self.gensim_embedder = gensim.downloader.load(self.name)
        self.c = 0
        self.initial_emb(vocab)
        

    def initial_emb(self,vocab):
        ids = list(vocab.values())
        ids.sort()
        id2token = {}
        embs = []
        for word,id in vocab.items():
            id2token[id] = word
        for id in ids:
            word = id2token[id]
            embs.append(self.get_gensim_emb(word))
        print('self.c  ',self.c)
        embs = np.array(embs)
        embs[vocab['[PAD]']] = 0
        self.embs = nn.Embedding.from_pretrained(torch.tensor(embs,dtype=self.dtype),freeze=self.freeze,padding_idx=vocab['[PAD]'])
        self.emb_dim = self.embs.embedding_dim

    def get_gensim_emb(self,word):
        try:
            return self.gensim_embedder.get_vector(word)
        except:
            self.c += 1
            return np.random.uniform(-0.5,+0.5,self.gensim_embedder.vector_size)


    def forward(self, input):
        output = self.embs(input)
        return output

