import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
import copy

import math
from customlayers.embedding import EmbeddingLayer
from tools.tokenizer import get_tokenizer
'''
The implementation is modified according to the implementation from labml
Todo: different dimension of q,k,v in multiheadattention module
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


class TransformerLayer(nn.Module):
    """
    <a id="TransformerLayer"></a>

    ## Transformer Layer

    This can act as an encoder layer or a decoder layer.

    🗒 Some implementations, including the paper seem to have differences
    in where the layer-normalization is done.
    Here we do a layer normalization before attention and feed-forward networks,
    and add the original residual vectors.
    Alternative is to do a layer normalization after adding the residuals.
    But we found this to be less stable when training.
    We found a detailed discussion about this in the paper
     [On Layer Normalization in the Transformer Architecture](https://papers.labml.ai/paper/2002.04745).
    """

    def __init__(self,n_heads, d_model,dim_feedforward, cross_attn = False, pre_norm = True, dropout = 0.1, bias = True, \
                 pe_type = 'absolute_sin'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.dropout = dropout
        self.bias = bias
        self.pe_type = pe_type
        self.self_attn = MultiHeadAttention(self.n_heads,self.d_model,self.dropout,self.bias,self.pe_type)
        self.self_attn_dropout = nn.Dropout(dropout)
        if cross_attn == True:
            self.cross_attn = MultiHeadAttention()
            self.cross_attn_dropout = nn.Dropout(dropout)
        else:
            self.cross_attn = None
            self.cross_attn_dropout = None

        # the feed_forward is copied from Pytorch transformer module _ff_block function
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, self.d_model),
            nn.Dropout(self.dropout),
        )
        
        self.norm_self_attn = nn.LayerNorm(d_model)
        if self.cross_attn is not None:
            self.norm_cross_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    '''
    for fixed_token_length, global attention is CLS + All LOCs + SEP
    '''
    def _sa_block_fixed_token_length_all_locs(self, x, mask, attn_mode):
        # global attention for [CLS](start position) and [SEP](end position) token
        cls = self.self_attn(x[:,0,:].unsqueeze(dim=1), x, x, mask)[0]
        # print("cls ",cls.shape,cls[0],key_padding_mask[0])
        
        # the last non-masked position should be SEP token
        # note the unsqueeze opsration in the transformer forward
        cm=mask.clone().detach()
        indices = torch.where(mask==1)
        cm[indices] = indices[1]
        cmindex = cm.max(dim=1)[1]
        sep_tokens = x[torch.arange(len(x)),cmindex,:].unsqueeze(dim=1)
        sep = self.self_attn(sep_tokens, x, x, mask)[0]
        # print("sep ",sep.shape,sep[0],key_padding_mask[0])
        # local attention for local blocks
        local_blocks = x[:,1:-1,:].reshape(-1, attn_mode['param1'], self.self_attn.d_model)
        local_blocks_mask = mask[:,1:-1].reshape(-1, attn_mode['param1'])
        # print("local_blocks ",local_blocks.shape,local_blocks_mask.shape,x.shape,attn_mode)
        '''
        when extracting local_blocks and local_blocks_mask, there are many padding-position tokens extracted, which will
        result in the padding-fullfilled local blocks and masks, like [pad_token,pad_token,...,pad_token] and [pad_mask,
        pad_mask,...,pad_mask]. Those tensors will result in the NA value later. To avoid this situation, we set the first
        position of each local_block_mask to be not masked. This operation won't impact the final result because in the 
        later computation, these positions will be masked as usual.
        '''
        local_blocks_mask[:,0] = 1 # set the first position of each local_block_mask to be not masked
        # note modify local_blocks_mask won't impact mask
        local_blocks, attn_weights = self.self_attn(local_blocks, local_blocks, local_blocks, local_blocks_mask)
        # '''
        # To count the imp_ids, we need to reduce the blocks only containing masked positions first
        # '''
        # sumed_attn = torch.sum(attn_weights, dim = 1)
        # imp_ids = torch.argmax(sumed_attn, dim = 1,keepdim=False)
        # # print("local_blocks_mask ",local_blocks_mask.shape)
        # stat_list = imp_ids.tolist()
        # tmp = []
        # for i in range(len(local_blocks_mask)):
        #     if torch.sum(local_blocks_mask[i]==False) > 1: # not all masked or only the first position is not masked
        #         tmp.append(stat_list[i])
        # stat_list = tmp
        # print("important ids ",Counter(stat_list),np.mean(stat_list),np.std(stat_list))
        # print("local_blocks ",local_blocks.shape,local_blocks[0],local_blocks_mask.shape,local_blocks_mask[0])
        if torch.isnan(local_blocks).any():
            print("attn_weights ",attn_weights)
            print("local_blocks_mask ",local_blocks_mask)
            print("mask ",mask)
            # idsss = torch.where(torch.isnan(local_blocks))
            # print(idsss)
            # print(local_blocks[idsss[0][0],idsss[1][0],idsss[2][0]])
            # print(local_blocks[idsss[0][0],idsss[1][0]])
            # print(local_blocks[idsss[0][0],idsss[2][0]])
            # print(local_blocks_mask[idsss[0][0]-1])
            # print(local_blocks_mask[idsss[0][0]])
            # print(local_blocks_mask[idsss[0][0]+1])
            # print(mask[idsss[0][0]-1])
            # print(mask[idsss[0][0]])
            # print(mask[idsss[0][0]+1])
            import sys
            sys.exit()
        # global attention for all [CLS] (start), [SEP] (end) and [LOC] tokens
        local_tokens = local_blocks[:,0,:].reshape(len(x),-1,self.self_attn.d_model)
        indices = [0] + [i for i in range(1, len(mask[0])-2, attn_mode['param1'])] + [len(mask[0])-1]
        global_tokens = torch.cat([cls, local_tokens, sep], dim=1)
        global_tokens_mask = mask.index_select(dim=1, index=torch.tensor(indices, device=mask.device))
        # print("ori_kpd_mask ",ori_kpd_mask.shape,ori_kpd_mask[indices[0], indices[1]].shape,global_tokens_mask.shape)
        global_tokens = self.self_attn(global_tokens, global_tokens, global_tokens,global_tokens_mask)[0]
        # print("global_tokens ",global_tokens.shape,global_tokens[0],global_tokens_mask[0])
        if torch.isnan(global_tokens).any():
            import sys
            sys.exit()
        cls = global_tokens[:,0,:].unsqueeze(dim=1)
        sep = global_tokens[:,-1,:].unsqueeze(dim=1)
        local_tokens = global_tokens[:,1:-1,:].reshape(len(local_blocks),1,self.self_attn.d_model)
        local_blocks = torch.cat([local_tokens, local_blocks[:,1:,:]], dim=1).reshape(len(x),-1,self.self_attn.d_model)
        x = torch.cat([cls, local_blocks, sep], dim=1)
        return self.self_attn_dropout(x),None
    '''
    for fixed_token_length, global attention is CLS + All important words (imps) + SEP
    '''
    def _sa_block_fixed_token_length_all_imps(self, x, mask, attn_mode):
        # global attention for [CLS](start position) and [SEP](end position) token
        cls = self.self_attn(x[:,0,:].unsqueeze(dim=1), x, x, mask)[0]
        # the last non-masked position should be SEP token
        cm=mask.clone().detach()
        indices = torch.where(mask==1)
        cm[indices] = indices[1]
        cmindex = cm.max(dim=1)[1]
        sep_tokens = x[torch.arange(len(x)),cmindex,:].unsqueeze(dim=1)
        sep = self.self_attn(sep_tokens, x, x, mask)[0]
        
        # local attention for local blocks
        local_blocks = x[:,1:-1,:].reshape(-1, attn_mode['param1'], self.self_attn.d_model)
        local_blocks_mask = mask[:,1:-1].reshape(-1, attn_mode['param1'])
        local_blocks_mask[:,0] = 1 # set the first position of each local_block_mask to be not masked
        # note modify local_blocks_mask won't impact mask

        local_blocks, attn_weights = self.self_attn(local_blocks, local_blocks, local_blocks, local_blocks_mask)

        # global attention for all [CLS] (start), [SEP] (end) and important tokens
        # important tokens are the tokens with the highest attended weights in each local block
        sumed_attn = torch.sum(attn_weights, dim = 1)
        imp_ids = torch.argmax(sumed_attn, dim = 1,keepdim=False)
        # '''
        # To count the imp_ids, we need to reduce the blocks only containing masked positions first
        # '''
        # # print("local_blocks_mask ",local_blocks_mask.shape)
        # stat_list = imp_ids.tolist()
        # tmp = []
        # for i in range(len(local_blocks_mask)):
        #     if torch.sum(local_blocks_mask[i]==False) > 1: # not all masked or only the first position is not masked
        #         tmp.append(stat_list[i])
        # stat_list = tmp
        # print("important ids ",Counter(stat_list),np.mean(stat_list),np.std(stat_list))
        imp_tokens = local_blocks[torch.arange(len(imp_ids)), imp_ids, :].reshape(len(x),\
                                                                                    -1,self.self_attn.d_model)
        imp_masks = local_blocks_mask[torch.arange(len(imp_ids)), imp_ids].reshape(len(x),-1)
        global_tokens = torch.cat([cls, imp_tokens, sep], dim=1)
        # print("ori_kpd_mask ",ori_kpd_mask.shape,imp_masks.shape)
        global_tokens_mask = torch.cat([mask[:,0].reshape(-1,1),imp_masks,mask[:,-1].reshape(-1,1)], dim=1)
       
        global_tokens = self.self_attn(global_tokens, global_tokens, global_tokens,global_tokens_mask)[0]
                      
        cls = global_tokens[:,0,:].unsqueeze(dim=1)
        sep = global_tokens[:,-1,:].unsqueeze(dim=1)
        local_tokens = global_tokens[:,1:-1,:].reshape(len(local_blocks),self.self_attn.d_model)
        local_blocks[torch.arange(len(imp_ids)), imp_ids, :] = local_tokens
        # print("local_tokens",local_tokens.shape, local_blocks.shape)
        # local_blocks = torch.cat([local_tokens, local_blocks[:,1:,:]], dim=1).reshape(len(x),-1,self.self_attn.d_model)
        # print("local blocks ",local_blocks.shape,cls.shape,sep.shape)
        x = torch.cat([cls, local_blocks.reshape(len(cls),-1,self.self_attn.d_model), sep], dim=1)
        return self.self_attn_dropout(x),None
        
    '''
    for fixed_token_length_wo_loc, global attention is CLS + All important words (imps) + SEP
    The progress of computing the attention is the same as fixed_token_length_all_imps
    '''
    def _sa_block_fixed_token_length_wo_loc_all_imps(self, x, mask, attn_mode):
        return self._sa_block_fixed_token_length_all_imps(x, mask, attn_mode)
    
    def _sa_block_key_phrase_split_all_locs(self, x, mask, attn_mode):
        return self._sa_block_fixed_token_length_all_locs(x, mask, attn_mode)
    
    def _sa_block_key_phrase_split_all_imps(self, x, mask, attn_mode):
        return self._sa_block_fixed_token_length_all_imps(x, mask, attn_mode)
    
    def _sa_block_key_phrase_split_wo_loc_all_imps(self, x, mask, attn_mode):
        return self._sa_block_fixed_token_length_all_imps(x, mask, attn_mode)
    
    # self-attention block
    def _sa_block(self, x, mask,attn_mode):
        x, attn = self.self_attn(query=x, key=x, value=x, mask=mask)
        return self.self_attn_dropout(x), attn

    # cross-attention block
    def _ca_block(self, x, src, mask, attn_mode):
        x, attn = self.cross_attn(query=x, key=src, value=src, mask=mask)
        return self.cross_attn_dropout(x), attn

    # feed forward block
    def _ff_block(self, x):
        return self.feed_forward(x)

    def forward(self, x, mask, attn_mode, src = None, src_mask = None):
        # print("attn_mode ",attn_mode)
        if attn_mode['name'] == 'default':
            pass
        elif attn_mode['name'] == 'fixed_token_length':
            if attn_mode['param2'] == 'all_locs':
                self._sa_block = self._sa_block_fixed_token_length_all_locs
            elif attn_mode['param2'] == 'all_imps':
                self._sa_block = self._sa_block_fixed_token_length_all_imps
        elif attn_mode['name'] in ('key_phrase_split','key_phrase_split2'):
            if attn_mode['param2'] == 'all_locs':
                self._sa_block = self._sa_block_key_phrase_split_all_locs
            elif attn_mode['param2'] == 'all_imps':
                self._sa_block = self._sa_block_key_phrase_split_all_imps
        elif attn_mode['name'] == 'fixed_token_length_wo_loc':
            if attn_mode['param2'] == 'all_imps':
                self._sa_block = self._sa_block_fixed_token_length_wo_loc_all_imps
        elif attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2'):
            pass

        if self.pre_norm == True:
            # print("x .shape",x.shape,self.norm_self_attn(x).shape)
            z, self_attn = self._sa_block(self.norm_self_attn(x), mask, attn_mode)
            x = x + z

            if src is not None:
                assert self.cross_attn is not None, "self.cross_attn is not None"
                z, cross_attn = self._ca_block(self.norm_cross_attn(x), src, src_mask, attn_mode)
                x = x + z

            x = x + self._ff_block(self.norm_ff(x))
        else:
            z, self_attn = self._sa_block(x,mask,attn_mode)
            x = self.norm_self_attn(x + z)

            if src is not None:
                assert self.cross_attn is not None, "self.cross_attn is not None"
                z, cross_attn = self._ca_block(x, src, src_mask,attn_mode)
                x = self.norm_cross_attn(x + z)

            x = self.norm_ff(x + self._ff_block(x))

        return x, self_attn # to modify later, return cross_attn as well if necessary


class Encoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model,dim_feedforward, cross_attn = False, pre_norm = True,\
                  dropout = 0.1, bias = True, pe_type = 'absolute_sin',final_norm = True):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.dropout = dropout
        self.bias = bias
        self.pe_type = pe_type
        self.cross_attn = cross_attn
        self.final_norm = final_norm

        layer = TransformerLayer(d_model = self.d_model, n_heads = self.n_heads, cross_attn=self.cross_attn, \
                                 pre_norm = self.pre_norm, pe_type=self.pe_type, bias = self.bias,\
                                    dim_feedforward = self.dim_feedforward, dropout = self.dropout)
        
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(self.n_layers)])

        # Final normalization layer
        if self.final_norm == True:
            self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x, mask, attn_mode=None):
        # Run through each transformer layer
        tmp_attn = None
        for layer in self.layers:
            x, self_attn = layer(x=x, mask=mask, attn_mode=attn_mode)
            if tmp_attn is None:
                tmp_attn = self_attn.detach()
            else:
                tmp_attn = tmp_attn + self_attn.detach()
        # Finally, normalize the vectors
        if self.final_norm == True:
            x = self.norm(x)
        return x, tmp_attn


class Decoder(nn.Module):

    def __init__(self, n_layers, n_heads, d_model,dim_feedforward, cross_attn = False, pre_norm = True,\
                  dropout = 0.1, bias = True, pe_type = 'absolute_sin'):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.dropout = dropout
        self.bias = bias
        self.pe_type = pe_type
        self.cross_attn = cross_attn


        layer = TransformerLayer(d_model = self.d_model, n_heads = self.n_heads, cross_attn=self.cross_attn, \
                                 pre_norm = self.pre_norm, pe_type=self.pe_type, bias = self.bias,\
                                    dim_feedforward = self.dim_feedforward, dropout = self.dropout)
        
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(self.n_layers)])

        # Final normalization layer
        if self.final_norm == True:
            self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x, mask, src, src_mask,attn_mode):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=mask, attn_mode = attn_mode, src=src, src_mask=src_mask,)
        # Finally, normalize the vectors
        if self.final_norm == True:
            x = self.norm(x)
        return x


class EncoderDecoder(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, src, src_mask):
        # Run the source through encoder
        enc = self.encode(src, src_mask)
        # Run encodings and targets through decoder
        return self.decode(x, mask, src, src_mask)

    def encode(self, x, mask):
        return self.encoder(self.src_embed(x), mask)

    def decode(self, x, mask, src, src_mask):
        return self.decoder(self.tgt_embed(x), mask, src, src_mask)
    

class Transformer_Encoder(nn.Module):
    def __init__(self,config,params):
        super(Transformer_Encoder, self).__init__()
        self.global_config = config
        self.params = params
        self.init_attr_from_config()
        self.init_model()

    def init_attr_from_config(self):
        # print("init_attr_from_config")
        model_config = self.global_config['MODEL']
        self.dim_feedforward = model_config.get('dim_feedforward',2048)
        self.layers = model_config.get('layers',6)
        self.head = model_config.get('head',8)
        self.dropout = model_config.get('dropout',0.1)
        self.embedding_params = model_config.get('embedding',{})
        self.d_model = model_config.get('d_model',512)
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})
        if self.embedding_params.get('learnable_weight', None) == 'init_kp_weight':
            assert 'init_weights' in self.embedding_params, 'No specify initial weight for key phrase tokens'
            assert self.embedding_params['init_weights'] == self.attn_mode['param3'], 'Inconsistent init weights'
        self.pe_type = model_config.get('pe_type',"absolute_sin")
        self.bias = model_config.get('bias',True)
        self.pre_norm = model_config.get('pre_norm', True)
        self.final_norm = model_config.get('final_norm', True)

        data_config = self.global_config['DATA']
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_real_name = 'results/cache/tokenizers/{}_{}/'.format(self.global_config['DATA']['dataset_name'],\
                                                                    self.tokenizer_name.replace('/','_'))

        self.max_seq_len = data_config.get('max_seq_len',128)
        self.chunk_vocab = self.params['daobj'].chunk_vocab if hasattr(self.params['daobj'],'chunk_vocab') else {}
    def init_model(self):
 
        self.tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
        self.embeddinglayer = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.tokenizer.get_vocab(),\
                **self.embedding_params['kwargs'],pad_token = self.tokenizer.pad_token,cls_token = self.tokenizer.cls_token,sep_token = self.tokenizer.sep_token)
        
        # print('init use_tr_tokenizer done')
        self.embedding_dim = self.embeddinglayer.embedding.emb_dim
        
        self.encoder = Encoder(n_layers = self.layers, d_model = self.d_model, n_heads = self.head, \
                                 pre_norm = self.pre_norm, pe_type=self.pe_type, bias = self.bias,\
                                    dim_feedforward = self.dim_feedforward, dropout = self.dropout, final_norm= self.final_norm)

        if self.pe_type == 'absolute_sin':
            self.pe = PositionalEncoding(d_model=self.embedding_dim, dropout=self.dropout, max_len=5000)
        
    def forward(self, input_ids, max_chunk_len = None, attention_mask=None, special_tokens_mask=None,\
                kp_token_weights=None,map_ids=None):
        # 0 mask, ~0 not mask in Huggingface
        # but 0=False, 1=True in Pytorch, and in pytorch transofmer, Ture is mask
        # attention_mask = (1-attention_mask).bool()
        # special_tokens_mask =(1-special_tokens_mask).bool()
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, sl]
        # special_tokens_mask = special_tokens_mask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, sl]
        # print(special_tokens_mask.shape,attention_mask.shape)
        # loc_index = torch.where(special_tokens_mask[0] == 1)[0]
        # loc_range = loc_index[2] - loc_index[1] # loc_index[1] is the start of the first block, loc_index[0] is [CLS]
        if self.attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2') or self.embedding_params['initialization'] == 'pretrain':
            out,attention_mask = self.embeddinglayer(input_ids,kp_token_weights=kp_token_weights,\
                                                     map_ids=map_ids,attention_mask=attention_mask)
            # out: [bs, sen_len, emb_dim]
            # print("out ",out.shape,attention_mask.shape)
        else:
            out = self.embeddinglayer(input_ids)# out: [bs, sen_len, emb_dim]
        if max_chunk_len is not None:
            self.attn_mode['param1'] = max_chunk_len
        if self.pe_type == 'absolute_sin':
            out = self.pe(out)
        # print("self.attn_mode ",self.attn_mode)
        out, attn = self.encoder(out, attention_mask, attn_mode = self.attn_mode)
        return out, attn
    