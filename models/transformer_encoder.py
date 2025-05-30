# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, List, Optional, Tuple, Union
import math
import numpy as np
from torch.autograd import Variable
from customlayers.embedding import EmbeddingLayer
from tools.tokenizer import get_tokenizer
from torch.nn import TransformerEncoderLayer, MultiheadAttention, TransformerEncoder
from collections import Counter
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.functional import _mha_shape_check,_canonical_mask,_none_or_dtype,_in_projection_packed,_in_projection,pad,\
scaled_dot_product_attention, dropout, softmax, linear
'''
The implementation is from pytorch 2.0.1
Modify the part needed to be modified
'''

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    pe_type: str = 'absolute_sin',
    rel: list = [],
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not nedeeded.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    '''
    TO support relpe, set need_weights = True
    '''
    need_weights = True
    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

        if pe_type == 'relative_pos':
            # bs, heads, len_q, head_dim = q.shape
            # _, _, len_k, _ = k.shape
            # print('q shape',q.shape)
            # q2 = q.permute(2,0,1,3).reshape(len_q, bs*heads, head_dim)
            q2 = q.transpose(0,1) # (len_q, bs*heads, head_dim)
            rel_weight = torch.matmul(q2, rel[0].transpose(1, 2)).transpose(0, 1)
            # rel_weight = rel_weight.contiguous().view(bs, heads, len_q, len_k)/math.sqrt(d_k)
            rel_weight = rel_weight.contiguous().view(bsz, num_heads, tgt_len, src_len)/math.sqrt(E)
            attn_logits += rel_weight

        attn_output_weights = softmax(attn_output_weights, dim=-1)

        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        if pe_type == 'relative_pos':
            # rel_weight = attention.permute(2, 0, 1, 3).contiguous().reshape(len_q, bs*heads, len_k)
            rel_weight = attn_output_weights.permute(2, 0, 1, 3).contiguous().reshape(tgt_len, bsz*num_heads, src_len)
            rel_weight = torch.matmul(rel_weight, rel[1]).transpose(0, 1)
            # rel_weight = rel_weight.contiguous().view(bs, heads, len_q, head_dim)
            rel_weight = rel_weight.contiguous().view(bsz, num_heads, tgt_len, head_dim)
            # values += rel_weight
            attn_output += rel_weight

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
    
def multiheadattention_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if self.pe_type == 'relative_pos':
            seq_length = key.shape[0]
            rel = [self.rel_pe_k(seq_length,seq_length),self.rel_pe_v(seq_length,seq_length)]
        else:
            rel = []

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                pe_type = self.pe_type,
                rel = rel)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                pe_type = self.pe_type,
                rel = rel)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
        
def encoder_layer_forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            special_tokens_mask: Optional[Tensor] = None,
            attn_mode: dict = {'name':'default', 'param1':0},
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        ori_kpd_mask = src_key_padding_mask
        ori_spe_mask = special_tokens_mask
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # self-attention block
        '''
        for default
        '''
        def _sa_block_default(self, x: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], ori_spe_mask: Optional[Tensor], \
                        ori_kpd_mask: Optional[Tensor],
                        is_causal: bool = False) -> Tensor:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            return self.dropout1(x)
        '''
        for fixed_token_length, global attention is CLS + All LOCs + SEP
        '''
        def _sa_block_fixed_token_length_all_locs(self, x: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], ori_spe_mask: Optional[Tensor], \
                        ori_kpd_mask: Optional[Tensor],
                        is_causal: bool = False) -> Tensor:
            # global attention for [CLS](start position) and [SEP](end position) token
            cls = self.self_attn(x[:,0,:].unsqueeze(dim=1), x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            # print("cls ",cls.shape,cls[0],key_padding_mask[0])
            if torch.isnan(cls).any():
                import sys
                sys.exit()
            sep = self.self_attn(x[:,-1,:].unsqueeze(dim=1), x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            # print("sep ",sep.shape,sep[0],key_padding_mask[0])
            if torch.isnan(sep).any():
                import sys
                sys.exit()
            # local attention for local blocks
            local_blocks = x[:,1:-1,:].reshape(-1, attn_mode['param1'], self.self_attn.embed_dim)
            local_blocks_mask = ori_kpd_mask[:,1:-1].reshape(-1, attn_mode['param1'])
            '''
            when extracting local_blocks and local_blocks_mask, there are many padding-position tokens extracted, which will
            result in the padding-fullfilled local blocks and masks, like [pad_token,pad_token,...,pad_token] and [pad_mask,
            pad_mask,...,pad_mask]. Those tensors will result in the NA value later. To avoid this situation, we set the first
            position of each local_block_mask to be not masked. This operation won't impact the final result because in the 
            later computation, these positions will be masked as usual.
            '''
            local_blocks_mask[:,0] = False # set the first position of each local_block_mask to be not masked
            local_blocks_mask = F._canonical_mask(
                mask=local_blocks_mask,
                mask_name="src_key_padding_mask",
                other_type=F._none_or_dtype(src_mask),
                other_name="local_blocks_mask",
                target_type=src.dtype
            )
            local_blocks, attn_weights = self.self_attn(local_blocks, local_blocks, local_blocks,
                                          attn_mask=attn_mask,
                                key_padding_mask=local_blocks_mask,
                                need_weights=True, is_causal=is_causal)[0]
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
                idsss = torch.where(torch.isnan(local_blocks))
                print(idsss)
                print(local_blocks[idsss[0][0],idsss[1][0],idsss[2][0]])
                print(local_blocks[idsss[0][0],idsss[1][0]])
                print(local_blocks[idsss[0][0],idsss[2][0]])
                print(local_blocks_mask[idsss[0][0]-1])
                print(local_blocks_mask[idsss[0][0]])
                print(local_blocks_mask[idsss[0][0]+1])
                print(ori_kpd_mask[idsss[0][0]-1])
                print(ori_kpd_mask[idsss[0][0]])
                print(ori_kpd_mask[idsss[0][0]+1])
                import sys
                sys.exit()
            # global attention for all [CLS] (start), [SEP] (end) and [LOC] tokens
            local_tokens = local_blocks[:,0,:].reshape(len(x),-1,self.self_attn.embed_dim)
            indices = [0] + [i for i in range(1, len(ori_kpd_mask[0])-2, attn_mode['param1'])] + [len(ori_kpd_mask[0])-1]
            global_tokens = torch.cat([cls, local_tokens, sep], dim=1)
            global_tokens_mask = ori_kpd_mask.index_select(dim=1, index=torch.tensor(indices, device=ori_kpd_mask.device))
            global_tokens_mask = F._canonical_mask(
                mask=global_tokens_mask,
                mask_name="src_key_padding_mask",
                other_type=F._none_or_dtype(src_mask),
                other_name="global_tokens_mask",
                target_type=src.dtype
            )
            # print("ori_kpd_mask ",ori_kpd_mask.shape,ori_kpd_mask[indices[0], indices[1]].shape,global_tokens_mask.shape)
            global_tokens = self.self_attn(global_tokens, global_tokens, global_tokens,
                                attn_mask=attn_mask,
                                key_padding_mask=global_tokens_mask,
                                need_weights=False, is_causal=is_causal)[0]
            # print("global_tokens ",global_tokens.shape,global_tokens[0],global_tokens_mask[0])
            if torch.isnan(global_tokens).any():
                import sys
                sys.exit()
            cls = global_tokens[:,0,:].unsqueeze(dim=1)
            sep = global_tokens[:,-1,:].unsqueeze(dim=1)
            local_tokens = global_tokens[:,1:-1,:].reshape(len(local_blocks),1,self.self_attn.embed_dim)
            local_blocks = torch.cat([local_tokens, local_blocks[:,1:,:]], dim=1).reshape(len(x),-1,self.self_attn.embed_dim)
            x = torch.cat([cls, local_blocks, sep], dim=1)
            return self.dropout1(x)
        '''
        for fixed_token_length, global attention is CLS + All important words (imps) + SEP
        '''
        def _sa_block_fixed_token_length_all_imps(self, x: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], ori_spe_mask: Optional[Tensor], \
                        ori_kpd_mask: Optional[Tensor],
                        is_causal: bool = False) -> Tensor:
            # global attention for [CLS](start position) and [SEP](end position) token
            cls = self.self_attn(x[:,0,:].unsqueeze(dim=1), x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            sep = self.self_attn(x[:,-1,:].unsqueeze(dim=1), x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False, is_causal=is_causal)[0]
            
            # local attention for local blocks
            local_blocks = x[:,1:-1,:].reshape(-1, attn_mode['param1'], self.self_attn.embed_dim)
            local_blocks_mask = ori_kpd_mask[:,1:-1].reshape(-1, attn_mode['param1'])
            local_blocks_mask[:,0] = False # set the first position of each local_block_mask to be not masked
            local_blocks_mask = F._canonical_mask(
                mask=local_blocks_mask,
                mask_name="src_key_padding_mask",
                other_type=F._none_or_dtype(src_mask),
                other_name="local_blocks_mask",
                target_type=src.dtype
            )
            local_blocks, attn_weights = self.self_attn(local_blocks, local_blocks, local_blocks,
                                          attn_mask=attn_mask,
                                key_padding_mask=local_blocks_mask,
                                need_weights=True, is_causal=is_causal)

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
                                                                                      -1,self.self_attn.embed_dim)
            imp_masks = local_blocks_mask[torch.arange(len(imp_ids)), imp_ids].reshape(len(x),-1)
            global_tokens = torch.cat([cls, imp_tokens, sep], dim=1)
            # print("ori_kpd_mask ",ori_kpd_mask.shape,imp_masks.shape)
            global_tokens_mask = torch.cat([ori_kpd_mask[:,0].reshape(-1,1),imp_masks,ori_kpd_mask[:,-1].reshape(-1,1)], dim=1)
            global_tokens_mask = F._canonical_mask(
                mask=global_tokens_mask,
                mask_name="src_key_padding_mask",
                other_type=F._none_or_dtype(src_mask),
                other_name="global_tokens_mask",
                target_type=src.dtype
            )
            global_tokens = self.self_attn(global_tokens, global_tokens, global_tokens,
                                attn_mask=attn_mask,
                                key_padding_mask=global_tokens_mask,
                                need_weights=False, is_causal=is_causal)[0]
            cls = global_tokens[:,0,:].unsqueeze(dim=1)
            sep = global_tokens[:,-1,:].unsqueeze(dim=1)
            local_tokens = global_tokens[:,1:-1,:].reshape(len(local_blocks),self.self_attn.embed_dim)
            local_blocks[torch.arange(len(imp_ids)), imp_ids, :] = local_tokens
            # print("local_tokens",local_tokens.shape, local_blocks.shape)
            # local_blocks = torch.cat([local_tokens, local_blocks[:,1:,:]], dim=1).reshape(len(x),-1,self.self_attn.embed_dim)
            # print("local blocks ",local_blocks.shape,cls.shape,sep.shape)
            x = torch.cat([cls, local_blocks.reshape(len(cls),-1,self.self_attn.embed_dim), sep], dim=1)
            return self.dropout1(x)
        
        '''
        for fixed_token_length_wo_loc, global attention is CLS + All important words (imps) + SEP
        The progress of computing the attention is the same as fixed_token_length_all_imps
        '''
        _sa_block_fixed_token_length__wo_loc_all_imps = _sa_block_fixed_token_length_all_imps
        

        if attn_mode['name'] == 'default':
            _sa_block = _sa_block_default
        elif attn_mode['name'] == 'fixed_token_length':
            if attn_mode['param2'] == 'all_locs':
                _sa_block = _sa_block_fixed_token_length_all_locs
            elif attn_mode['param2'] == 'all_imps':
                _sa_block = _sa_block_fixed_token_length_all_imps
        elif attn_mode['name'] == 'fixed_token_length_wo_loc':
            if attn_mode['param2'] == 'all_imps':
                _sa_block = _sa_block_fixed_token_length__wo_loc_all_imps
        x = src
        if self.norm_first:
            x = x + _sa_block(self,self.norm1(x), src_mask, src_key_padding_mask,ori_spe_mask=ori_spe_mask, \
                              ori_kpd_mask=ori_kpd_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + _sa_block(self, x, src_mask, src_key_padding_mask,ori_spe_mask=ori_spe_mask, \
                                         ori_kpd_mask=ori_kpd_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

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
        self.pe_type = model_config.get('pe_type',"absolute_sin")

        data_config = self.global_config['DATA']
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_real_name = 'results/cache/tokenizers/{}_{}/'.format(self.global_config['DATA']['dataset_name'],\
                                                                    self.tokenizer_name)

        self.max_seq_len = data_config.get('max_seq_len',128)
        self.chunk_vocab = self.params['daobj'].chunk_vocab if hasattr(self.params['daobj'],'chunk_vocab') else {}
    def init_model(self):

        self.tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
        self.embeddinglayer = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.tokenizer.get_vocab(),\
                **self.embedding_params['kwargs'])
        
        # print('init use_tr_tokenizer done')
        self.embedding_dim = self.embeddinglayer.embedding.emb_dim
        # support custom modification
        TransformerEncoderLayer.forward = encoder_layer_forward
        MultiheadAttention.forward = multiheadattention_forward
        F.multi_head_attention_forward = multi_head_attention_forward

        self.layers = nn.ModuleList([TransformerEncoderLayer( d_model = self.d_model, nhead = self.head, \
        dim_feedforward = self.dim_feedforward, dropout = self.dropout, activation = 'relu',batch_first=True) for _ in range(self.layers)])
        for layer in self.layers:
                setattr(layer.self_attn,'pe_type',self.pe_type)
        if self.pe_type == 'absolute_sin':
            self.pe = PositionalEncoding(d_model=self.embedding_dim, dropout=self.dropout, max_len=5000)
        elif self.pe_type == 'relative_pos':
            for layer in self.layers:
                setattr(layer.self_attn,'rel_pe_k',RelativePosition(layer.self_attn.head_dim))
                setattr(layer.self_attn,'rel_pe_v',RelativePosition(layer.self_attn.head_dim))
        
    def forward(self, input_ids, attention_mask=None, special_tokens_mask=None):
        # 0 mask, ~0 not mask in Huggingface
        # but 0=False, 1=True in Pytorch, and in pytorch transofmer, Ture is mask
        attention_mask = (1-attention_mask).bool()
        special_tokens_mask =(1-special_tokens_mask).bool()
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, sl]
        # special_tokens_mask = special_tokens_mask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, sl]
        # print(special_tokens_mask.shape,attention_mask.shape)
        # loc_index = torch.where(special_tokens_mask[0] == 1)[0]
        # loc_range = loc_index[2] - loc_index[1] # loc_index[1] is the start of the first block, loc_index[0] is [CLS]
        out = self.embeddinglayer(input_ids)# out: [bs, sen_len, emb_dim]

        if self.pe_type == 'absolute_sin':
            out = self.pe(out)
        for layer in self.layers:
            out = layer(out,src_key_padding_mask=attention_mask,special_tokens_mask = special_tokens_mask,attn_mode = self.attn_mode)
        return out