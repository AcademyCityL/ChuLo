# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import AutoTokenizer,BertForSequenceClassification,AutoModelForSequenceClassification,BertGenerationConfig, BertGenerationDecoder,LongformerConfig, T5ForTokenClassification, T5Config, AutoModelForTokenClassification, T5ForSequenceClassification
from customlayers.embedding import EmbeddingLayer
from tools.tokenizer import get_tokenizer
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import inspect
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import torch.nn as nn
import types
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert_generation.modeling_bert_generation import BertGenerationOnlyLMHead
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from copy import deepcopy


_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype = torch.float32, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = (1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(dtype).to(device) / self.dim))).to(dtype)
        # print("inv_freq ",inv_freq.dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # print("freqs ",freqs.dtype)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q,position_ids_k, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
    sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
    sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed
def _upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        # print(f" key_layer {key_layer.shape}, attention_mask {attention_mask.shape}, query_length {query_length}")
        # key_layer torch.Size([1, 16, 16, 64]), attention_mask torch.Size([1, 1, 16, 16]), query_length 16 
        # print(attention_mask)
        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
        # print(f"_upad_input attention_mask {attention_mask.shape}")
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # print(f"_upad_input indices_k {indices_k.shape}, cu_seqlens_k {cu_seqlens_k.shape}, max_seqlen_in_batch_k {max_seqlen_in_batch_k}")

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # print(f"input hidden_states {type(hidden_states)}, attention_mask {type(attention_mask)}, head_mask {type(head_mask)}, encoder_hidden_states {type(encoder_hidden_states)}, encoder_attention_mask {type(encoder_attention_mask)}")
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        # print(f"is_cross_attention {is_cross_attention}, past_key_value {past_key_value}")
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            # attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            # attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        kv_seq_len = key_layer.shape[-2]
        q_seq_len = query_layer.shape[-2]

        use_cache = past_key_value is not None
        # print(f"kv_seq_len {kv_seq_len}, q_seq_len {q_seq_len}, use_cache {use_cache}")
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
        # print("value_layer ",value_layer.dtype)
        # cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
        # position_ids_q = torch.arange(q_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
        # position_ids_k = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
        # query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids_q, position_ids_k)
        # print("cos.dtype,query_layer.dtype ",cos.dtype,query_layer.dtype)
        softmax_scale = math.sqrt(self.attention_head_size)
        # print(f"sliding_window {self.sliding_window}, query_layer {query_layer.shape}, key_layer {key_layer.shape}")
        if self.sliding_window is not None:
            # in this case, q_seq_len == kv_seq_len actually
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            position_ids_q = torch.arange(q_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
            position_ids_k = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids_q, position_ids_k)

            batch_size = query_layer.shape[0]
            query_layer = query_layer.transpose(1, 2).to(torch.bfloat16)
            key_layer = key_layer.transpose(1, 2).to(torch.bfloat16)
            value_layer = value_layer.transpose(1, 2).to(torch.bfloat16)
            # print(f"in sw 1, query_states {query_layer.shape}, key_states {key_layer.shape}, value_states {value_layer.shape}")
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                query_layer, key_layer, value_layer, attention_mask, q_seq_len
            )
            # print(f"in sw, query_states {query_states.shape}, key_states {key_states.shape}, value_states {value_states.shape}")
            # print(f"in sw, indices_q {indices_q}, cu_seq_lens {cu_seq_lens}, max_seq_lens {max_seq_lens}")
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
           
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=self.dropout.p,
                softmax_scale=softmax_scale,
                causal=self.causal,
                window_size=self.sliding_window,
            )
            # print(f"in sw, attn_output_unpad {attn_output_unpad.shape}, {attn_output_unpad.dtype}")
            context_layer = pad_input(attn_output_unpad, indices_q, batch_size, q_seq_len).to(self.value.weight.dtype)
            # print(f"in sw, context_layer {context_layer.shape}, {context_layer.dtype}")

            # context_layer = value_layer.permute(0, 2, 1, 3)
            # context_layer = hidden_states
        else:
            if kv_seq_len > q_seq_len:
                cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            else:
                cos, sin = self.rotary_emb(query_layer, seq_len=q_seq_len)
            position_ids_q = torch.arange(q_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
            position_ids_k = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids_q, position_ids_k)

            # time.sleep(5)
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # print(f"in no sw, {query_layer.dtype}, key_layer {key_layer.dtype}")
            attention_scores = attention_scores / softmax_scale
            # print(attention_scores.dtype, encoder_attention_mask.dtype)
            # print(f"in no sw, attention_scores {attention_scores.shape}, encoder_attention_mask {encoder_attention_mask.shape}")
            # Apply the attention mask is (precomputed for all layers in BertGenerationModel forward() function
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertGenerationModel forward() function)
                attention_scores = attention_scores + encoder_attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # print(f"in no sw, attention_probs {attention_probs.shape}, value_layer {value_layer.shape}")
            context_layer = torch.matmul(attention_probs, value_layer)
            # print(f"in no sw, context_layer {context_layer.dtype}, {value_layer.device}, {context_layer.device},{attention_probs.device},{encoder_attention_mask.device}")

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # print(f"context_layer {context_layer.shape}")
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

def forward_encoder(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`: `1` for
            tokens that are NOT MASKED, `0` for MASKED tokens.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        # if attention_mask is None:
        #     attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask = None
        # if not use_cache:
        #     extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # attention_mask -> left-padding 2d mask for slide window attention (when use slide attention only)
        # attention_mask -> no left-padding (for global attention when using longformer self attn)
        # encoder_attention_mask -> right padding 4d mask for cross attention

        encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask,dtype = embedding_output.dtype, tgt_len=attention_mask.shape[1])
        if self.left_padding == True:
            attention_mask = 1 - attention_mask
        # print("encoder_attention_mask, ",encoder_attention_mask)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

def forward_emb(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    return inputs_embeds

def _pad_to_window_size(
        self, # LongformerSelfAttention obj
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = hidden_states.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # this path should be recorded in the ONNX export, it is fine with padding_len == 0 as well
        if padding_len > 0:
            # logger.warning_once(
            #     f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
            #     f"`config.attention_window`: {attention_window}"
            # )
            input_ids_padding = hidden_states.new_full(
                (batch_size, padding_len),
                self.config.pad_token_id,
                dtype=torch.long,
            )

            hidden_states_padding = self.embeddings(input_ids_padding)
            hidden_states = torch.cat([hidden_states, hidden_states_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens

        return padding_len, attention_mask, hidden_states

def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
    # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
    # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
    # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
    if attention_mask is not None:
        attention_mask = attention_mask * (global_attention_mask + 1)
    else:
        # simply use `global_attention_mask` as `attention_mask`
        # if no `attention_mask` is given
        attention_mask = global_attention_mask + 1
    return attention_mask

def forward_long_self(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False):
    
    # we need to pad the hidden_states in order to use the longformer's sliding window attention, like what they do in LongformerModel._pad_to_window_size
    # print("hidden_states ",hidden_states.shape)
    # pad the inputs
    padding_len, attention_mask, hidden_states = _pad_to_window_size(self,hidden_states,attention_mask)

    # get extended attention mask for longformer self attention
    # print("attention_mask 1",attention_mask)
    # print("attention_mask 1",attention_mask[0])
    extended_attention_mask = attention_mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min
    # print("extended_attention_mask ",extended_attention_mask)
    extended_attention_mask = extended_attention_mask[:, 0, 0, :]
    # print("attention_mask ",attention_mask)
    # print("attention_mask 2",attention_mask)
    # print("attention_mask 2",attention_mask[0])
    # import sys
    # sys.exit(0)
    

    is_index_masked = extended_attention_mask < 0
    is_index_global_attn = extended_attention_mask > 0

    # Record `is_global_attn == True` to enable ONNX export
    is_global_attn = is_index_global_attn.flatten().any().item()

    outputs,key_vectors, value_vectors = _longformer_self_forward(self, hidden_states,attention_mask=extended_attention_mask,layer_head_mask=None,is_index_masked = is_index_masked,is_index_global_attn = is_index_global_attn,is_global_attn=is_global_attn, output_attentions=False)
    hidden_states = outputs[0]
    # unpadding
    hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]

    return (hidden_states, (key_vectors, value_vectors,))

def _longformer_self_forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention

        copied from longformer self attn module, just add rotary embedding
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # add rotary embedding
        # in this case, q_seq_len == kv_seq_len actually
        # print("query_vectors ",query_vectors.shape, key_vectors.shape, value_vectors.shape )
        # query_vectorsuery_vectors  torch.Size([8, 512, 12, 64]) torch.Size([8, 512, 12, 64]) torch.Size([512, 8, 768])

        cos, sin = self.rotary_emb(value_vectors, seq_len=seq_len)
        position_ids_q = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
        position_ids_k = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(1, -1)
        query_vectors, key_vectors = apply_rotary_pos_emb(query_vectors.transpose(1,2), key_vectors.transpose(1,2), cos, sin, position_ids_q, position_ids_k)
        query_vectors = query_vectors.transpose(1,2)
        key_vectors = key_vectors.transpose(1,2)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs, key_vectors, value_vectors

class BERT(nn.Module):
    save_path = 'models/'
    def __init__(self, config,params):
        super(BERT, self).__init__()
        self.global_config = config
        model_config = self.global_config['MODEL']

        self.model_name = model_config.get('model_name','bert-base-uncased')
        self.freeze = model_config.get('freeze',False)
        self.num_labels = model_config.get('num_labels',2)
        self.embedding_params = model_config.get('embedding',{})
        self.attn_mode = model_config.get('attn_mode',{'name':'default','param1':0})
        self.hidden_dropout_prob = model_config.get('hidden_dropout_prob', 0.1)
        data_config = self.global_config['DATA']
        self.tokenizer_name = data_config.get('tokenizer_name','whitespace')
        self.tokenizer_real_name = 'results/cache/tokenizers/{}_{}/'.format(self.global_config['DATA']['dataset_name'],\
                                                                    self.tokenizer_name.replace('/','_'))

        self.decoder_cfg = model_config.get('decoder_cfg',None)
        self.decoder = None

        if 'longformer' in self.model_name or 'bigbird' in self.model_name :
            self.token_cls = model_config.get('token_cls',False)
            self.token_cls_conll = model_config.get('token_cls_conll',False)
            init_class = AutoModelForTokenClassification if self.token_cls == True else AutoModelForSequenceClassification
            try:
                self.model = init_class.from_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')),num_labels=self.num_labels,hidden_dropout_prob=self.hidden_dropout_prob,gradient_checkpointing=True)
            except:
                self.model = init_class.from_pretrained(self.model_name,num_labels=self.num_labels,hidden_dropout_prob=self.hidden_dropout_prob,gradient_checkpointing=True)
                self.model.save_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')), from_pt=True)
        elif 't5' in self.model_name:
            self.token_cls = model_config.get('token_cls',False)
            T5cfg = T5Config.from_pretrained(self.model_name)
            T5cfg.gradient_checkpointing = True
            T5cfg.num_labels = self.num_labels
            init_class = T5ForTokenClassification if self.token_cls == True else T5ForTokenClassification
            try:
                self.model = init_class.from_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')),config=T5cfg)
            except:
                self.model = init_class.from_pretrained(self.model_name,config=T5cfg)
                self.model.save_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')), from_pt=True)
        else:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')),num_labels=self.num_labels,hidden_dropout_prob=self.hidden_dropout_prob)
            except:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,num_labels=self.num_labels,hidden_dropout_prob=self.hidden_dropout_prob)
                self.model.save_pretrained('{}/{}'.format(self.save_path,self.model_name.replace('/','_')), from_pt=True)
        # print(self.model)
        self.tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
        
        if self.embedding_params['initialization']  == 'original':
            pass
        else:
            if 'bert' in self.model_name and 'roberta' not in self.model_name:
                pre_emb = self.model.bert.embeddings
            elif 'roberta' in self.model_name:
                pre_emb = self.model.roberta.embeddings
            elif 'longformer' in self.model_name:
                pre_emb = self.model.longformer.embeddings
            self.embeddinglayer = EmbeddingLayer(self.embedding_params['initialization'], vocab=self.tokenizer.get_vocab(),\
                    **self.embedding_params['kwargs'],pretrain_emb=pre_emb,pad_token = self.tokenizer.pad_token,cls_token = self.tokenizer.cls_token,sep_token = self.tokenizer.sep_token)

        if self.decoder_cfg is not None:
            if self.decoder_cfg['selfattn'] == 'slide_win':
                self.init_decoder_sw(self.decoder_cfg)
            elif self.decoder_cfg['selfattn'] == 'global_and_slide_win':
                self.init_decoder_g_sw(self.decoder_cfg)
        
        for param in self.model.parameters():
            param.requires_grad = not self.freeze
        # print(self.named_parameters)

    def init_decoder_g_sw(self, decoder_cfg):
        self.model = self.model.bert
        config = BertGenerationConfig.from_pretrained("google-bert/bert-base-uncased")
        config.is_decoder = True
        config.add_cross_attention = decoder_cfg['add_cross_attention']
        config.num_hidden_layers = decoder_cfg['hidden_layer']
        max_pos_emb = decoder_cfg['dec_max_seq_len']
        model_d = BertGenerationDecoder.from_pretrained("google-bert/bert-base-uncased", config=config)
        config.vocab_size = self.num_labels
        model_d.lm_head = BertGenerationOnlyLMHead(config)
        head_dim = int(config.hidden_size/config.num_attention_heads)

        model_d.bert.embeddings.forward = types.MethodType(forward_emb, model_d.bert.embeddings)
        model_d.bert.forward = types.MethodType(forward_encoder, model_d.bert)
        setattr(model_d.bert,"left_padding", False)

        l_config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
        l_config.attention_window = [512]
        l_config.pad_token_id = config.pad_token_id
        for bert_layer in model_d.bert.encoder.layer:
            l_self = LongformerSelfAttention(l_config, 0)
            l_self.query.load_state_dict(deepcopy(bert_layer.attention.self.query.state_dict()))
            l_self.key.load_state_dict(deepcopy(bert_layer.attention.self.key.state_dict()))
            l_self.value.load_state_dict(deepcopy(bert_layer.attention.self.value.state_dict()))
            # to use the pretrained parameters, we share the parameters to the global projection
            l_self.query_global.load_state_dict(deepcopy(bert_layer.attention.self.query.state_dict()))
            l_self.key_global.load_state_dict(deepcopy(bert_layer.attention.self.key.state_dict()))
            l_self.value_global.load_state_dict(deepcopy(bert_layer.attention.self.value.state_dict()))
            setattr(l_self,"embeddings",model_d.bert.embeddings)
            l_self.forward = types.MethodType(forward_long_self, l_self)
            setattr(l_self, "rotary_emb", MixtralRotaryEmbedding(head_dim,max_position_embeddings=max_pos_emb,base=10000,device = 'cuda'))
            bert_layer.attention.self = l_self

            setattr(bert_layer.crossattention.self, "sliding_window", None)
            setattr(bert_layer.crossattention.self, "rotary_emb", MixtralRotaryEmbedding(head_dim,max_position_embeddings=max_pos_emb,base=10000,device = 'cuda'))
            # bert_layer.attention.self.forward = types.MethodType(forward, bert_layer.attention.self)
            bert_layer.crossattention.self.forward = types.MethodType(forward, bert_layer.crossattention.self)
        self.decoder = model_d

    def init_decoder_sw(self, decoder_cfg):
        self.model = self.model.bert
        config = BertGenerationConfig.from_pretrained("google-bert/bert-base-uncased")
        config.is_decoder = True
        config.add_cross_attention = decoder_cfg['add_cross_attention']
        config.num_hidden_layers = decoder_cfg['hidden_layer']
        max_pos_emb = decoder_cfg['dec_max_seq_len']
        causal = decoder_cfg['causal']
        model_d = BertGenerationDecoder.from_pretrained("google-bert/bert-base-uncased", config=config)
        config.vocab_size = self.num_labels
        model_d.lm_head = BertGenerationOnlyLMHead(config)
        head_dim = int(config.hidden_size/config.num_attention_heads)

        model_d.bert.embeddings.forward = types.MethodType(forward_emb, model_d.bert.embeddings)
        model_d.bert.forward = types.MethodType(forward_encoder, model_d.bert)
        setattr(model_d.bert,"left_padding", True)

        for bert_layer in model_d.bert.encoder.layer:
            setattr(bert_layer.attention.self, "sliding_window", decoder_cfg['sliding_win'])
            setattr(bert_layer.attention.self, "causal", causal)
            setattr(bert_layer.attention.self, "rotary_emb", MixtralRotaryEmbedding(head_dim,max_position_embeddings=max_pos_emb,base=10000,device = 'cuda'))
            setattr(bert_layer.crossattention.self, "sliding_window", None)
            setattr(bert_layer.crossattention.self, "rotary_emb", MixtralRotaryEmbedding(head_dim,max_position_embeddings=max_pos_emb,base=10000,device = 'cuda'))
            bert_layer.attention.self.forward = types.MethodType(forward, bert_layer.attention.self)
            bert_layer.crossattention.self.forward = types.MethodType(forward, bert_layer.crossattention.self)
        self.decoder = model_d

    def forward(self, input_ids, attention_mask=None, max_chunk_len = None,kp_token_weights=None,map_ids=None,special_tokens_mask=None,sent_map_ids = None,sentence_textrank_scores=None,o_input_ids=None,o_attention_mask=None,o_token_labels=None,):
        enc_output_hidden = self.decoder is not None 
        enc_output_attention =  self.decoder is None 

        if self.embedding_params['initialization'] == 'original':
            if 'longformer' in self.model_name:
                # print("longformer input check ",input_ids.shape,o_input_ids.shape)
                if self.token_cls == False:
                    global_attention_mask = torch.zeros_like(input_ids)
                    global_attention_mask[:, 0] = 1
                    out = self.model(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,output_attentions=True)
                elif self.token_cls_conll == True:
                    global_attention_mask = torch.zeros_like(input_ids)
                    out = self.model(input_ids =input_ids, attention_mask = attention_mask,global_attention_mask=global_attention_mask,output_attentions=True)
                else:
                    global_attention_mask = torch.zeros_like(o_input_ids)
                    global_attention_mask[:, 0] = 1 # init global attention mask
                    out = self.model(input_ids =o_input_ids, attention_mask = o_attention_mask,global_attention_mask=global_attention_mask,output_attentions=True)
                logits, attn = out['logits'], out['attentions'][-1]
                return logits, attn
            else:
                if self.token_cls == False:
                    out = self.model(input_ids=input_ids,attention_mask=attention_mask,output_attentions=True)
                elif self.token_cls_conll == True:
                    out = self.model(input_ids=input_ids,attention_mask=attention_mask,output_attentions=True)
                    logits, attn = out['logits'], out['attentions'][-1]
                    return logits, attn
                else:
                    out = self.model(input_ids =o_input_ids, attention_mask = o_attention_mask,output_attentions=True)
                    logits, attn = out['logits'], out['attentions'][-1]
                    return logits, attn
        else:
            if self.attn_mode['name'] in ('key_phrase_chunk_rep','key_phrase_chunk_rep2') or self.embedding_params['initialization'] == 'pretrain':
                out,attention_mask = self.embeddinglayer(input_ids,kp_token_weights=kp_token_weights,\
                                                        map_ids=map_ids,attention_mask=attention_mask,sent_map_ids=sent_map_ids,sentence_textrank_scores=sentence_textrank_scores)
                # out: [bs, sen_len, emb_dim]
                # print("out ",out.shape,attention_mask.shape)
            else:
                out = self.embeddinglayer(input_ids)# out: [bs, sen_len, emb_dim]
            # print("out 1",out.shape)
            # print("inputs_embeds ",out.shape)
            if 'longformer' in self.model_name:
                # print("out ",out.shape,input_ids.shape,attention_mask.shape)
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1
                out = self.model(inputs_embeds=out,attention_mask=attention_mask,global_attention_mask=global_attention_mask,output_attentions=True)
                logits, attn = out['logits'], out['attentions'][-1]
                return logits, attn
            else:
                out = self.model(inputs_embeds=out,attention_mask=attention_mask,output_attentions=enc_output_attention,output_hidden_states = enc_output_hidden)

        # print(self.decoder )
        if self.decoder is None:
            logits, attn = out['logits'], out['attentions'][-1]
        else:
            # print("o_attention_mask ",o_attention_mask)
            # print("o_attention_mask  0", o_attention_mask[0])
            # print("o_attention_mask  sum",o_attention_mask[0].sum())
            out = self.decoder(input_ids =o_input_ids, attention_mask = o_attention_mask, encoder_hidden_states = out['last_hidden_state'], encoder_attention_mask=attention_mask, use_cache=False)
            logits, attn = out['logits'], None
            # print("logits ",logits.shape)
            
        # print("out 2",out.shape)
        return logits, attn