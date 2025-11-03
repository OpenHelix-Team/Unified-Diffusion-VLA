# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/transformers/blob/52daf4ec768fb9ffe84a0c373834172a7c54aecc/src/transformers/models/llama/modeling_llama.py
#
""" PyTorch Emu3 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_emu3 import Emu3Config


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import sys
sys.path.append("/data/user/wsong890/user68/project/UniVLA")
sys.path.insert(0, Path(__file__).absolute().parents[4].as_posix())
from models.policy_head.noise_schedulers import FlowMatchingScheduler
import random
# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Emu3Config"

def attn_mask_image_blocks_bidir_after_second(
    sequence: torch.Tensor,          # [B, L], dtype=long
    boi_token_id: int = 151852,      # <BOI>
    eoi_token_id: int = 151853,      # <EOI>
    boa_token_id: int = 151844,      # <BOA>
    eoa_token_id: int = 151845,      # <EOA>
    pad_token_id: int = 151643,
    include_boundary_as_image: bool = True,
    return_bool_mask: bool = True,
    past_key_values_length: int = 0,
):
    """
    规则：
      1) 每个图像块（BOI/BOA..EOI/EOA，含边界可选）内双向；
      2) 块与块之间默认因果；
      3) 但从“第2个图像块”起，所有这些图像块之间互相双向；
      4) 永远不关注 PAD 列；其余（文本等）保持因果。
    """
    B, L = sequence.shape
    dev = sequence.device
    P = int(past_key_values_length)
    K = P + L

    # 标注 token 类别
    is_pad = (sequence == pad_token_id)
    is_boi = (sequence == boi_token_id) | (sequence == boa_token_id)
    is_eoi = (sequence == eoi_token_id) | (sequence == eoa_token_id)

    # 标出图像块区间
    cstart = torch.cumsum(is_boi, dim=1)
    cend   = torch.cumsum(is_eoi, dim=1)
    inside = (cstart > cend)
    is_image = inside | is_boi | is_eoi if include_boundary_as_image else inside

    # 给图像块编号：块内位置编号为其起始计数，其它位置为0
    img_block_id = torch.where(is_image, torch.cumsum(is_boi, dim=1), torch.zeros_like(sequence))

    # 基础因果掩码
    tril = torch.tril(torch.ones((L, L), dtype=torch.bool, device=dev))
    allowed_cur = tril[None, :, :].expand(B, L, L).clone()  # [B,L,L]

    # 不允许关注 PAD 列（列维）
    nonpad_cols = (~is_pad)[:, None, :]                     # [B,1,L]
    allowed_cur &= nonpad_cols

    # ① 块内双向：同一图像块内（block_id相等且>0）的行列设为可见
    same_block = (img_block_id[:, :, None] == img_block_id[:, None, :]) & (img_block_id[:, :, None] > 0)  # [B,L,L]
    allowed_cur = torch.where(same_block, nonpad_cols.expand(-1, L, -1), allowed_cur)

    # ② 第2个及之后图像块之间互相双向：
    ge2 = (img_block_id >= 3)
    cross_ge2 = ge2[:, :, None] & ge2[:, None, :]          # [B,L,L] 行/列均在第2块及之后
    allowed_cur = torch.where(cross_ge2, nonpad_cols.expand(-1, L, -1), allowed_cur)

    # 拼接 KV cache（如有）
    if P > 0:
        allowed_past = torch.ones((B, L, P), dtype=torch.bool, device=dev)  # 需要可再收紧
        allowed_full = torch.cat([allowed_past, allowed_cur], dim=-1)       # [B,L,K]
    else:
        allowed_full = allowed_cur

    mask4d = allowed_full.unsqueeze(1)                                       # [B,1,L,K]
    if return_bool_mask:
        return mask4d
    else:
        bias = (~mask4d).to(torch.float32)
        bias.masked_fill_(bias.bool(), float("-inf"))
        return bias


def build_blockwise_attn_mask(
    sequence: torch.Tensor,                     # [B, L]
    boi_token_id: int = 151852,
    eoi_token_id: int = 151853,
    boa_token_id: int = 151844,
    eoa_token_id: int = 151845,
    pad_token_id: int = 151643,
    include_boundary_as_image: bool = True,     # True: <BOI>/<EOI> 也算图像块
    return_bool_mask: bool = True,              # False: 返回加性 bias(允许=0/屏蔽=-inf)
    past_key_values_length: int = 0,            # <-- KV cache 长度（列前缀）
):
    B, L = sequence.shape
    device = sequence.device
    P = int(past_key_values_length)             # 过去的 key 列数
    K = P + L                                   # 总 key 列

    # 标注
    is_pad = (sequence == pad_token_id)
    is_boi = (sequence == boi_token_id) | (sequence == boa_token_id)
    is_eoi = (sequence == eoi_token_id) | (sequence == eoa_token_id)

    # 图像/动作块定位
    starts_cum = torch.cumsum(is_boi, dim=1)
    ends_cum   = torch.cumsum(is_eoi, dim=1)
    level = (starts_cum - ends_cum).clamp(min=0)
    is_image = (level > 0) | is_boi | is_eoi if include_boundary_as_image else (level > 0)
    is_text = ~is_image

    # 切块 + 块序号
    cur = is_image
    prev = torch.zeros_like(cur); prev[:, 1:] = cur[:, :-1]
    block_start = (cur != prev) | torch.zeros((B,1), dtype=torch.bool, device=device)
    block_id = torch.cumsum(block_start.to(torch.int32), dim=1)  # [B,L] 1-based

    # 块间因果（当前段内）
    query_block = block_id[:, :, None]   # [B,L,1]
    key_block   = block_id[:, None, :]   # [B,1,L]
    allowed_cur = (key_block <= query_block)      # [B,L,L]

    # 文本块内因果（当前段内）
    tril = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
    allowed_cur &= ((~is_text)[:, :, None] | tril[None, :, :])

    # 屏蔽当前段内的 PAD 列（列维度）
    allowed_cur &= ~is_pad[:, None, :]   # [B,L,L]

    # ===== 关键：拼接过去的列（KV cache）到左侧，得到 [B,L,K] =====
    if P > 0:
        # 过去列：对所有 query 允许（它们都属于“之前的块”）；如需更严格规则，可在这里加限制
        allowed_past = torch.ones((B, L, P), dtype=torch.bool, device=device)
        allowed_full = torch.cat([allowed_past, allowed_cur], dim=-1)  # [B,L,P+L]
    else:
        allowed_full = allowed_cur  # [B,L,L]

    # 如果你还想禁止“PAD 行”作为查询（较少见），可再： allowed_full[is_pad] = False

    mask4d = allowed_full.unsqueeze(1)  # [B,1,L,K]
    if return_bool_mask:
        return mask4d
    else:
        bias = (~mask4d).to(torch.float32)
        bias = bias.masked_fill(bias.bool(), float("-inf"))
        return bias


def delete_false_key_value(#删除后面错误的token
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
DynamicCache.delete_false_key_value = delete_false_key_value
# def delete_false_key_value(past_kv, num_of_false_tokens):
#     # 删除 past_kv 中每层的 key 和 value 中最后 num_of_false_tokens 个 token
#     for layer_idx in range(len(past_kv)):
#         past_kv[layer_idx] = (
#             past_kv[layer_idx][0][..., :-num_of_false_tokens, :],  # 删除 key_cache 中最后 num_of_false_tokens 个 token
#             past_kv[layer_idx][1][..., :-num_of_false_tokens, :]   # 删除 value_cache 中最后 num_of_false_tokens 个 token
#         )
#     return past_kv
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.emu3.modeling_emu3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class Emu3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Emu3RMSNorm)


class Emu3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 强制 float32，不依赖 arange(dtype=...)，规避环境污染
        t = torch.arange(self.max_seq_len_cached, device="cpu").float().to(device=device, dtype=torch.float32)
        # inv_freq = self.inv_freq.to(torch.float32).to(device=device)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)).to(torch.float32).to(device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # def _set_cos_sin_cache(self, seq_len, device, dtype):
    #     self.max_seq_len_cached = seq_len
    #     t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

    #     freqs = torch.outer(t, self.inv_freq)
    #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
    #     emb = torch.cat((freqs, freqs), dim=-1)
    #     self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    #     self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Emu3LinearScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Emu3DynamicNTKScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Emu3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Emu3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # modify here
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Emu3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Emu3LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Emu3DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Emu3FlashAttention2(Emu3Attention):
    """
    Emu3 flash attention module. This module inherits from `Emu3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Emu3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Emu3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Emu3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

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
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
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


class Emu3SdpaAttention(Emu3Attention):
    """
    Emu3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Emu3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Emu3Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Emu3Model is using Emu3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


EMU3_ATTENTION_CLASSES = {
    "eager": Emu3Attention,
    "flash_attention_2": Emu3FlashAttention2,
    "sdpa": Emu3SdpaAttention,
}


class Emu3DecoderLayer(nn.Module):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_dropout)
        self.self_attn = EMU3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


EMU3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3PreTrainedModel(PreTrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Emu3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3Model(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3DecoderLayer`]

    Args:
        config: Emu3Config
    """

    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout = nn.Dropout(config.attention_dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # print("config.vocab_size:",config.vocab_size)
        # print("self.padding_idx:",self.padding_idx)
        # print("config.hidden_size:",config.hidden_size)
        self.layers = nn.ModuleList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.use_blockwise_attn_mask = config.use_blockwise_attn_mask
        self.use_bidirectional_attn_mask = config.use_bidirectional_attn_mask
        print("use_blockwise_attn_mask:", self.use_blockwise_attn_mask)
        print("use_bidirectional_attn_mask:", self.use_bidirectional_attn_mask)
        self.norm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # print("input_ids:",input_ids)
            inputs_embeds = self.embed_tokens(input_ids)

        if self.use_blockwise_attn_mask:
            # print("use_blockwise_attn_mask")
            if attention_mask is not None and len(attention_mask.shape) == 4:
                attention_mask = attention_mask
            else:
                attention_mask = build_blockwise_attn_mask(
                    input_ids,
                    boi_token_id=151852,
                    eoi_token_id=151853,
                    pad_token_id=151643,
                    boa_token_id=151844,
                    eoa_token_id=151845,
                    include_boundary_as_image=True,
                    return_bool_mask=False,
                    past_key_values_length=past_key_values_length,
                ).to(dtype=inputs_embeds.dtype)
        elif self.use_bidirectional_attn_mask:
            if attention_mask is not None and len(attention_mask.shape) == 4:
                attention_mask = attention_mask
            else:
                attention_mask = attn_mask_image_blocks_bidir_after_second(
                    input_ids,
                    boi_token_id=151852,
                    eoi_token_id=151853,
                    pad_token_id=151643,
                    boa_token_id=151844,
                    eoa_token_id=151845,
                    include_boundary_as_image=True,
                    return_bool_mask=False,
                    past_key_values_length=past_key_values_length,
                ).to(dtype=inputs_embeds.dtype)
        elif self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = self.dropout(inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # print("type(next_cache):",type(next_cache))
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class Emu3ForCausalLM(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        >>> from transformers.generation.configuration_utils import GenerationConfig
        >>> from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
        >>> from transformers import Emu3Processor
        >>> from PIL import Image

        >>> model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> image_processor = AutoImageProcessor.from_pretrained(PATH_TO_CONVERTED_IMAGE_PROCESSER)
        >>> image_tokenizer = AutoModel.from_pretrained(PATH_TO_CONVERTED_TOKENIZER_WEIGHTS).eval()
        >>> processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        >>> # Generation
        >>> prompt = "An Emu in cartoon style, it is wearing sunglasses."

        >>> pos_inputs = processor(text=prompt, mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")
        >>> neg_inputs = processor(text="", mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")

        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     use_cache=True,
        >>>     eos_token_id=model.config.eos_token_id,
        >>>     pad_token_id=model.config.pad_token_id,
        >>>     max_new_tokens=40960,
        >>>     do_sample=True,
        >>>     top_k=2048,
        >>> )

        >>> h, w = pos_inputs.image_size[0]
        >>> constrained_fn = processor.build_prefix_constrained_fn(h, w)
        >>> logits_processor = LogitsProcessorList([
        >>>     UnbatchedClassifierFreeGuidanceLogitsProcessor(
        >>>         classifier_free_guidance, 
        >>>         model,
        >>>         unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
        >>>     ),
        >>>     PrefixConstrainedLogitsProcessor(
        >>>         constrained_fn,
        >>>         num_beams=1,
        >>>     ),
        >>> ])

        >>> outputs = model.generate(pos_inputs.input_ids.to("cuda:0"), GENERATION_CONFIG, logits_processor=logits_processor)
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> mm_list = processor.decode(outputs[0])

        >>> # Understanding
        >>> prompt = "Provide a one-sentence caption for the provided image."
        >>> image = Image.open(TEST_IMAGE_PATH)

        >>> inputs = processor(text=text, image=image, mode='U', padding_side="left", padding="longest", return_tensors="pt")
        >>> input_ids = inputs.input_ids.to("cuda:0")
        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     pad_token_id=tokenizer.pad_token_id,
        >>>     bos_token_id=tokenizer.bos_token_id,
        >>>     eos_token_id=tokenizer.eos_token_id,
        >>> )

        >>> outputs = model.generate(input_ids, GENERATION_CONFIG, max_new_tokens=100)
        >>> outputs = outputs[:, input_ids.shape[-1]:]
        >>> answer = processor.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class ActionProjector(nn.Module):
    def __init__(self, in_channels, dim):
        super(ActionProjector, self).__init__()
        # Initialize the linear layers W1, W2, W3
        self.W1 = nn.Linear(in_channels, dim)
        self.W2 = nn.Linear(dim + dim, dim)  # Concatenating 2 encodings (dim + dim)
        self.W3 = nn.Linear(dim, dim)
        self.nonlinearity = nn.SiLU()  # swish
        
        # Initialize the weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Use Xavier initialization for the linear layer weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        
        # Initialize the biases to zeros
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)
        if self.W3.bias is not None:
            nn.init.zeros_(self.W3.bias)

    def forward(self, x, tau):
        """
        Forward pass through the ActionProjector.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim)
            tau (torch.Tensor): Timestep tensor, shape (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, dim)
        """
        # Apply linear transformation W1 to each element in the sequence (along dim=2)
        out1 = self.W1(x)  # Shape: (batch_size, seq_len, dim)

        # Concatenate out1 and tau along the last dimension
        out2 = self.W2(torch.cat([out1, tau], dim=-1))  # Shape: (batch_size, seq_len, dim)

        # Apply linear transformation W3
        out3 = self.W3(self.nonlinearity(out2))  # Shape: (batch_size, seq_len, dim)

        return out3

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Emu3MoE(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        # Base model (the same as in Emu3ForCausalLM)
        self.model = Emu3Model(config)
        # print("config.use_blockwise_attn_mask:", config.use_blockwise_attn_mask)
        self.vocab_size = config.vocab_size
        # self.use_blockwise_attn_mask = config.use_blockwise_attn_mask

        if hasattr(config, "vision_loss_weight"):
            self.use_weight = True
            self.vision_loss_weight = config.vision_loss_weight
            self.eov_token_id = config.eov_token_id
            self.bov_token_id = config.bov_token_id
        else:
            self.use_weight = False

        if config.action_experts:
            self.action_experts = config.action_experts
            action_config = Emu3Config.from_dict(config.action_config)
            self.vision_loss_weight = action_config.vision_loss_weight
            self.action_projector = ActionProjector(config.action_dim, action_config.hidden_size)
            self.action_layers = nn.ModuleList(
                [Emu3DecoderLayer(action_config, layer_idx) for layer_idx in range(action_config.num_hidden_layers)]
            )
            self.action_decoder = FinalLayer(action_config.hidden_size, config.action_dim)
            # self.rf = FlowMatchingScheduler(sample_method="uniform", s = 1.0)
            self.rf = FlowMatchingScheduler(sample_method="beta", s = 1.0)
            self.tau_emb = SinusoidalPosEmb(action_config.hidden_size)
        
        # Output head (same as Emu3ForCausalLM)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Example output will be the same as in Emu3ForCausalLM, with the inclusion of MoE-based processing.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        seq_len = hidden_states.shape[1]

        # processing action
        if action is not None and self.action_experts and self.training:
            # Generate noise with the same shape and data type as the action tensor
            noise = torch.randn_like(action, dtype=action.dtype)

            # Sample tau values and ensure the data type matches the noise tensor
            tau = self.rf.sample_t(noise.shape[0]).to(noise.dtype)

            noise_action = self.rf.add_noise(action, noise, tau)

            # Use forward_action to compute predictions and updated hidden states
            velo_pred, hidden_states_refine = self.forward_action(noise_action, tau, hidden_states)

            # flow matching loss
            loss_action = F.mse_loss(noise - action, velo_pred)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.use_weight:
                weights = torch.ones(self.config.vocab_size)
                vision_token_range = range(self.bov_token_id,self.eov_token_id+1)
                weights[vision_token_range] = self.vision_loss_weight
                loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))

            else:
                loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
            if action is not None and self.action_experts:
                loss += loss_action * self.vision_loss_weight
            # loss = loss_action
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def forward_action(self, z, t, cond):

        # Embed the sampled tau values and adjust the data type to match the noise tensor
        tau_emb = self.tau_emb(t).to(z.dtype)

        # Repeat tau embeddings along the action dimension to match the input shape
        tau_emb = tau_emb.repeat(1, z.shape[1], 1)

        seq_len = cond.shape[1]

        # Compute action embeddings using the action projector and the tau embeddings
        action_hidden_states = self.action_projector(z, tau_emb)

        # Concat in sequence dimension
        action_hidden_states = torch.cat([cond, action_hidden_states], dim=1)
        # transformer layers
        for action_layer in self.action_layers:
            action_hidden_states = action_layer(
                action_hidden_states
            )[0]
        hidden_states, action_hidden_states = action_hidden_states[:, :seq_len, :], action_hidden_states[:, seq_len:, :]
        velo_pred = self.action_decoder(action_hidden_states, tau_emb)

        return velo_pred, hidden_states

    def generate_action(self, outputs, sample_steps = 20, frames = 8, action_dim = 7):

        input_ids = outputs
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]

        # action generation 
        z = torch.randn((batch_size, frames, action_dim), dtype=hidden_states.dtype).to(hidden_states.device)
        dt = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * batch_size).to(hidden_states.device)

            velo_pred, hidden_states_i = self.forward_action(z, t, cond = hidden_states)

            z = z - dt * velo_pred
        
        return z
    def generate_jacobi_kv(
        self, 
        input_ids: Optional[torch.LongTensor] = None, 
        max_new_tokens=747+70, 
        max_iter= 747+70,
        max_new_seq_len:int = 2400,
        **kwargs: str
    ):                          
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        # print("kwargs keys:", list(kwargs.keys()))
        past_key_values = kwargs.get("past_kv", None)
        first_correct_token = kwargs.get("first_correct_token", None)
        allowed_token_ids = kwargs.get("allowed_token_ids", None)
        eos_token = kwargs["eos_token"]
        attention_mask = kwargs["attention_mask"]
        # max_new_tokens=max_new_tokens
        # max_new_seq_len=max_new_seq_len
        # max_iter=max_iter
        converge_step = []
        forward_times = 0
        all_jacobian_trajectory = []
        prompt_len = input_ids.shape[1]  # 直接获取序列长度
        # print("prompt_len:",prompt_len)
        generation = input_ids
        ### prefill the kv-cache
        if past_key_values is None:
            past_key_values, first_correct_token = self.jacobi_forward(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                past_key_values=None, 
                use_cache = True, 
                # prefill_phase = True,
                max_iter=max_iter,
                allowed_token_ids=allowed_token_ids)#返回键值对缓存，和第一个预测的元素
        # print("first_correct_token:",first_correct_token)
        ### generation phase
        itr = 0
        eos_reached = False
        while True:
            itr+=1
            bsz = 1 # only support batch_size = 1 now
            # randomly initialize the first point of jacobian trajectory
            valid_tokens = generation[0].tolist()
            if itr*max_new_tokens > max_new_seq_len:
                random_point = torch.tensor(random.choices(valid_tokens, k=(max_new_seq_len-(itr-1)*max_new_tokens)-1), device="cuda").view(1,-1)
            else:
                random_point = torch.tensor(random.choices(valid_tokens, k=(max_new_tokens-1)), device="cuda").view(1,-1)
            input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)#
            # jacobian_trajectory整个轨迹 n_gram_generation收敛点 iter_steps迭代步数
            jacobian_trajectory, coverage_point, first_correct_token, iter_steps,past_kv = self.jacobi_forward(
                input_ids=input_ids,
                pixel_values=None, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                past_key_values=past_key_values, 
                use_cache = True, 
                # prefill_phase = False,
                max_iter=max_iter,
                allowed_token_ids=allowed_token_ids,
                )
            forward_times += iter_steps
            all_jacobian_trajectory.append(jacobian_trajectory)#4维度 jacobian_trajectory3维[迭代数,1,16]
            eos_positions = torch.where(coverage_point[0]==eos_token)[0]

            if len(eos_positions)>0:
                eos_reached = True            
            ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
            generation = torch.cat((generation, coverage_point), dim=-1)
            # print("coverage_point.shape:",coverage_point.shape)
            # print("generation.shape:",generation.shape)
            past_key_values = past_kv
            # print("type(past_key_values):",type(past_key_values))
            if eos_reached or itr*max_new_tokens > max_new_seq_len:
                break
        
        # to support bsz > 1
        converge_step.append(forward_times / itr)#平均每次小循环的迭代步数
        generated_ids=generation
        # Extract predicted action tokens and translate into (normalized) continuous actions
        # predicted_action_token_ids = generated_ids[:, -max_new_tokens :]
        predicted_action_token_ids = generated_ids[:, prompt_len :]
        # past_key_values = delete_false_key_value(past_key_values,itr*max_new_tokens-max_new_seq_len)

        return {"output_token_ids":predicted_action_token_ids, 
                "converge_step":converge_step, 
                "all_jacobian_trajectory":all_jacobian_trajectory,
                "past_key_values":past_key_values,
                "final_tokens":generated_ids,
                # "first_correct_token":first_correct_token,
                }#这里的all 指一条数据
        # return actions

    # === Core Prismatic VLM `forward()` Logic ===
    def jacobi_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        max_iter: Optional[int] = 36,
        max_new_tokens: Optional[int] = 36,
        prefill_phase: Optional[bool] = True,
        position_ids: Optional[torch.LongTensor] = None,
        mask_action_tokens: Optional[bool] = False,
        allowed_token_ids: Optional[List[int]] = None,
    ): 
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training
        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]

        # === Handle Unimodal Forward ===
        if past_key_values:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            # assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"
            # print("input_ids",input_ids)
            jacobian_trajectory = []
            next_point = input_ids
            jacobian_trajectory.append(next_point)
            iter_counter = 0
            while True:
                current_point = next_point
                inputs_embeds = self.get_input_embeddings()(current_point)
                # inputs_embeds = self.model.embed_tokens(current_point)
                # attention_mask = None
                # position_ids = None
                # seq_length = current_point.shape[1]
                # if use_cache:#更新key value缓存
                #     use_legacy_cache = not isinstance(past_key_values, Cache)
                #     if use_legacy_cache:
                #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                #     past_key_values_length = past_key_values.get_usable_length(seq_length) 
                    # print("past_key_values_length:",past_key_values_length) # return previous_seq_length
                # if position_ids is None:
                #     device = input_ids.device if input_ids is not None else inputs_embeds.device
                #     position_ids = torch.arange(
                #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                #     )#获取位置编码
                #     position_ids = position_ids.unsqueeze(0)
                    # print("position_ids:",position_ids)
                # # attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                # if self._use_flash_attention_2:
                #     # 2d mask is passed through the layers
                #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                # elif self._use_sdpa :
                #     # output_attentions=True can not be supported when using SDPA, and we fall back on
                #     # the manual implementation that requires a 4D causal mask in all cases.
                #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                #         attention_mask,
                #         (batch_size, seq_length),
                #         inputs_embeds,
                #         past_key_values_length,
                #     )
                # else:
                #     # 4d mask is passed through the layers
                #     attention_mask = _prepare_4d_causal_attention_mask(
                #         attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                #     )

                language_model_output = self.forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    # labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                logits = language_model_output.logits
                
                # if isinstance(past_kv, tuple):
                #     print("past_kv is tuple")
                #     print("len(past_kv):",len(past_kv))
                #     print("len(past_kv[0]):",len(past_kv[0]))
                #     print("past_kv[0][0].shape:",past_kv[0][0].shape)
                logits = logits.float()
                if allowed_token_ids is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    # 将 allowed_token_ids 转换为一个 (batch_size, seq_length) 的索引矩阵
                    # print("allowed_token_ids:",allowed_token_ids)
                    allowed_token_ids_tensor = torch.tensor(allowed_token_ids, device=logits.device)  # (vocab_size, )
                    # print("allowed_token_ids_tensor.shape:",allowed_token_ids_tensor.shape)
                    # 创建 batch 和 seq 的索引
                    # batch_indices = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)  # (batch_size, 1)
                    # seq_indices = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)  # (1, seq_length)
                    # 使用 allowed_token_ids 索引出有效位置，设置为 0
                    mask[:, :, allowed_token_ids_tensor] = 0  # 设置有效位置为 0
                    # 将掩码应用到 logits 上
                    logits += mask
                softmax_logits = torch.nn.functional.softmax(logits / 0.01, dim=-1)
                all_shift_one_token = torch.argmax(softmax_logits, dim=-1)
                                
                # if allowed_token_ids is not None:
                #     # 创建一个掩码，初始化为 -inf
                #     mask = torch.full_like(logits, float('-inf'))  # shape: (batch_size, seq_length, vocab_size)

                #     # 将 allowed_token_ids 转换为 tensor
                #     allowed_token_ids_tensor = torch.tensor(allowed_token_ids).unsqueeze(0).expand(logits.size(0), -1)  # shape: (batch_size, seq_length)
                #     print("allowed_token_ids_tensor.shape:",allowed_token_ids_tensor.shape)

                #     # 扩展掩码，使得 allowed_token_ids 对应位置设置为 0
                #     mask.scatter_(2, allowed_token_ids_tensor.unsqueeze(-1), 0)  # 在 vocab_size 维度上设置 0

                #     # 将掩码应用到 logits 上
                #     logits += mask
                    
                    


                # 构造 next_point: 拼接首 token 和预测的 token 序列
                next_tokens = all_shift_one_token[0, -max_new_tokens:-1].view(1, -1)
                next_point = torch.cat((current_point[0, 0].view(1, -1), next_tokens), dim=-1)  # shape: [1, 16]

                jacobian_trajectory.append(next_point)

                # 判断是否收敛
                if torch.equal(current_point, next_point):
                    # print('Successfully break!')
                    first_correct_token = all_shift_one_token[:, -1]  # 取最后一个 token，维度 [1]
                    past_kv = language_model_output.past_key_values
                    
                    break

                iter_counter += 1
                # print("iter_counter:",iter_counter)
                # print("current_point",current_point)
                # print("next_point",next_point)
                if iter_counter == max_iter:
                    print('Max iteration reached!')
                    # print("current_point",current_point)
                    # print("next_point",next_point)
                    first_correct_token = all_shift_one_token[:, -1]
                    past_kv = language_model_output.past_key_values
                    break
                # language_model_output.past_key_values.delete_false_key_value(seq_length)

            # print("jacobian_trajectory",jacobian_trajectory)
            return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter,past_kv

        # === Handle Multimodal Forward ===
        else:##

            # past_key_values_length = 0
            # if use_cache:
            #     use_legacy_cache = not isinstance(past_key_values, Cache)
            #     if use_legacy_cache:
            #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            #     past_key_values_length = past_key_values.get_usable_length(seq_length) #根据当前序列长度和缓存中的历史状态来计算出可以复用的部分
            #     print("past_key_values_length:",past_key_values_length)
            # if position_ids is None:
            #     device = input_ids.device if input_ids is not None else inputs_embeds.device
            #     position_ids = torch.arange(
            #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            #     )
            #     position_ids = position_ids.unsqueeze(0)
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            # if self._use_flash_attention_2:#生成注意力掩码
            #         # 2d mask is passed through the layers
            #         attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            # elif self._use_sdpa :
            #         # output_attentions=True can not be supported when using SDPA, and we fall back on
            #         # the manual implementation that requires a 4D causal mask in all cases.
            #         attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            #             attention_mask,
            #             (batch_size, seq_length),
            #             inputs_embeds,
            #             past_key_values_length,
            #         )
            # else:
            #         # 4d mask is passed through the layers
            #         attention_mask = _prepare_4d_causal_attention_mask(
            #             attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            #         )

            # Dispatch to Language Model
            language_model_output = self.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                # labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            past_key_values = language_model_output.past_key_values
            logits = language_model_output.logits
            logits = logits.float()
            predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
            first_correct_token = predict_next_tokens[:, -1]
            # print("length of past_key_values:",len(past_key_values))
            return past_key_values,first_correct_token
        # === Otherwise =>> Assume Invalid! ===
    def generate_jacobi_kv_mutil_maxnewtokens(
        self, 
        input_ids: Optional[torch.LongTensor] = None, 
        max_new_tokens=[747,70], 
        max_iter=[747,70],
        max_new_seq_len:int = 1700,
        **kwargs: str
    ):                          
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        # print("kwargs keys:", list(kwargs.keys()))
        eos_token = kwargs["eos_token"]
        attention_mask = kwargs["attention_mask"]
        # max_new_tokens=max_new_tokens
        # max_new_seq_len=max_new_seq_len
        # max_iter=max_iter
        converge_step = []
        forward_times = 0
        all_jacobian_trajectory = []
        # prompt_len = input_ids.shape[1]  # 直接获取序列长度
        # print("prompt_len:",prompt_len)
        generation = input_ids
        ### prefill the kv-cache
        past_key_values, first_correct_token = self.jacobi_forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=max_new_tokens, 
            past_key_values=None, 
            use_cache = True, 
            # prefill_phase = True,
            max_iter=max_iter)#返回键值对缓存，和第一个预测的元素
        # print("first_correct_token:",first_correct_token)
        ### generation phase
        itr = 0
        eos_reached = False
        for mnt,mi in zip(max_new_tokens,max_iter):
            itr+=1
            bsz = 1 # only support batch_size = 1 now
            # randomly initialize the first point of jacobian trajectory
            valid_tokens = generation[0].tolist()
            random_point = torch.tensor(random.choices(valid_tokens, k=(mnt-1)), device="cuda").view(1,-1)
            input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)#
            # jacobian_trajectory整个轨迹 n_gram_generation收敛点 iter_steps迭代步数
            jacobian_trajectory, coverage_point, first_correct_token, iter_steps,past_kv = self.jacobi_forward(
                input_ids=input_ids,
                pixel_values=None, 
                attention_mask=attention_mask, 
                max_new_tokens=mnt, 
                past_key_values=past_key_values, 
                use_cache = True, 
                # prefill_phase = False,
                max_iter=mi)
            forward_times += iter_steps
            all_jacobian_trajectory.append(jacobian_trajectory)#4维度 jacobian_trajectory3维[迭代数,1,16]
            eos_positions = torch.where(coverage_point[0]==eos_token)[0]

            if len(eos_positions)>0:
                eos_reached = True            
            ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
            generation = torch.cat((generation, coverage_point), dim=-1)
            past_key_values = past_kv
            if eos_reached :
                break
        
        # to support bsz > 1
        converge_step.append(forward_times / itr)#平均每次小循环的迭代步数
        generated_ids=generation
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[:, -sum(max_new_tokens) :]

        return {"output_token_ids":predicted_action_token_ids, 
                "converge_step":converge_step, 
                "all_jacobian_trajectory":all_jacobian_trajectory}#这里的all 指一条数据