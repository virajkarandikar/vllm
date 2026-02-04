# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.config import VllmConfig, CacheConfig, SchedulerConfig, get_current_vllm_config
from vllm.model_executor.custom_op import CustomOp
from vllm.attention.layer import Attention
from vllm.sequence import IntermediateTensors
from vllm.compilation.decorators import support_torch_compile

from vllm.v1.attention.backends.fastconformer_conv import (
    FastConformerConvBackend,
    FastConformerConvMetadata,
)
from vllm.forward_context import get_forward_context
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.fastconformer_rpe_attention import (
    FastConformerRPEBackend,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, FastConformerConvSpec
from vllm.model_executor.models.fastconformer_preprocessor import FastConformerPreprocessor
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
)
from vllm.utils import direct_register_custom_op
from vllm.transformers_utils.configs.fastconformer import FastConformerCTCConfig
import math


class ConformerFFN(nn.Module):
    """Conformer FeedForward module."""
    def __init__(self, d_model: int, ff_mult: int = 4, use_bias: bool = True):
        super().__init__()
        d_ff = ff_mult * d_model
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window: int, use_bias: bool,
                 cache_config: CacheConfig, scheduler_config: SchedulerConfig, prefix: str):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.window = int(window)
        self.prefix = prefix
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        
        self.use_bias = use_bias
        self.q_proj = nn.Linear(d_model, d_model, bias=self.use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=self.use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=self.use_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=self.use_bias)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.dh))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.dh))

        # 1. config for FastConformerRPEBackend
        # used in `_forward_sdpa_2`
        self.attn = Attention(
            num_heads=self.h,
            head_size=self.dh,
            scale=self.dh ** -0.5,
            num_kv_heads=self.h,
            cache_config=CacheConfig(
                sliding_window=self.window,
                cache_dtype="auto",
                block_size=cache_config.block_size,
                calculate_kv_scales=False,
            ),
            prefix=self.prefix,
            pos_bias_u=self.pos_bias_u,
            pos_bias_v=self.pos_bias_v,
            linear_pos=self.linear_pos,
            attn_backend=FastConformerRPEBackend,
        )

        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        if cache_config.block_size != 128:
            raise Exception(
                f"attn cache block size must be 128, got {cache_config.block_size}"
            )

        self.register_buffer(
            "qkv_weight",
            torch.empty(3 * d_model, d_model, dtype=self.q_proj.weight.dtype)
        )
        if self.use_bias:
            self.register_buffer(
                "qkv_bias",
                torch.empty(3 * d_model, dtype=self.q_proj.weight.dtype)
            )
        else:
            self.qkv_bias = None
        self._qkv_fused_ready: bool = False


    def _fused_qkv_projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._qkv_fused_ready
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)  # [B, T, 3D]
        D = x.size(-1)
        q, k, v = qkv.split(D, dim=-1)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE(vklimkov): clean up possible attention implementations for RPE.
        # see git history for more details.
        q, k, v = self._fused_qkv_projection(x)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)

@CustomOp.register("fastconformer_conv_module")
class ConformerConvModule(CustomOp, AttentionLayerBase):
    def __init__(self, d_model: int, k: int, use_bias: bool, norm_type: str,prefix: str, cache_config: CacheConfig, dtype: torch.dtype):
        super().__init__()
        assert k % 2 == 1
        self.prefix = prefix
        self.use_bias = use_bias

        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=self.use_bias)
        self.dw  = nn.Conv1d(d_model, d_model, k, padding=(k-1)//2, groups=d_model, bias=self.use_bias)
        self.norm_type = norm_type
        if norm_type == "batch_norm":
            self.bn  = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.bn  = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        self.activation = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1, bias=self.use_bias)

        self.d_model = d_model
        self.k = int(k)
        self.left_ctx = self.k - 1  # L

        # The conv cache page size must match the attn page size by vLLM requirements.
        # TODO: is there a less hacky way to do this?
        self.left_shape = self.left_ctx * 32

        self.cache_config = cache_config
        self.kv_cache = [torch.tensor([])]
        self.dtype = dtype

        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.fastconformer_conv_module(
            hidden_states,
            self.prefix,
        )

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        B, T, D = hidden_states.shape
        assert D == self.d_model

        y = hidden_states.transpose(1, 2)         # [B, D, T]
        y = self.pw1(y)               # [B, 2D, T]
        a, b = y.chunk(2, dim=1)
        pre_dw = a * torch.sigmoid(b)  # GLU

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata
        if not isinstance(attn_meta_all, dict):
            # dummy path: plain conv
            y_dw = self.dw(pre_dw)
            if self.norm_type == "batch_norm":
                y_dw = self.bn(y_dw)
            elif self.norm_type == "layer_norm":
                y_dw = y_dw.transpose(1, 2)  # B D T -> B T D
                y_dw = self.bn(y_dw)
                y_dw = y_dw.transpose(1, 2)  # B T D -> B D T

            y_dw = F.silu(y_dw)
            y = self.pw2(y_dw)
            out = y.transpose(1, 2)   # [B, T, D]
            if is_2d:
                out = out.squeeze(0)
            return out

        attn_metadata: FastConformerConvMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        # the length of the cache is not exactly equal to the left context length
        # because of an invariant that vLLM enforces where the page size must
        # be the same for both conv and attention layers.
        store = self.kv_cache[fctx.virtual_engine]  # [num_pages, L', D]
        L = self.left_ctx
        hist_full = store[page_indices]                   # [B, L', D]
        hist = hist_full[:, -L:, :].transpose(1, 2).contiguous()   # [B, D, L]

        x_cat = torch.cat([hist, pre_dw], dim=-1)                 # [B, D, L+T]

        y_dw = F.conv1d(
            x_cat,
            self.dw.weight,
            self.dw.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=D,
        )[:, :, -T:]

        if self.norm_type == "batch_norm":
            y_dw = self.bn(y_dw)
        elif self.norm_type == "layer_norm":
            y_dw = y_dw.transpose(1, 2)  # B D T -> B T D
            y_dw = self.bn(y_dw)
            y_dw = y_dw.transpose(1, 2)  # B T D -> B D T

        y_dw = F.silu(y_dw)
        y = self.pw2(y_dw).transpose(1, 2)                        # [B, T, D]

        # update cache
        new_hist = x_cat[:, :, -L:]                       # [B, D, L]
        store[page_indices, -L:, :] = new_hist.transpose(1, 2)   # [B, L, D]

        if is_2d:
            y = y.squeeze(0)
        return y

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() == 2, "forward_cuda expects a 2D tensor [T, D]"
        T, D = hidden_states.shape
        assert D == self.d_model

        w1 = self.pw1.weight.squeeze(-1)   # [2D, D]
        b1 = self.pw1.bias                 # [2D] or None
        y_pw1 = F.linear(hidden_states, w1, b1)        # [T, 2D]
        a, b = y_pw1.chunk(2, dim=-1)      # [T, D], [T, D]
        pre_dw = a * torch.sigmoid(b)      # GLU -> [T, D]

        fctx = get_forward_context()
        attn_meta_all = fctx.attn_metadata

        if attn_meta_all is None:
            return torch.zeros_like(hidden_states)

        K = self.dw.weight.size(2)
        conv_weights = self.dw.weight.view(D, K)
        conv_bias = self.dw.bias  #  could be None
        pre_dw_2d = pre_dw.transpose(0, 1)

        attn_metadata: FastConformerConvMetadata = attn_meta_all[self.prefix]
        block_table = attn_metadata.block_table_tensor
        page_indices = block_table[:, 0]

        store = self.kv_cache[fctx.virtual_engine]
        conv_state = store.contiguous().transpose(1, 2)

        query_start_loc = attn_metadata.query_start_loc

        has_initial_state = torch.ones(
            page_indices.size(0), dtype=torch.bool, device=pre_dw_2d.device
        )

        y_dw_2d = causal_conv1d_fn(  # dim x cu_seq_len
            pre_dw_2d,
            conv_weights,
            conv_bias,
            conv_state,
            query_start_loc,
            cache_indices=page_indices,
            has_initial_state=has_initial_state,
            activation=None,
            metadata=attn_metadata,
        )

        if self.norm_type == "batch_norm":
            y_bn = self.bn(y_dw_2d.unsqueeze(0)).squeeze(0)
        elif self.norm_type == "layer_norm":
            # need to transpose dim x seq -> seq x dim
            y_dw_2d = y_dw_2d.transpose(0, 1)  # seq x dim
            y_bn = self.bn(y_dw_2d).transpose(0, 1)  # dim x seq

        y_act = F.silu(y_bn)
        w2 = self.pw2.weight.squeeze(-1)
        b2 = self.pw2.bias
        y_out = F.linear(y_act.transpose(0, 1), w2, b2)
        return y_out

    def get_attn_backend(self) -> AttentionBackend:
        return FastConformerConvBackend

    def get_kv_cache_spec(self) -> KVCacheSpec:
        return FastConformerConvSpec(
            # block_size=self.cache_config.block_size,
            # WARNING: when caching is enabled, block size be the same across all layers.
            # for conv we need only 1 though.
            block_size=1,
            shape=(self.left_shape, self.d_model),
            dtype=self.dtype,
        )



def fastconformer_conv_fwd(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.forward_cuda(hidden_states=hidden_states)


def fastconformer_conv_fwd_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.zeros_like(hidden_states)


direct_register_custom_op(
    op_name="fastconformer_conv_module",
    op_func=fastconformer_conv_fwd,
    fake_impl=fastconformer_conv_fwd_fake,
)


class ConformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads: int,
        k_conv: int,
        ff_mult: int,
        attn_window: int,
        use_bias: bool,
        norm_type: str,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        dtype: torch.dtype,
        prefix: str,
    ):
        super().__init__()
        self.ln_ff1 = nn.LayerNorm(d_model)
        self.ff1 = ConformerFFN(d_model, ff_mult, use_bias=use_bias)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = RelPosSelfAttention(d_model, n_heads, attn_window, use_bias, cache_config, scheduler_config, prefix=f"{prefix}.attn")
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, k_conv, use_bias, norm_type, prefix=f"{prefix}.conv", cache_config=cache_config, dtype=dtype)
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2 = ConformerFFN(d_model, ff_mult, use_bias=use_bias)
        self.ln_out = nn.LayerNorm(d_model)
        self.fc_factor = 0.5
        self.prefix = prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_ff1(x)
        y = self.ff1(y)
        x = x + y * self.fc_factor

        y = self.ln_attn(x)
        y = self.attn(y)
        x = x + y

        y = self.ln_conv(x)
        y = self.conv(y)
        x = x + y

        y = self.ln_ff2(x)
        y = self.ff2(y)
        x = x + y * self.fc_factor

        x = self.ln_out(x)
        return x

@support_torch_compile
class FastConformerCTC(nn.Module):
    """FastConformerCTC for vLLM."""
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: FastConformerCTCConfig = vllm_config.model_config.hf_config
        self.config = config

        self.d_model = self.config.d_model
        self.xscale = math.sqrt(self.d_model) if self.config.xscale else None
        self.dtype = vllm_config.model_config.dtype

        att_window = int(config.att_left_ctx + config.att_right_ctx)
        assert att_window > 0, "att_window must be positive"

        self.prefix = prefix
        self.preprocessor = FastConformerPreprocessor(
            vllm_config=vllm_config,
            prefix=prefix,
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                n_heads=config.num_attention_heads,
                k_conv=config.k_conv,
                ff_mult=config.ff_mult,
                attn_window=att_window,
                use_bias=config.use_bias,
                norm_type=config.norm_type,
                cache_config=vllm_config.cache_config,
                scheduler_config=vllm_config.scheduler_config,
                dtype=self.dtype,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(config.n_layers)
        ])
        self.adapter = None
        if self.config.adapted_dimension is not None:
            self.adapter = nn.Linear(self.d_model, self.config.adapted_dimension)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise Exception("not applicable for this model")

    def forward(
        self,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,            # unused
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.preprocessor(audio)
        if self.xscale:
            x = x * self.xscale
        for blk in self.blocks:
            x = blk(x)
        if self.adapter is not None:
            x = self.adapter(x)
        return x, x

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # do nothing, we dont do sampling for fastconformer
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        nemo = {name: tensor for name, tensor in weights}
        self.preprocessor.load_weights(nemo)

        loaded_pairs: list[tuple[str, str]] = []
        skipped: list[tuple[str, str]] = []

        model_params = dict(self.named_parameters())
        loaded_param_names: set[str] = set()

        # helper to copy tensors with shape checking
        def copy_(dst_param: torch.nn.Parameter, src: torch.Tensor,
                dst_name: str, src_name: str):
            if dst_param.shape != src.shape:
                if src.dim() == 3 and src.shape[-1] == 1 and dst_param.shape == src.shape[:2]:
                    dst_param.data.copy_(src.squeeze(-1))
                else:
                    skipped.append((src_name,
                                    f"shape {tuple(src.shape)} -> {dst_name} {tuple(dst_param.shape)}"))
                    return
            else:
                dst_param.data.copy_(src)
            loaded_pairs.append((src_name, dst_name))
            loaded_param_names.add(dst_name)

        if self.adapter is not None:
            weight_name = "adapter.weight"
            copy_(self.adapter.weight, nemo[weight_name], weight_name, weight_name)
            bias_name = "adapter.bias"
            copy_(self.adapter.bias, nemo[bias_name], bias_name, bias_name)

        for i, blk in enumerate(self.blocks):
            base = f"encoder.layers.{i}"

            ln_pairs = [
                (f"{base}.norm_feed_forward1.weight", blk.ln_ff1.weight, f"blocks.{i}.ln_ff1.weight"),
                (f"{base}.norm_feed_forward1.bias",   blk.ln_ff1.bias,   f"blocks.{i}.ln_ff1.bias"),
                (f"{base}.norm_self_att.weight",      blk.ln_attn.weight,f"blocks.{i}.ln_attn.weight"),
                (f"{base}.norm_self_att.bias",        blk.ln_attn.bias,  f"blocks.{i}.ln_attn.bias"),
                (f"{base}.norm_conv.weight",          blk.ln_conv.weight,f"blocks.{i}.ln_conv.weight"),
                (f"{base}.norm_conv.bias",            blk.ln_conv.bias,  f"blocks.{i}.ln_conv.bias"),
                (f"{base}.norm_feed_forward2.weight", blk.ln_ff2.weight, f"blocks.{i}.ln_ff2.weight"),
                (f"{base}.norm_feed_forward2.bias",   blk.ln_ff2.bias,   f"blocks.{i}.ln_ff2.bias"),
                (f"{base}.norm_out.weight",           blk.ln_out.weight, f"blocks.{i}.ln_out.weight"),
                (f"{base}.norm_out.bias",             blk.ln_out.bias,   f"blocks.{i}.ln_out.bias"),
            ]
            for n_src, p_dst, n_dst in ln_pairs:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn1 = [
                (f"{base}.feed_forward1.linear1.weight", blk.ff1.linear1.weight, f"blocks.{i}.ff1.linear1.weight"),
                (f"{base}.feed_forward1.linear1.bias",   blk.ff1.linear1.bias,   f"blocks.{i}.ff1.linear1.bias"),
                (f"{base}.feed_forward1.linear2.weight", blk.ff1.linear2.weight, f"blocks.{i}.ff1.linear2.weight"),
                (f"{base}.feed_forward1.linear2.bias",   blk.ff1.linear2.bias,   f"blocks.{i}.ff1.linear2.bias"),
            ]
            for n_src, p_dst, n_dst in ffn1:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            attn = [
                (f"{base}.self_attn.linear_q.weight", blk.attn.q_proj.weight, f"blocks.{i}.attn.q_proj.weight"),
                (f"{base}.self_attn.linear_q.bias",   blk.attn.q_proj.bias,   f"blocks.{i}.attn.q_proj.bias"),
                (f"{base}.self_attn.linear_k.weight", blk.attn.k_proj.weight, f"blocks.{i}.attn.k_proj.weight"),
                (f"{base}.self_attn.linear_k.bias",   blk.attn.k_proj.bias,   f"blocks.{i}.attn.k_proj.bias"),
                (f"{base}.self_attn.linear_v.weight", blk.attn.v_proj.weight, f"blocks.{i}.attn.v_proj.weight"),
                (f"{base}.self_attn.linear_v.bias",   blk.attn.v_proj.bias,   f"blocks.{i}.attn.v_proj.bias"),
                (f"{base}.self_attn.linear_out.weight", blk.attn.o_proj.weight, f"blocks.{i}.attn.o_proj.weight"),
                (f"{base}.self_attn.linear_out.bias",   blk.attn.o_proj.bias,   f"blocks.{i}.attn.o_proj.bias"),
                (f"{base}.self_attn.linear_pos.weight", blk.attn.linear_pos.weight, f"blocks.{i}.attn.linear_pos.weight"),
                (f"{base}.self_attn.pos_bias_u",        blk.attn.pos_bias_u,        f"blocks.{i}.attn.pos_bias_u"),
                (f"{base}.self_attn.pos_bias_v",        blk.attn.pos_bias_v,        f"blocks.{i}.attn.pos_bias_v"),
            ]
            for n_src, p_dst, n_dst in attn:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            conv = [
                (f"{base}.conv.pointwise_conv1.weight", blk.conv.pw1.weight, f"blocks.{i}.conv.pw1.weight"),
                (f"{base}.conv.pointwise_conv1.bias",   blk.conv.pw1.bias,   f"blocks.{i}.conv.pw1.bias"),
                (f"{base}.conv.depthwise_conv.weight",  blk.conv.dw.weight,  f"blocks.{i}.conv.dw.weight"),
                (f"{base}.conv.depthwise_conv.bias",    blk.conv.dw.bias,    f"blocks.{i}.conv.dw.bias"),
                (f"{base}.conv.pointwise_conv2.weight", blk.conv.pw2.weight, f"blocks.{i}.conv.pw2.weight"),
                (f"{base}.conv.pointwise_conv2.bias",   blk.conv.pw2.bias,   f"blocks.{i}.conv.pw2.bias"),
            ]
            if blk.conv.norm_type == "batch_norm":
                conv.extend([
                    (f"{base}.conv.batch_norm.weight",      blk.conv.bn.weight,  f"blocks.{i}.conv.bn.weight"),
                    (f"{base}.conv.batch_norm.bias",        blk.conv.bn.bias,    f"blocks.{i}.conv.bn.bias"),
                    (f"{base}.conv.batch_norm.running_mean", blk.conv.bn.running_mean, f"blocks.{i}.conv.bn.running_mean"),
                    (f"{base}.conv.batch_norm.running_var",  blk.conv.bn.running_var,  f"blocks.{i}.conv.bn.running_var"),
                    (f"{base}.conv.batch_norm.num_batches_tracked", blk.conv.bn.num_batches_tracked, f"blocks.{i}.conv.bn.num_batches_tracked"),
                ])
            elif blk.conv.norm_type == "layer_norm":
                conv.extend([
                    (f"{base}.conv.batch_norm.weight",      blk.conv.bn.weight,  f"blocks.{i}.conv.bn.weight"),
                    (f"{base}.conv.batch_norm.bias",        blk.conv.bn.bias,    f"blocks.{i}.conv.bn.bias"),
                ])
            for n_src, p_dst, n_dst in conv:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            ffn2 = [
                (f"{base}.feed_forward2.linear1.weight", blk.ff2.linear1.weight, f"blocks.{i}.ff2.linear1.weight"),
                (f"{base}.feed_forward2.linear1.bias",   blk.ff2.linear1.bias,   f"blocks.{i}.ff2.linear1.bias"),
                (f"{base}.feed_forward2.linear2.weight", blk.ff2.linear2.weight, f"blocks.{i}.ff2.linear2.weight"),
                (f"{base}.feed_forward2.linear2.bias",   blk.ff2.linear2.bias,   f"blocks.{i}.ff2.linear2.bias"),
            ]
            for n_src, p_dst, n_dst in ffn2:
                if n_src in nemo:
                    copy_(p_dst, nemo[n_src], n_dst, n_src)

            with torch.no_grad():
                attn_mod = blk.attn
                w_cat = torch.cat([
                    attn_mod.q_proj.weight,
                    attn_mod.k_proj.weight,
                    attn_mod.v_proj.weight,
                ], dim=0)
                attn_mod.qkv_weight.copy_(w_cat)
                if attn_mod.use_bias:
                    b_cat = torch.cat([
                        attn_mod.q_proj.bias,
                        attn_mod.k_proj.bias,
                        attn_mod.v_proj.bias,
                    ], dim=0)
                    attn_mod.qkv_bias.copy_(b_cat)
                attn_mod._qkv_fused_ready = True

        loaded_src = {src for (src, _) in loaded_pairs}

        print(f"[load_weights] Loaded {len(loaded_pairs)} tensors.")

        if skipped:
            print(f"[load_weights] Skipped {len(skipped)} tensors (showing first 40):")
            for n, why in skipped[:40]:
                if n not in loaded_src:
                    print(f"  - {n}: {why}")

        # preprocessor is loaded separately
        model_params_lst = [x for x in model_params.keys() if not x.startswith("preprocessor.")]
        unused_model_params = sorted(set(model_params_lst) - loaded_param_names)
        if unused_model_params:
            print(f"[load_weights] Model params with NO checkpoint match ({len(unused_model_params)} shown first 40):")
            for n in unused_model_params[:40]:
                print(f"  - {n} : shape {tuple(model_params[n].shape)}")

        if not skipped and not unused_model_params:
            print("[load_weights] all weights loaded successfully.")
