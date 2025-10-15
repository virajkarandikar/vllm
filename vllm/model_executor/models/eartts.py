# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union, Iterable

import torch
from torch import nn
import numpy as np
from transformers.generation.logits_process import (
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from .utils import (
    AutoWeightsLoader,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # TODO: is casting really needed?
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y = self.mlp(y)
        y = self.post_norm(y)
        x = x + y
        return x

def gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates a tensor of Gumbel noise with the same shape as the input tensor.

    This is used for the Gumbel-Max trick, a technique to sample from a categorical
    distribution in a differentiable way (using a straight-through estimator).

    Args:
        tensor (torch.Tensor): The input tensor to match the shape of.
        eps (float): A small epsilon value for numerical stability.

    Returns:
        torch.Tensor: A tensor containing Gumbel noise.
    """
    # Sample from a uniform distribution
    u = torch.rand_like(tensor)
    # Apply the inverse CDF of the Gumbel distribution
    return -torch.log(-torch.log(u + eps) + eps)


def batch_matmul(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs a batched matrix multiplication using PyTorch's native functions.
    In NeMo this is implemented as a custom kernel using triton.
    TODO: check vLLM kernels if there is one available, check if triton can be used here.

    Args:
        x (Tensor): The input tensor of shape `[batch_size, d_in]`.
        w (Tensor): The weight tensor of shape `[num_weights, d_out, d_in]`.
        y (Tensor): The index tensor of shape `[batch_size]`.

    Returns:
        Tensor: The result of the multiplication, shape `[batch_size, d_out]`.
    """
    # w[y] gathers the weight matrices for each item in the batch.
    # x.unsqueeze(2) reshapes x to [batch_size, d_in, 1] for bmm.
    # The result is squeezed to remove the trailing dimension of size 1.
    return torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)


class MoGHead(nn.Module):
    """
    A Mixture of Gaussians (MoG) prediction head.

    This module takes a hidden state and predicts the parameters for a mixture of
    Gaussian distributions. It's suitable for modeling continuous, multi-modal data.

    Args:
        hidden_size (int): The dimensionality of the input hidden state.
        intermediate_size (int): The dimensionality of the MLP layers.
        out_size (int): The dimensionality of the output vectors (the mean of each Gaussian).
        num_layers (int): The number of MLP layers in the stack.
        num_predictions (int): The number of Gaussian components in the mixture.
        low_rank (int | None): The dimensionality used for compressing the hidden states.
        min_log_std (float): The minimum value for the logarithm of the standard deviation.
        eps (float): A small epsilon value for the RMSNorm layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_size: int,
        num_layers: int,
        num_predictions: int,
        low_rank: Optional[int] = 64,
        top_p_or_k: Optional[float | int] = 1.0,
        min_log_std: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_size = out_size
        self.low_rank = low_rank
        self.num_predictions = num_predictions
        self.min_log_std = min_log_std
        self.top_p_or_k = top_p_or_k

        self.logits_processor = TopPLogitsWarper(self.top_p_or_k) if isinstance(self.top_p_or_k, float) \
            else TopKLogitsWarper(self.top_p_or_k) if isinstance(self.top_p_or_k, int) else None

        self.mlp_stack = nn.Sequential(
            *[MLPLayer(hidden_size, intermediate_size, eps=eps) for _ in range(num_layers)],
            RMSNorm(hidden_size, eps=eps),
        )

        if low_rank is None:
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)  # Predicts mixture weights
            self.proj_mus = nn.Linear(hidden_size, num_predictions * out_size, bias=False)  # Predicts means
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)  # Predicts log standard deviations
        else:
            assert low_rank < out_size
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)  # Predicts mixture weights
            self.proj_mus = nn.Linear(hidden_size, num_predictions * low_rank, bias=False)  # Predicts means
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)  # Predicts log standard deviations
            self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
            self.low_mat = nn.Parameter(torch.randn(num_predictions, out_size, low_rank) * (low_rank**-0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs inference by sampling from the predicted mixture distribution.

        Args:
            x (Tensor): The input hidden state.
            top_p_or_k (float | int): The value for top-p (nucleus) or top-k sampling of the mixture components.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the mean of the chosen component,
                                   and the log standard deviations.
        """
        bt = x.size(0)
        n, d = self.num_predictions, self.low_rank or self.out_size

        x = self.mlp_stack(x)

        logits = self.proj_logits(x)

        # Apply top-p or top-k filtering to the mixture logits
        if self.logits_processor is not None:
            logits = self.logits_processor(None, logits.view(-1, n)).view_as(logits)

        # Sample a mixture component using the Gumbel-Max trick
        mixture_indices = (nn.functional.log_softmax(logits, dim=-1) + gumbel_like(logits)).argmax(-1)

        # Select the mean corresponding to the sampled component
        mu = batch_matmul(
            x.view(bt, -1),
            self.proj_mus.weight.detach().view(n, d, -1),
            mixture_indices.view(bt),
        ).view(bt, d)
        if self.proj_mus.bias is not None:
            mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

        if self.low_rank:
            #assert math.log2(d).is_integer() and math.log2(self.out_size).is_integer()
            mu = batch_matmul(
                mu.view(bt, -1),
                self.low_mat.detach().view(n, self.out_size, -1),
                mixture_indices.view(bt),
                # TODO: these are the arguments for custom kernel impl
                #BLOCK_SIZE_DIN=d,
                #BLOCK_SIZE_DOUT=self.out_size,
            ).view(bt, self.out_size)
            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((bt, d), device=x.device)

        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mu * torch.exp(logs) + mu_res, logs


class EarTTSForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.backbone = Gemma3Model(vllm_config=vllm_config, prefix=prefix)

        # easy access of cruicial config params
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size
        self.noise_scale = self.config.codebook_size

        # pre-compute how many tokens are unmasked at each iteration
        rates = np.linspace(0.0, 1.0, self.config.num_iter + 1)[:-1].reshape(-1, 1)
        masking_rates = np.power(1 - np.power(rates, self.config.exponent), 1 / self.config.exponent)
        num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(int)
        num_maskings_shifted = np.pad(num_maskings[1:], ((0, 1), (0, 0)), constant_values=0)
        sampling_per_step = num_maskings - num_maskings_shifted
        sampling_per_step_flat = sampling_per_step.flatten()
        # Drop any values at the beginning that are 0
        first_nonzero = np.argmax(sampling_per_step_flat != 0)
        self.num_to_sample = sampling_per_step_flat[first_nonzero:].tolist()

        # create layers used outside of backbone
        # the `codebook_size` token is reserved for padding
        # Store as Parameters so they can be used as tensors directly
        self.rvq_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.codebook_size + 1, self.config.latent_size))
            for _ in range(self.num_quantizers)
        ])
        self.padding_idx = self.codebook_size
        self.embed_code = nn.Linear(self.config.latent_size, self.config.hidden_size, bias=False)
        self.mog_head = MoGHead(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            out_size=self.config.latent_size,
            num_layers=self.config.mog_num_layers,
            num_predictions=self.config.mog_num_predictions,
            low_rank=self.config.mog_low_rank,
            top_p_or_k=self.config.top_p_or_k,
            min_log_std=self.config.mog_min_log_std,
            eps=self.config.mog_eps,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # this is for compatability, it is not supposed to be used
        return self.backbone.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.backbone(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        codes = self._generate_step(hidden_states)  # quantizers x BT
        return codes.transpose(0, 1).to(inputs_embeds.dtype)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # TODO: skip prefixes for embeddings, we dont use them
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
        """
        Embedds all codes into a single embedding.
        Args:
            code: Tensor (num_quantizers x BT) Acoustic codes to embed and add

        Returns:
            Tensor (BT x latent_size) - embedded codes
        """
        res = nn.functional.embedding(code[0], self.rvq_embeddings[0], padding_idx=self.padding_idx)
        for i in range(1, self.num_quantizers):
            res = res + nn.functional.embedding(code[i], self.rvq_embeddings[i], padding_idx=self.padding_idx)
        return res

    def _depthsum_encoding_step_reshaped(
        self,
        r: torch.Tensor,           # [B*T, hidden_size]
        code: torch.Tensor,        # [num_quantizers, B*T]
        depth_str: int,
        k: int,
    ) -> torch.Tensor:
        """
        RVQ encoding with reshaped code tensor.
        
        Args:
            embs: [num_quantizers, vocab_size, hidden_size] - RVQ codebook embeddings
            r: [B*T, hidden_size] - residual to quantize
            code: [num_quantizers, B*T] - output code tensor
            depth_str: starting quantizer level
            k: number of quantizer levels to process
        """
        for i in range(depth_str, depth_str + k):
            # self.rvq_embeddings[i]: [vocab_size, latent_size]
            # r: [B*T, latent_size]
            
            # Compute distances: ||emb||² - 2⟨r, emb⟩
            idx_sel = (
                self.rvq_embeddings[i].pow(2).sum(-1)  # [vocab_size]
                - 2 * (r @ self.rvq_embeddings[i].T)    # [B*T, vocab_size]
            ).argmin(-1)                 # [B*T]
            
            # Update residual
            emb_i = nn.functional.embedding(idx_sel, self.rvq_embeddings[i], padding_idx=self.padding_idx)  # [B*T, latent_size]
            r = r - emb_i
            
            # Store selected indices
            code[i] = idx_sel
        
        return code

    def _generate_step(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the iterative unmasking process for a single generation step.
        This function takes the hidden state from the backbone transformer and generates
        codes through an iterative unmasking process.

        Args:
            hidden_states: Tensor (BT x hidden_size) - The hidden states from the backbone

        Returns:
            Tensor (num_quantizers x BT) - The generated codes
        """

        device = hidden_states.device
        # Initialize the full code tensor
        code = torch.zeros((self.num_quantizers, hidden_states.shape[0]), dtype=torch.long, device=device) + self.codebook_size
        # Iteratively unmask the continuous part of the code
        cnt = 0
        for k in self.num_to_sample:
            # Prepare input for the MoG head
            mog_input_embeds = self.embed_code(self._depthsum_embedding(code))  # (BT x hidden_size)
            mog_input_embeds += hidden_states

            mog_mu, mog_logs = self.mog_head(
                mog_input_embeds,
            )
            z = mog_mu + torch.exp(mog_logs) * torch.randn_like(mog_mu) * self.noise_scale
            code = self._depthsum_encoding_step_reshaped(z, code, cnt, k)
            cnt += k
        return code