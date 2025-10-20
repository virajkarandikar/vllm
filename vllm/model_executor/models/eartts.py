# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import json
from typing import Optional, Union, Iterable

import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from transformers.generation.logits_process import (
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from transformers import (
    AutoConfig,
    AutoModelForTextEncoding,
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


def sequence_mask(
    lengths: torch.Tensor, max_length: torch.Tensor | int | None = None
) -> torch.Tensor:
    """
    Creates a boolean mask from a 1D tensor of sequence lengths.

    This function is useful for masking out padding in sequences. Given a tensor
    of lengths, it produces a 2D boolean tensor where `mask[i, j]` is `True` if
    `j < lengths[i]` and `False` otherwise.

    Args:
        lengths (Long Tensor): A 1D tensor of integer lengths. Shape: `[batch_size]`.
        max_length (Long Tensor | int | None, optional): The maximum length of the mask. If None,
                                           it is inferred from the maximum value
                                           in `lengths`. Defaults to None.

    Returns:
        Tensor: The boolean mask. Shape: `[batch_size, max_length]`.
    """
    if max_length is None:
        max_length = lengths.max()

    # Create a range tensor from 0 to max_length - 1
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)  # type: ignore[arg-type]

    # Compare each length with the range tensor to create the mask via broadcasting
    return x.unsqueeze(0) < lengths.unsqueeze(1)


class CharAwareSubwordEncoder(nn.Module):
    """
    An encoder that creates subword embeddings from character-level embeddings.
    This module replaces a standard subword embedding layer. It breaks down each
    subword into its constituent characters, embeds the characters, and then
    aggregates these character embeddings (e.g., via mean pooling) to form the
    final subword representation. This allows the model to handle rare or out-of-vocabulary
    subwords more gracefully.

    Args:
        out_size (int): The dimensionality of the output embedding vectors.
        pretrained_tokenizer_name (str): The name of the base Hugging Face tokenizer.
        backbone_type (str | None): The type of backbone model from Hugging Face (e.g., "t5gemma").
        backbone_model_class (str | None): The class name of the backbone model if not using AutoModel.
        backbone_config_class (str | None): The class name of the backbone config.
        backbone_config (Config | None): A configuration for the backbone model.
    """

    def __init__(
        self,
        out_size: int,
        pretrained_tokenizer_name: str,
        model_dir: str,
        backbone_type: str,
        backbone_config: dict,
    ):
        super().__init__()
        # load dictionaries from the model directory
        with open(os.path.join(model_dir, "subword_id_to_char_ids.json"), "r") as fp:
            self.subword_id_to_char_ids = {int(k): v for k, v in json.load(fp).items()}
        with open(os.path.join(model_dir, "char_vocab.json"), "r") as fp:
            self.char_vocab = json.load(fp)
        self.char_padding_idx = len(self.char_vocab)
        # 2. Initialize the backbone model
        config = AutoConfig.for_model(backbone_type, **backbone_config)
        self.backbone = AutoModelForTextEncoding.from_config(config)
        self.hidden_size = self.backbone.get_input_embeddings().weight.size(-1)
        delattr(self.backbone.encoder, "embed_tokens")
        self.embed_tokens = nn.Embedding(
            len(self.char_vocab) + 1,
            self.hidden_size,
            padding_idx=self.char_padding_idx,
        )
        self.proj_embedding = nn.Linear(self.hidden_size, out_size, bias=False)

    def prepare_inputs(
        self, subword_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of subword IDs into a padded batch of character IDs.

        Args:
            subword_ids (Tensor): A tensor of subword IDs. Shape: `[batch, seq_len]`.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - Padded character IDs. Shape: `[num_valid_subwords, max_char_len]`.
                - Lengths of each character sequence. Shape: `[num_valid_subwords]`.
        """
        device = subword_ids.device
        assert (
            subword_ids.size(1) == 1
        ), "Supporting one subword per sequence during generation"
        subword_ids = subword_ids.squeeze(1)
        # Select only the valid subword IDs
        subword_id_list = subword_ids.cpu().tolist()
        # Map each subword ID to its sequence of character IDs
        char_id_list = [
            list(self.subword_id_to_char_ids.get(x, ())) for x in subword_id_list
        ]
        char_lengths = torch.tensor(
            [len(x) for x in char_id_list], dtype=torch.long, device=device
        )
        batch_size = char_lengths.size(0)
        max_len = int(char_lengths.max().item()) if batch_size > 0 else 0
        # Create a padded tensor for the character IDs
        char_ids = torch.full(
            (batch_size, max_len),
            self.char_padding_idx,
            dtype=torch.long,
            device=device,
        )
        for i, char_seq in enumerate(char_id_list):
            char_ids[i, : len(char_seq)] = torch.tensor(
                char_seq, dtype=torch.long, device=device
            )
        return char_ids, char_lengths

    def forward(self, subword_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to get character-aware subword embeddings.
        Args:
            subword_ids (Tensor): A tensor of subword IDs. Shape: `[batch, seq_len]`.

        Returns:
            Tensor: The final subword embeddings. Shape: `[batch, seq_len, hidden_size]`.
        """
        # 1. Convert subword IDs to character IDs
        char_ids, char_lengths = self.prepare_inputs(subword_ids)
        # char_mask = sequence_mask(char_lengths).float()
        char_mask = sequence_mask(char_lengths)
        # 2. Get character embeddings and pass them through the backbone
        char_embeds = self.embed_tokens(char_ids)
        # The backbone model should be able to accept `inputs_embeds`
        char_hidden_states = self.backbone(
            inputs_embeds=char_embeds, attention_mask=char_mask
        ).last_hidden_state
        # 3. Aggregate character embeddings to form subword embeddings (mean pooling)
        # We mask the padding characters before summing to get a correct mean.
        masked_sum = (char_hidden_states * char_mask.unsqueeze(-1)).sum(dim=1)
        # Avoid division by zero for empty sequences
        mean_emb = masked_sum / (char_lengths.unsqueeze(-1).clamp(min=1))
        # 4. Scatter the aggregated embeddings back to the original subword sequence shape
        out_emb = self.proj_embedding(mean_emb)
        return out_emb


def depthsum_embedding(
    code: torch.Tensor, rvq_embeddings: nn.ParameterList,
) -> torch.Tensor:
    """
    Embedds all codes into a single embedding.
    Args:
        code: Tensor (num_quantizers x BT) Acoustic codes to embed and add
        rvq_embeddings: Tensor (num_quantizers x codebook_size x latent_size) RVQ embeddings

    Returns:
        Tensor (BT x latent_size) - embedded codes
    """
    embs = nn.functional.pad(rvq_embeddings, [0, 0, 0, 1]) # num_quantizers x (codebook_size + 1) x latent_size
    res = nn.functional.embedding(code[0], embs[0])
    for i in range(1, len(embs)):
        res = res + nn.functional.embedding(
            code[i], embs[i]
        )
    return res


# module that takes text tokens, audio tokens and prepares input embedding for EarTTS model
class EarTTSInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_code = nn.Linear(
            self.config.latent_size, self.config.hidden_size, bias=False
        )
        self.codebook_size = self.config.codebook_size
        self.rvq_embs = nn.Parameter(torch.empty(self.config.num_quantizers, self.config.codebook_size, self.config.latent_size))
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.context_hidden_size
        )
        self.embed_context = nn.Linear(
            self.config.context_hidden_size, self.config.hidden_size, bias=False
        )
        self.embed_subword = CharAwareSubwordEncoder(
            out_size=self.config.hidden_size,
            pretrained_tokenizer_name=self.config.pretrained_tokenizer_name,
            model_dir=self.config.model_dir,
            backbone_type=self.config.backbone_type,
            backbone_config=OmegaConf.to_container(self.config.backbone_config),
        )
        self.bos_emb = nn.Parameter(torch.empty(self.config.hidden_size))

    def forward(
        self,
        context_text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Works for context and generation phases to prepare the input embedding for vLLM engine.
        At context phase:
            context_text_tokens (T) or (128)
            audio_tokens (num_quantizers x T) or (31 x 128)
        At generation phase
            context_text_tokens (1)
            audio_tokens (num_quantizers x 1) or (31 x 1)
            text_tokens (1)

        Returns:
            embedding of shape (T x dim) or (1 x dim)
        """

        # embed the context text token
        context_emb = self.embed_tokens(context_text_tokens)  # T x dim
        context_emb_proj = self.embed_context(context_emb)  # T x dim

        # embed previously predicted acoustic tokens
        audio_emb = depthsum_embedding(
            audio_tokens, self.rvq_embs
        )  # T x dim
        audio_emb_proj = self.embed_code(audio_emb)  # T x dim

        # for generation phase, also embed current text token using subword encoder,
        # that encodes characters
        if text_tokens is not None:
            text_emb = self.embed_subword(text_tokens.unsqueeze(0)).squeeze(
                0
            )  # T x dim
            return context_emb_proj + audio_emb_proj + text_emb  # T x dim
        else:
            audio_emb_proj[0] += self.bos_emb  # 1 x dim
            return context_emb_proj + audio_emb_proj  # 1 x DIM


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

        self.logits_processor = (
            TopPLogitsWarper(self.top_p_or_k)
            if isinstance(self.top_p_or_k, float)
            else (
                TopKLogitsWarper(self.top_p_or_k)
                if isinstance(self.top_p_or_k, int)
                else None
            )
        )
        self.logits_processor = None

        self.mlp_stack = nn.Sequential(
            *[
                MLPLayer(hidden_size, intermediate_size, eps=eps)
                for _ in range(num_layers)
            ],
            RMSNorm(hidden_size, eps=eps),
        )

        if low_rank is None:
            self.proj_logits = nn.Linear(
                hidden_size, num_predictions, bias=False
            )  # Predicts mixture weights
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * out_size, bias=False
            )  # Predicts means
            self.proj_logs = nn.Linear(
                hidden_size, 1, bias=False
            )  # Predicts log standard deviations
        else:
            assert low_rank < out_size
            self.proj_logits = nn.Linear(
                hidden_size, num_predictions, bias=False
            )  # Predicts mixture weights
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * low_rank, bias=False
            )  # Predicts means
            self.proj_logs = nn.Linear(
                hidden_size, 1, bias=False
            )  # Predicts log standard deviations
            self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
            self.low_mat = nn.Parameter(
                torch.empty(num_predictions, out_size, low_rank)
            )

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
        #mixture_indices = (nn.functional.log_softmax(logits, dim=-1)).argmax(-1)

        # Select the mean corresponding to the sampled component
        mu = batch_matmul(
            x.view(bt, -1),
            self.proj_mus.weight.detach().view(n, d, -1),
            mixture_indices.view(bt),
        ).view(bt, d)
        if self.proj_mus.bias is not None:
            mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

        if self.low_rank:
            # assert math.log2(d).is_integer() and math.log2(self.out_size).is_integer()
            mu = batch_matmul(
                mu.view(bt, -1),
                self.low_mat.detach().view(n, self.out_size, -1),
                mixture_indices.view(bt),
                # TODO: these are the arguments for custom kernel impl
                # BLOCK_SIZE_DIN=d,
                # BLOCK_SIZE_DOUT=self.out_size,
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
        self.noise_scale = self.config.noise_scale

        # pre-compute how many tokens are unmasked at each iteration
        rates = np.linspace(0.0, 1.0, self.config.num_iter + 1)[:-1].reshape(-1, 1)
        masking_rates = np.power(
            1 - np.power(rates, self.config.exponent), 1 / self.config.exponent
        )
        num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(int)
        num_maskings_shifted = np.pad(
            num_maskings[1:], ((0, 1), (0, 0)), constant_values=0
        )
        sampling_per_step = num_maskings - num_maskings_shifted
        sampling_per_step_flat = sampling_per_step.flatten()
        # Drop any values at the beginning that are 0
        first_nonzero = np.argmax(sampling_per_step_flat != 0)
        self.num_to_sample = sampling_per_step_flat[first_nonzero:].tolist()

        # create layers used outside of backbone
        # Store as Parameters so they can be used as tensors directly
        self.rvq_embs = nn.Parameter(torch.empty(self.config.num_quantizers, self.config.codebook_size, self.config.latent_size))
        self.padding_idx = self.codebook_size
        self.embed_code = nn.Linear(
            self.config.latent_size, self.config.hidden_size, bias=False
        )
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
        #return hidden_states
        return codes

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
        return depthsum_embedding(code, self.rvq_embs)

    def _depthsum_encoding_step_reshaped(
        self,
        r: torch.Tensor,  # [B*T, hidden_size]
        code: torch.Tensor,  # [num_quantizers, B*T]
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
                self.rvq_embs[i].pow(2).sum(-1)  # [vocab_size]
                - 2 * (r @ self.rvq_embs[i].T)  # [B*T, vocab_size]
            ).argmin(
                -1
            )  # [B*T]

            # Update residual
            emb_i = nn.functional.embedding(
                idx_sel, self.rvq_embs[i], #padding_idx=self.padding_idx
            )  # [B*T, latent_size]
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
        code = (
            torch.zeros(
                (self.num_quantizers, hidden_states.shape[0]),
                dtype=torch.long,
                device=device,
            )
            + self.codebook_size
        )
        # Iteratively unmask the continuous part of the code
        cnt = 0
        for k in self.num_to_sample:
            # Prepare input for the MoG head
            mog_input_embeds = self.embed_code(
                self._depthsum_embedding(code)
            )  # (BT x hidden_size)
            mog_input_embeds += hidden_states

            mog_mu, mog_logs = self.mog_head(
                mog_input_embeds,
            )
            z = (
                mog_mu
                + torch.exp(mog_logs) * torch.randn_like(mog_mu) * self.noise_scale
            )
            code = self._depthsum_encoding_step_reshaped(z, code, cnt, k)
            cnt += k
        return code
