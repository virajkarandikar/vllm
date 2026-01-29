# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from transformers import PretrainedConfig


class EarTTSConfig(PretrainedConfig):
    model_type = "eartts"
    
    def __init__(
        self,
        # gemma 3 config
        hidden_size: int = 1152,
        context_hidden_size: int = 1536,
        intermediate_size: int = 4608,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        vocab_size: int = 1,
        max_position_embeddings: int = 8192,

        # custom config, related sampling
        num_quantizers: int = 31,
        codebook_size: int = 1024,
        num_iter: int = 8,
        top_p_or_k: float = 0.8,
        noise_scale: float = 0.8,
        exponent: float = 3.0,
        latent_size: int = 512,
        mog_low_rank: int = 64,
        mog_num_layers: int = 3,
        mog_num_predictions: int = 1024,
        mog_min_log_std: float = -4.0,
        mog_eps: float = 1e-6,
        enable_guidance: bool = False,

        # Gemma3-specific attributes required by Gemma3Model
        query_pre_attn_scalar: float = 256.0,  # Default attention scaling
        attention_bias: bool = False,  # Gemma models typically don't use attention bias
        rms_norm_eps: float = 1e-6,  # RMS normalization epsilon
        layer_types: Optional[list] = None,  # Layer types ("global_attention" or "sliding_attention")
        sliding_window: Optional[int] = 4096,  # Sliding window size for local attention
        rope_local_base_freq: float = 10000.0,  # RoPE base frequency for local attention
        rope_theta: float = 10000.0,  # RoPE theta for global attention
        rope_scaling: Optional[dict] = None,  # RoPE scaling configuration
        hidden_activation: str = "gelu_pytorch_tanh",  # Activation function (required by Gemma3)
        tie_word_embeddings: bool = True,  # Whether embeddings are tied
        final_logit_softcapping: Optional[float] = None,  # Final logit softcapping
        attn_logits_soft_cap: Optional[float] = None,  # Attention logits softcapping
        use_bidirectional_attention: bool = False,  # Whether to use bidirectional attention (False for causal)
        is_causal: bool = True,  # Whether the model is causal

        # subword encoding config
        emb_backbone_config: dict = None,
        emb_backbone_type: str = "t5gemma",
        max_char_len: int = 128,
        emb_char_vocab_size: int = 256,
        emb_vocab_size: int = 151936,
        pretrained_tokenizer_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        use_subword_flag_emb: bool = True,
        use_bos_eos_emb: bool = True,
        use_gated_fusion_for_text_audio: bool = True,
        **kwargs,
    ):
        # gemma 3 config
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        
        # custom config, related sampling
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.num_iter = num_iter
        self.top_p_or_k = top_p_or_k
        self.noise_scale = noise_scale
        self.exponent = exponent
        self.latent_size = latent_size
        self.mog_low_rank = mog_low_rank
        self.mog_num_layers = mog_num_layers
        self.mog_num_predictions = mog_num_predictions
        self.mog_min_log_std = mog_min_log_std
        self.mog_eps = mog_eps
        self.enable_guidance = enable_guidance

        # Gemma3-specific attributes
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        # Default all layers to global attention if not specified
        self.layer_types = layer_types if layer_types is not None else ["global_attention"] * num_hidden_layers
        self.sliding_window = sliding_window
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_activation = hidden_activation
        self.tie_word_embeddings = tie_word_embeddings
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.use_bidirectional_attention = use_bidirectional_attention
        self.is_causal = is_causal

        # subword encoding config
        self.emb_backbone_config = emb_backbone_config
        self.emb_backbone_type = emb_backbone_type
        self.max_char_len = max_char_len
        self.emb_char_vocab_size = emb_char_vocab_size
        self.emb_vocab_size = emb_vocab_size

        self.pretrained_tokenizer_name = pretrained_tokenizer_name
        self.use_subword_flag_emb = use_subword_flag_emb
        self.use_bos_eos_emb = use_bos_eos_emb
        self.use_gated_fusion_for_text_audio = use_gated_fusion_for_text_audio
        
        super().__init__(**kwargs)

