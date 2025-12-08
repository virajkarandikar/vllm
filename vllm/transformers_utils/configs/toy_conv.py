


from typing import Optional
from transformers import PretrainedConfig


class ToyConvConfig(PretrainedConfig):
    model_type = "toy_conv"

    def __init__(
        self,
        vocab_size: int = 1,
        d_model: int = 128,
        k_conv: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k_conv = k_conv
        super().__init__(**kwargs)