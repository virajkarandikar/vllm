from transformers import PretrainedConfig


class FastConformerCTCConfig(PretrainedConfig):
    model_type = "fastconformer_ctc"

    def __init__(
        self,
        hidden_size: int = 80*8, # mel*8
        d_model: int = 512,
        n_layers: int = 17,
        n_heads: int = 8,
        ff_mult: int = 4,
        k_conv: int = 9,
        subsampling: dict = None,
        att_left_ctx: int = 70,
        att_right_ctx: int = 1,
        use_bias: bool = True,
        norm_type: str = "batch_norm",
        xscale: bool = True,
        ctc: dict = None,
        blank_id: int = 0,
        vocab_size: int = 1024,
        frontend: dict = None,
        tokenizer: dict = None,
        notes: str = "",
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_attention_heads = n_heads
        self.ff_mult = ff_mult
        self.k_conv = k_conv
        self.subsampling = subsampling if subsampling is not None else {}
        self.att_left_ctx = att_left_ctx
        self.att_right_ctx = att_right_ctx
        self.use_bias = use_bias
        self.norm_type = norm_type
        self.xscale = xscale
        self.ctc = ctc if ctc is not None else {}
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.frontend = frontend if frontend is not None else {}
        self.tokenizer = tokenizer if tokenizer is not None else {}
        self.notes = notes

        super().__init__(**kwargs)
