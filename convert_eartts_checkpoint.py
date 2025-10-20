# Copyright 2025 NVIDIA. All rights reserved.

import os
import json
import yaml
import argparse

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # load config
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    # config modification that is needed to run inference
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.data.add_audio_prompt_after_description = True
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.model.subword_mask_exactly_as_eartts = False
    cfg.model.context_hidden_mask_exactly_as_eartts = False
    cfg.model.tts_config.disable_eos_prediction = True
    cfg.model.inference_force_speech_silence_on_eos = True
    cfg.model.use_word_sep_tokenizer = False
    cfg.model.tts_config.use_subword_flag_emb = False
    cfg.model.num_delay_speech_tokens = 0
    cfg.data.source_sample_rate = 22050
    cfg.data.target_sample_rate = 22050
    model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True)).eval()

    # load checkpoint
    weights = torch.load(args.ckpt)["state_dict"]

    # create weights for the embedding model that runs outside of the eartts
    embedding_module_weights = {}
    embedding_module_weights["bos_emb"] = weights["tts_model.bos_emb"]
    embedding_module_weights["embed_code.weight"] = weights[
        "tts_model.embed_code.weight"
    ]
    embedding_module_weights["rvq_embs"] = weights["tts_model.rvq_embs"]
    embedding_module_weights["embed_tokens.weight"] = model.embed_tokens.weight
    embedding_module_weights["embed_context.weight"] = weights[
        "tts_model.embed_context.weight"
    ]
    # embedding transformer has a lot of weights
    for key, weight in weights.items():
        if "tts_model.embed_subword" in key:
            key = key[len("tts_model.") :]
            embedding_module_weights[key] = weight
    # save to .ckpt file so that a torch module can load weights from it
    torch.save(
        embedding_module_weights,
        os.path.join(args.outdir, "eartts_input_embedding.ckpt"),
    )
    print(f"Created ckpt for embedding module")

    # create config for embedding module
    embedding_module_config = {}
    for key in [
        "latent_size",
        "codebook_size",
        "num_quantizers",
        "context_hidden_size",
    ]:
        embedding_module_config[key] = cfg.model.tts_config[key]
    for key in ["pretrained_tokenizer_name", "backbone_type"]:
        embedding_module_config[key] = cfg.model.tts_config.cas_config[key]
    embedding_module_config["backbone_config"] = OmegaConf.to_container(
        cfg.model.tts_config.cas_config.backbone_config, resolve=True
    )
    embedding_module_config["hidden_size"] = (
        cfg.model.tts_config.backbone_config.hidden_size
    )
    embedding_module_config["vocab_size"] = model.embed_tokens.weight.shape[0]
    with open(
        os.path.join(args.outdir, "eartts_input_embedding_config.yaml"), "w"
    ) as f:
        yaml.safe_dump(embedding_module_config, f, indent=2, default_flow_style=False)
    print(f"Created config for embedding module")

    # drop unused weights
    unused_keys = ["bos_emb", "null_emb", "embed_subword", "embed_context"]
    unused_keys = [f"tts_model.{k}" for k in unused_keys]
    weights = {
        k: v
        for k, v in weights.items()
        if all(not k.startswith(uk) for uk in unused_keys)
    }
    # filter weights for vLLM model
    weights = {
        k.replace("tts_model.", ""): v for k, v in weights.items() if "tts_model." in k
    }
    # create minimal dummy weights for embedding layer inside gemma3 backbone
    # it is not getting used so size and type do not matter
    hidden_size = cfg.model.tts_config.backbone_config.hidden_size
    weights["backbone.embed_tokens.weight"] = torch.randn(1, hidden_size).to(
        torch.float16
    )

    # save weights
    safetensors_path = os.path.join(args.outdir, "model.safetensors")
    save_file(weights, safetensors_path)
    print(f"Saved weights for vllm model")
    weight_map = {name: "model.safetensors" for name in weights.keys()}
    index = {
        "metadata": {
            "total_size": sum(w.numel() * w.element_size() for w in weights.values())
        },
        "weight_map": weight_map,
    }
    index_path = os.path.join(args.outdir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved model index")

    # save config.json
    flat_config = {"architectures": ["EarTTSForCausalLM"], "model_type": "eartts"}
    # not using vocab size of the backbone model
    flat_config["vocab_size"] = 1
    # forward backbone configs
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ]:
        flat_config[key] = cfg.model.tts_config.backbone_config[key]
    # forward overall configs
    for key in ["latent_size", "codebook_size", "num_quantizers", "exponent"]:
        flat_config[key] = cfg.model.tts_config[key]
    # forward mog head configs
    for key in ["num_layers", "low_rank", "num_predictions", "min_log_std", "eps"]:
        flat_config[f"mog_{key}"] = cfg.model.tts_config.mog_head_config[key]
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(flat_config, f, indent=2)
    print("Saved vllm config")

    # save subword encoder vocabs and config
    subword_id_to_char_ids = model.tts_model.embed_subword.subword_id_to_char_ids
    char_vocab = model.tts_model.embed_subword.char_vocab
    with open(os.path.join(args.outdir, "subword_id_to_char_ids.json"), "w") as f:
        json.dump(subword_id_to_char_ids, f, indent=2)
    with open(os.path.join(args.outdir, "char_vocab.json"), "w") as f:
        json.dump(char_vocab, f, indent=2)
    print("Saved vocabs for char encoding")


if __name__ == "__main__":
    main()
