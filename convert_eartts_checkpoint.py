# Copyright 2025 NVIDIA. All rights reserved.

import os
import json
import argparse

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

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

    # load checkpoint
    weights = torch.load(args.ckpt)["state_dict"]
    # filter weights
    weights = {
        k.replace("tts_model.", ""): v for k, v in weights.items() if "tts_model." in k
    }

    def is_unused_weight(k):
        return any(
            k.startswith(prefix)
            for prefix in ["bos_emb", "null_emb", "embed_subword", "embed_context"]
        )

    weights = {k: v for k, v in weights.items() if not is_unused_weight(k)}
    # rename rvq embeddings
    rvq_embs = weights.pop("rvq_embs")
    # pad rvq embeddings to
    # num_quantizers x (codebook_size + 1) x hidden_size
    rvq_embs = torch.nn.functional.pad(rvq_embs, [0, 0, 0, 1])
    rvq_embs_lst = [x.squeeze(0) for x in torch.split(rvq_embs, 1, dim=0)]
    for i, emb in enumerate(rvq_embs_lst):
        weights[f"rvq_embeddings.{i}"] = emb
    # create minimal dummy weights for embedding layer inside gemma3 backbone
    hidden_size = cfg.model.tts_config.backbone_config.hidden_size
    weights["backbone.embed_tokens.weight"] = (torch.randn(1, hidden_size) * 0.02).to(
        rvq_embs.dtype
    )

    # save weights
    os.makedirs(args.outdir, exist_ok=True)
    safetensors_path = os.path.join(args.outdir, "model.safetensors")
    save_file(weights, safetensors_path)
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


if __name__ == "__main__":
    main()
