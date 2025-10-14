# reference nemo config: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml
# this script converts to an hf config compatible with vLLM

import json
import os
import sys
from collections import OrderedDict

from safetensors.torch import save_file

try:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
except Exception as e:
    print(
        "ERROR: This script needs NeMo (nemo_toolkit) and omegaconf installed.\n"
        "  pip install nemo_toolkit[asr] omegaconf safetensors torch sentencepiece\n"
        f"Upstream import error: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


def _cfg_get(cfg, path, default=None):
    print(f"attempting to get {path}")
    try:
        val = OmegaConf.select(cfg, path)
        print(f"got {val} from {path}")
        return default if val is None else val
    except Exception:
        print(f"error getting {path}")
        return default


def _extract_frontend(cfg):
    sr = _cfg_get(cfg, "preprocessor.sample_rate", None)
    n_fft = _cfg_get(cfg, "preprocessor.n_fft", None)
    n_mels = _cfg_get(cfg, "preprocessor.features", None)

    # not sure if these are needed or important
    fmin = _cfg_get(cfg, "preprocessor.lowfreq", 0) # not found
    fmax = _cfg_get(cfg, "preprocessor.highfreq", None) # not found

    win_length = _cfg_get(cfg, "preprocessor.window_size", None)
    hop_length = _cfg_get(cfg, "preprocessor.window_stride", None)
    log_mel = True

    return {
        "sample_rate": sr,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "fmin": fmin,
        "fmax": fmax,
        "win_length_sec": win_length,
        "hop_length_sec": hop_length,
        "log_mel": log_mel,
    }


def _extract_encoder_cfg(cfg, vocab_size):
    d_model = _cfg_get(cfg, "encoder.d_model", None)
    n_layers = _cfg_get(cfg, "encoder.n_layers", None)
    n_heads = _cfg_get(cfg, "encoder.n_heads", None)
    ff_mult = _cfg_get(cfg, "encoder.ff_expansion_factor", None)
    k_conv = _cfg_get(cfg, "encoder.conv_kernel_size", None)

    subs_type     = _cfg_get(cfg, "encoder.subsampling", None)
    subs_factor   = _cfg_get(cfg, "encoder.subsampling_factor", None)
    subs_channels = _cfg_get(cfg, "encoder.subsampling_conv_channels", None)

    att_ctx = _cfg_get(cfg, "encoder.att_context_size", None)
    att_ctx = list(att_ctx)
    assert len(att_ctx) == 2
    att_left_ctx = att_ctx[0]
    att_right_ctx = att_ctx[1]
    # mainly we care that these values aren't -1. we expect windowed attention
    assert att_left_ctx == 70
    assert att_right_ctx == 1

    return {
        "model_type": "fastconformer_ctc",
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ff_mult": ff_mult,
        "k_conv": k_conv,
        "subsampling": {
            "type": subs_type,
            "factor": subs_factor,
            "channels": subs_channels,
            "total_stride": 8
        },
        "att_left_ctx": att_left_ctx,
        "att_right_ctx": att_right_ctx,
        "ctc": {
            "vocab_size": vocab_size
        }
    }


def _maybe_find_ctc_head_keys(sd):
    candidates = [
        ("decoder.ctc.decoder.weight", "decoder.ctc.decoder.bias"),
        ("decoder.decoder_layers.0.fc.weight", "decoder.decoder_layers.0.fc.bias"),
        ("ctc_decoder.proj.weight", "ctc_decoder.proj.bias"),
        ("ctc.proj.weight", "ctc.proj.bias"),
        ("char_classifier.proj.weight", "char_classifier.proj.bias"),
    ]
    for w, b in candidates:
        if w in sd and b in sd:
            print(f"found ctc head keys: {w}, {b}")
            return w, b
    w = next((k for k in sd.keys() if "ctc" in k and k.endswith("weight")), None)
    b = next((k for k in sd.keys() if "ctc" in k and k.endswith("bias")), None)
    return w, b


def _guess_vocab_size(model, sd):
    for k in ["ctc_decoder.proj.weight", "ctc.proj.weight",
              "decoder.ctc.decoder.weight", "char_classifier.proj.weight"]:
        if k in sd:
            return sd[k].shape[0]
    try:
        return int(model.tokenizer.vocab_size)
    except Exception:
        pass
    for k in sd:
        if k.endswith("pred_embed.weight"):
            return sd[k].shape[0]
    raise RuntimeError("Could not determine vocab size.")


def _dump_spm_from_loaded(model, out_dir):
    spm_out = os.path.join(out_dir, "sentencepiece.bpe.model")
    try:
        tok_wrap = getattr(model, "tokenizer", None)
        spp = getattr(tok_wrap, "tokenizer", None) or getattr(tok_wrap, "_tokenizer", None)
        if spp is not None and hasattr(spp, "serialized_model_proto"):
            data = spp.serialized_model_proto()
            with open(spm_out, "wb") as f:
                f.write(data)
            return spm_out
        raise RuntimeError("error")
    except Exception:
        pass
    return None



def _get_spm_path(model, out_dir):
    spm_path = _dump_spm_from_loaded(model, out_dir)
    if spm_path is None:
        raise RuntimeError("error")
    return spm_path


def convert(nemo_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Restoring NeMo model from: {nemo_path}")
    model = nemo_asr.models.EncDecHybridRNNTCTCModel.restore_from(nemo_path, map_location="cpu")
    model.eval()

    cfg = getattr(model, "_cfg", getattr(model, "cfg", None))
    if cfg is None:
        raise RuntimeError("Could not access model._cfg / model.cfg")

    sd = model.state_dict()

    spm_path = _get_spm_path(
        model, out_dir
    )

    blank_id = int(_cfg_get(cfg, "decoding.blank_id", 0)) # not found

    vocab_size = _guess_vocab_size(model, sd)
    assert spm_path is not None
    print(f"[INFO] BPE (SentencePiece) detected. V={vocab_size} | blank_id={blank_id}")
    print(f"[OK] SentencePiece model: {spm_path}")
    tokenizer_meta = {"type": "sentencepiece", "path": "sentencepiece.bpe.model"}

    encoder_cfg = _extract_encoder_cfg(cfg, vocab_size)
    frontend = _extract_frontend(cfg)

    config = {
        "architectures": ["FastConformerCTC"],
        "model_type": encoder_cfg.get("model_type"),
        # mel*8
        "hidden_size": 80*8,
        "d_model": encoder_cfg.get("d_model"),
        "n_layers": encoder_cfg.get("n_layers"),
        "n_heads": encoder_cfg.get("n_heads"),
        "ff_mult": encoder_cfg.get("ff_mult"),
        "k_conv": encoder_cfg.get("k_conv"),
        "subsampling": encoder_cfg.get("subsampling"),
        "att_left_ctx": encoder_cfg.get("att_left_ctx"),
        "att_right_ctx": encoder_cfg.get("att_right_ctx"),
        "ctc": encoder_cfg.get("ctc"),
        "blank_id": blank_id,
        "vocab_size": vocab_size,
        "frontend": frontend,
        "tokenizer": tokenizer_meta,
        "notes": "Weights use original NeMo parameter names. Map at load time in vLLM backend."
    }

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Wrote {config_path}")

    ctc_w, ctc_b = _maybe_find_ctc_head_keys(sd)
    if ctc_w is None or ctc_b is None:
        print("[WARN] Could not positively identify CTC head params in state_dict. "
              "If your model is RNNT-only, you must add a CTC head or switch to RNNT decoding.",
              file=sys.stderr)
    else:
        print(f"[INFO] Found CTC head candidates: {ctc_w}, {ctc_b} "
              f"with shapes {tuple(sd[ctc_w].shape)}, {tuple(sd[ctc_b].shape)}")

    weights_path = os.path.join(out_dir, "model.safetensors")
    save_file(OrderedDict(sd), weights_path)
    print(f"[OK] Wrote {weights_path}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    input_nemo_pth = "/home/scratch.jdaw_coreai/landrew/stt_en_fastconformer_hybrid_large_streaming_80ms_v1.20.0/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo"
    output_nemo_pth = "/home/scratch.jdaw_coreai/landrew/fastconformer_hf/"
    convert(input_nemo_pth, output_nemo_pth)
