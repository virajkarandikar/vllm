# Copyright 2025 NVIDIA. All rights reserved.

import os
import json
import argparse
import tarfile
import shutil

import torch
import yaml
from safetensors.torch import save_file


def convert(nemo_path: str, outdir: str):
    """Convert a .nemo file to vLLM format.
    
    Args:
        nemo_path: Path to the input .nemo file
        outdir: Path to output directory
    """
    os.makedirs(outdir, exist_ok=True)

    print(f"Processing: {nemo_path}")

    # Create a temporary directory for extraction
    temp_dir = "temp_nemo_extraction"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Unpack the .nemo tarball
        with tarfile.open(nemo_path, "r:") as tar:
            tar.extractall(path=temp_dir)

        # Parse model_config.yaml
        config_path = os.path.join(temp_dir, "model_config.yaml")
        with open(config_path, "r") as fp:
            nemo_config = yaml.safe_load(fp)
            print("Loaded NeMo config:")
            for key in nemo_config.keys():
                print(f"  - {key}")

        # Extract encoder config
        encoder_cfg = nemo_config.get("encoder", {})
        preprocessor_cfg = nemo_config.get("preprocessor", {})
        decoder_cfg = nemo_config.get("decoder", {})

        # Calculate input dimension from preprocessor
        subsampling_factor = 8
        hop_length = int(preprocessor_cfg.get("sample_rate") * preprocessor_cfg.get("window_stride"))
        input_dim = subsampling_factor * hop_length

        # Get vocab size from decoder
        vocab_size = decoder_cfg.get("vocabulary", [])
        if isinstance(vocab_size, list):
            vocab_size = len(vocab_size)

        # Locate the weights file
        ckpt_path = os.path.join(temp_dir, "model_weights.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Could not find 'model_weights.ckpt' inside {nemo_path}"
            )

        print("Loading checkpoint into memory (this may take a moment)...")

        # Load the PyTorch Checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Extract the State Dictionary
        if "state_dict" in checkpoint:
            weights = checkpoint["state_dict"]
        else:
            weights = checkpoint

        print(f"Converting {len(weights)} tensors...")

        # Save as SafeTensors
        safetensors_path = os.path.join(outdir, "model.safetensors")
        for k in weights.keys():
            print(k)
        save_file(weights, safetensors_path)
        print(f"Saved weights for vllm model")

        # Save model index
        weight_map = {name: "model.safetensors" for name in weights.keys()}
        index = {
            "metadata": {
                "total_size": sum(w.numel() * w.element_size() for w in weights.values())
            },
            "weight_map": weight_map,
        }
        index_path = os.path.join(outdir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Saved model index")

        # Build and save config.json
        att_context_size = encoder_cfg.get("att_context_size")
        expected_ctx = [70, 0]
        assert expected_ctx in att_context_size, f"vllm impl of fastconformer is developed for {str(expected_ctx)} context size"

        flat_config = {
            "architectures": ["FastConformerCTC"],
            "model_type": "fastconformer_ctc",
            "hidden_size": encoder_cfg.get("d_model", 1024),
            "d_model": encoder_cfg.get("d_model", 1024),
            "n_layers": encoder_cfg.get("n_layers", 24),
            "n_heads": encoder_cfg.get("n_heads", 8),
            "ff_mult": encoder_cfg.get("ff_expansion_factor", 4),
            "k_conv": encoder_cfg.get("conv_kernel_size", 9),
            "att_left_ctx": expected_ctx[0],
            "att_right_ctx": expected_ctx[1],
            "use_bias": encoder_cfg.get("use_bias", False),
            "norm_type": "layer_norm",
            "xscale": encoder_cfg.get("xscaling", False),
            "vocab_size": vocab_size if vocab_size else 1024,
            "subsampling": {
                "hop_length": hop_length,
                "subsampling_factor": subsampling_factor,
            },
            "custom_input_specs": [
                {
                    "name": "audio",
                    "dim": input_dim
                }
            ],
            "custom_outputs": ["acoustic_emb"]
        }

        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(flat_config, f, indent=2)
        print("Saved vllm config")

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert .nemo file to vLLM format")
    parser.add_argument("--nemo", type=str, required=True, help="Path to the input .nemo file")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    convert(args.nemo, args.outdir)


if __name__ == "__main__":
    main()
