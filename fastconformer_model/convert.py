import tarfile
import torch
import os
import shutil
import argparse
from safetensors.torch import save_file

def convert_nemo_to_safetensors(nemo_file_path, output_file_path="model.safetensors"):
    print(f"Processing: {nemo_file_path}")
    
    # Create a temporary directory for extraction
    temp_dir = "temp_nemo_extraction"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Unpack the .nemo tarball
        with tarfile.open(nemo_file_path, "r:") as tar:
            tar.extractall(path=temp_dir)

        config_path = os.path.join(temp_dir, "model_config.yaml")
        with open(config_path, "r") as fp:
            for line in fp:
                print(line.strip())
            
        # 2. Locate the weights file
        # NeMo usually stores weights in 'model_weights.ckpt'
        ckpt_path = os.path.join(temp_dir, "model_weights.ckpt")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Could not find 'model_weights.ckpt' inside {nemo_file_path}")
            
        print("Loading checkpoint into memory (this may take a moment)...")
        
        # 3. Load the PyTorch Checkpoint
        # map_location='cpu' prevents OOM errors if you don't have a massive GPU available
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # 4. Extract the State Dictionary
        # PyTorch Lightning checkpoints store weights under the 'state_dict' key
        if "state_dict" in checkpoint:
            weights = checkpoint["state_dict"]
        else:
            # Fallback if the file is just the raw dict
            weights = checkpoint
            
        print(f"Converting {len(weights)} tensors...")

        # 5. Save as SafeTensors
        # We save directly to the output path
        for k in weights.keys():
            print(k)
        save_file(weights, output_file_path)
        
        print(f"Success! Weights saved to: {output_file_path}")

    finally:
        # 6. Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .nemo file to .safetensors")
    parser.add_argument("input_file", help="Path to the input .nemo file")
    parser.add_argument("--output", default="model.safetensors", help="Path to the output .safetensors file")
    
    args = parser.parse_args()
    
    convert_nemo_to_safetensors(args.input_file, args.output)
