import argparse
import torch
import json


def main():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on KataGo activations.")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations and metadata.")
    args = parser.parse_args()
    
    print(f"Loading data from {args.activations_file}...")
    data = torch.load(args.activations_file)
    
    metadata = data["metadata"]
    activations = data["activations"]
    
    print("\n--- Successfully loaded data ---")
    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    
    print("\nActivations Tensor Info:")
    print(f"  - Shape: {activations.shape}")
    print(f"  - DType: {activations.dtype}")
    
    print("\nThis is a placeholder for the SAE training script.")
    print("The next step would be to define an SAE model and train it on these activations.")
    

if __name__ == "__main__":
    main()
