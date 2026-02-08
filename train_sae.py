import argparse
import torch
import torch.nn as nn
import json


class SparseAutoencoder(nn.Module):
    def __init__(self, input_features, dict_features):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_features, dict_features)
        self.decoder = nn.Linear(dict_features, input_features)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


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

    # Flatten the activations
    num_samples, C, H, W = activations.shape
    input_features = C * H * W
    activations = activations.view(num_samples, -1)
    print(f"  - Flattened shape: {activations.shape}")

    # TODO: Make this configurable
    # For an SAE, the dictionary size is typically much larger than the input size.
    dict_features = input_features * 4

    print(f"\nInitializing SAE model...")
    print(f"  - Input features: {input_features}")
    print(f"  - Dictionary features: {dict_features}")
    model = SparseAutoencoder(input_features, dict_features)
    print(model)

    print("\nModel initialized. Next steps are to implement the training loop.")
    

if __name__ == "__main__":
    main()
