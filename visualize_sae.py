import argparse
import torch
import torch.nn as nn
import random
import sys

# The SAE model architecture must match the one used for training.
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
    parser = argparse.ArgumentParser(description="Visualize a trained Sparse Autoencoder.")
    parser.add_argument("--sae-model-file", required=True, help="Path to the trained SAE model (.pt).")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Activations ---
    print(f"\nLoading activations from {args.activations_file}...")
    data = torch.load(args.activations_file, map_location=device)
    activations = data["activations"]
    print(f"  - Activations shape: {activations.shape}")

    # Reshape activations just like in training
    num_samples, C, H, W = activations.shape
    input_features = C
    activations = activations.permute(0, 2, 3, 1).contiguous()
    activations = activations.view(-1, C)
    print(f"  - Reshaped for SAE: {activations.shape}")

    # --- Load SAE Model ---
    # Infer dictionary size from the model file's state dict
    state_dict = torch.load(args.sae_model_file, map_location=device)
    input_features_from_model = state_dict['decoder.weight'].shape[1]
    dict_features = state_dict['encoder.weight'].shape[0]
    
    if input_features_from_model != input_features:
        print(f"Error: Model's input features ({input_features_from_model}) do not match activations' features ({input_features}).", file=sys.stderr)
        return
        
    print(f"\nLoading SAE model from {args.sae_model_file}...")
    model = SparseAutoencoder(input_features, dict_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  - Model loaded successfully.")
    print(model)

    # --- Analysis ---
    print("\n--- Running analysis on a random activation vector ---")

    # Select a random activation vector
    random_idx = random.randint(0, activations.shape[0] - 1)
    random_activation = activations[random_idx].unsqueeze(0) # Add batch dimension
    print(f"Selected random activation vector at index {random_idx}")

    # Run through the model
    with torch.no_grad():
        reconstructed_activation, encoded_features = model(random_activation)

    # Calculate reconstruction error (MSE)
    mse = nn.MSELoss()(random_activation, reconstructed_activation).item()
    print(f"\nReconstruction Mean Squared Error (MSE): {mse:.6f}")

    # Analyze the sparse encoded features
    encoded_features = encoded_features.squeeze(0) # Remove batch dim
    num_active_features = (encoded_features > 0).sum().item()
    total_features = encoded_features.shape[0]
    sparsity = (num_active_features / total_features) * 100
    
    print(f"\nEncoded features analysis:")
    print(f"  - Sparsity: {num_active_features} / {total_features} features are active ({sparsity:.2f}%)")

    if num_active_features > 0:
        # Find the most active feature
        max_val, max_idx = torch.max(encoded_features, 0)
        print(f"  - Most active feature index: {max_idx.item()} (value: {max_val.item():.4f})")
    
        # Print top 5 active features
        print("  - Top 5 active features (index: value):")
        top_k_vals, top_k_indices = torch.topk(encoded_features, k=min(5, num_active_features))
        for i in range(len(top_k_vals)):
            print(f"    - {top_k_indices[i].item()}: {top_k_vals[i].item():.4f}")
    else:
        print("  - No features were active for this input.")


if __name__ == "__main__":
    main()
