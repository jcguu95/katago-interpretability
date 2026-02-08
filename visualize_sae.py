import argparse
import torch
import torch.nn as nn
import random
import sys
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sae_model import SparseAutoencoder

def main():
    parser = argparse.ArgumentParser(description="Visualize a trained Sparse Autoencoder.")
    parser.add_argument("--sae-model-file", required=True, help="Path to the trained SAE model (.pt).")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for analysis.")
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
    print(f"\nLoading SAE model from {args.sae_model_file}...")
    checkpoint = torch.load(args.sae_model_file, map_location=device)

    hyperparameters = checkpoint['hyperparameters']
    state_dict = checkpoint['model_state_dict']

    input_features_from_model = hyperparameters['input_features']
    dict_features = hyperparameters['dict_features']

    if input_features_from_model != input_features:
        print(f"Error: Model's input features ({input_features_from_model}) do not match activations' features ({input_features}).", file=sys.stderr)
        return
        
    model = SparseAutoencoder(input_features, dict_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  - Model loaded successfully.")

    print("\n--- Model Hyperparameters ---")
    # Pretty print the hyperparameters from the training run
    print(json.dumps(hyperparameters, indent=2))
    print(model)

    # --- Global Feature Analysis ---
    print("\n--- Running global analysis on all activation vectors ---")

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    num_total_samples = len(dataset)
    
    freq_counts = torch.zeros(dict_features, device=device)
    summed_magnitudes = torch.zeros(dict_features, device=device)
    max_magnitudes = torch.zeros(dict_features, device=device)

    print("Processing activations to calculate feature statistics...")
    with torch.no_grad():
        for batch in tqdm(dataloader, unit="batch"):
            inputs = batch[0].to(device)
            _, encoded_features = model(inputs)

            active_mask = encoded_features > 0
            freq_counts += active_mask.sum(dim=0)
            
            # To get avg magnitude *when active*, we sum up all magnitudes and divide by frequency.
            summed_magnitudes += encoded_features.sum(dim=0)

            # Find max magnitude for each feature in the batch and update global max
            if encoded_features.shape[0] > 0:
                batch_max_magnitudes, _ = torch.max(encoded_features, dim=0)
                max_magnitudes = torch.max(max_magnitudes, batch_max_magnitudes)

    activation_frequency = freq_counts / num_total_samples
    # Clamp to avoid division by zero for features that never activate.
    avg_magnitude_when_active = summed_magnitudes / freq_counts.clamp(min=1)

    print("\n--- Global Feature Statistics ---")

    # --- Top 5 Features by Activation Frequency ---
    print("\nTop 5 Features by Activation Frequency:")
    top_freq_vals, top_freq_indices = torch.topk(activation_frequency, 5)
    for i in range(len(top_freq_vals)):
        idx = top_freq_indices[i].item()
        freq = top_freq_vals[i].item() * 100
        avg_mag = avg_magnitude_when_active[idx].item()
        max_mag = max_magnitudes[idx].item()
        print(f"  - Feature {idx}: Activated in {freq:.2f}% of samples. Avg Mag: {avg_mag:.4f}, Max Mag: {max_mag:.4f}")

    # --- Top 5 Features by Average Magnitude ---
    print("\nTop 5 Features by Average Magnitude (when active):")
    top_mag_vals, top_mag_indices = torch.topk(avg_magnitude_when_active, 5)
    for i in range(len(top_mag_vals)):
        idx = top_mag_indices[i].item()
        freq = activation_frequency[idx].item() * 100
        avg_mag = top_mag_vals[i].item()
        max_mag = max_magnitudes[idx].item()
        print(f"  - Feature {idx}: Avg Mag: {avg_mag:.4f}. Activated in {freq:.2f}% of samples, Max Mag: {max_mag:.4f}")

    # --- Overall Sparsity ---
    avg_active_features = freq_counts.sum() / num_total_samples
    total_features = dict_features
    sparsity = (avg_active_features / total_features) * 100
    print(f"\nOverall Average Sparsity:")
    print(f"  - On average, {avg_active_features:.2f} / {total_features} features are active per sample ({sparsity:.2f}%)")


if __name__ == "__main__":
    main()
