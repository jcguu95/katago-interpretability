import argparse
import torch
from tqdm import tqdm
import json
from sae_model import SparseAutoencoder

def find_source_from_index(index, source_map, H, W):
    """
    Finds the SGF file, move number, and board location for a given activation vector index.
    """
    vectors_per_state = H * W
    state_index = index // vectors_per_state
    
    vector_in_state_idx = index % vectors_per_state
    row = vector_in_state_idx // W
    col = vector_in_state_idx % W

    cumulative_states = 0
    for entry in source_map:
        num_states_in_sgf = entry['num_states']
        if state_index < cumulative_states + num_states_in_sgf:
            move_number = state_index - cumulative_states
            return entry['sgf_file'], move_number, (row, col)
        cumulative_states += num_states_in_sgf
    return "Unknown", -1, (-1, -1)

def main():
    parser = argparse.ArgumentParser(description="Find game states that maximally activate a given SAE feature.")
    parser.add_argument("--sae-model-file", required=True, help="Path to the trained SAE model (.pt).")
    parser.add_argument("--activations-file", required=True, help="Path to the .pt file with activations.")
    parser.add_argument("--feature-index", type=int, required=True, help="The index of the feature to analyze.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top activating examples to find.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing activations.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Activations and Metadata ---
    print(f"\nLoading activations from {args.activations_file}...")
    data = torch.load(args.activations_file, map_location=device)
    activations_tensor = data["activations"]
    metadata = data["metadata"]
    source_map = metadata.get("source_map")

    if not source_map:
        print("Error: 'source_map' not found in activations file metadata. Please regenerate with a newer script version.")
        return

    num_samples, C, H, W = activations_tensor.shape
    activations = activations_tensor.permute(0, 2, 3, 1).contiguous().view(-1, C)
    print(f"  - Reshaped activations for SAE: {activations.shape}")

    # --- Load SAE Model ---
    print(f"\nLoading SAE model from {args.sae_model_file}...")
    checkpoint = torch.load(args.sae_model_file, map_location=device)
    hyperparameters = checkpoint['hyperparameters']
    state_dict = checkpoint['model_state_dict']
    
    model = SparseAutoencoder(hyperparameters['input_features'], hyperparameters['dict_features'])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  - Model loaded successfully.")

    if args.feature_index >= model.encoder.out_features:
        print(f"Error: feature_index {args.feature_index} is out of bounds for model with {model.encoder.out_features} features.")
        return

    # --- Find Top Activating Examples ---
    print(f"\nFinding top {args.top_k} activating examples for feature {args.feature_index}...")

    top_k_activations = [] # List of (activation_value, index)

    with torch.no_grad():
        for i in tqdm(range(0, len(activations), args.batch_size), desc="Scanning activations"):
            batch = activations[i:i+args.batch_size]
            _, encoded_batch = model(batch)
            
            feature_activations = encoded_batch[:, args.feature_index]
            
            for j, activation_value in enumerate(feature_activations):
                current_value = activation_value.item()
                if len(top_k_activations) < args.top_k:
                    top_k_activations.append((current_value, i + j))
                    top_k_activations.sort(key=lambda x: x[0], reverse=True)
                elif current_value > top_k_activations[-1][0]:
                    top_k_activations.pop()
                    top_k_activations.append((current_value, i + j))
                    top_k_activations.sort(key=lambda x: x[0], reverse=True)

    print("\n--- Top Activating Examples ---")
    for value, index in top_k_activations:
        sgf_file, move_num, (row, col) = find_source_from_index(index, source_map, H, W)
        # Note: move_num is 0-indexed. 0 is the empty board, 1 is the first move.
        print(f"Activation: {value:.4f} | SGF: {sgf_file} | Move: {move_num} | Location: ({row}, {col})")


if __name__ == "__main__":
    main()
