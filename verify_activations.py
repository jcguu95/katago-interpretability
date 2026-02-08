import argparse
import sys
import torch

from sgf_utils import get_state_at_move
from feature_extraction.extract_katago_features import KataGoFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Verify the activation extraction for a single game state.")
    parser.add_argument("--sgf-file", required=True, help="Path to the SGF file.")
    parser.add_argument("--move-number", type=int, required=True, help="The move number to analyze (0 for empty board).")
    parser.add_argument("--model-path", required=True, help="Path to the KataGo model file.")
    args = parser.parse_args()

    print("Initializing feature extractor...")
    extractor = KataGoFeatureExtractor(args.model_path, pos_len=19)
    print("Initialization complete.")
    
    print(f"\nLoading game state from {args.sgf_file} at move {args.move_number}...")
    game_state = get_state_at_move(args.sgf_file, args.move_number)
    
    if game_state is None:
        sys.exit(1)
        
    print("Game state loaded successfully.")
    print("Board:")
    game_state.board.show()

    print("\nExtracting activations...")
    activations_np = extractor.extract_trunkfinal_output(game_state)
    activations = torch.from_numpy(activations_np)
    print("Extraction complete.")

    print("\n--- Activation Tensor Summary ---")
    print(f"  - Shape: {list(activations.shape)}")
    print(f"  - DType: {activations.dtype}")
    print(f"  - Mean:  {activations.mean().item():.6f}")
    print(f"  - Std:   {activations.std().item():.6f}")
    print(f"  - Min:   {activations.min().item():.6f}")
    print(f"  - Max:   {activations.max().item():.6f}")


if __name__ == "__main__":
    main()
