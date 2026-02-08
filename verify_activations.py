import argparse
import sys
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from sgf_utils import get_state_at_move
from feature_extraction.extract_katago_features import KataGoFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Verify the activation extraction for a single game state.")
    parser.add_argument("--sgf-file", required=True, help="Path to the SGF file.")
    parser.add_argument("--move-number", type=int, required=True, help="The move number to analyze (0 for empty board).")
    parser.add_argument("--model-path", required=True, help="Path to the KataGo model file.")
    parser.add_argument("--visualize-channel", type=int, help="If provided, plots a heatmap of the specified activation channel.")
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

    if args.visualize_channel is not None:
        if not MATPLOTLIB_AVAILABLE:
            print("\nError: --visualize-channel requires matplotlib.", file=sys.stderr)
            print("Please install it by running: pip install matplotlib", file=sys.stderr)
            sys.exit(1)
        
        if not (0 <= args.visualize_channel < activations.shape[1]):
            print(f"\nError: Channel index must be between 0 and {activations.shape[1]-1}.", file=sys.stderr)
            sys.exit(1)
            
        print(f"\nVisualizing channel {args.visualize_channel}...")
        
        channel_data = activations[0, args.visualize_channel, :, :].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # Use a diverging colormap centered at 0
        norm = mcolors.TwoSlopeNorm(vcenter=0)
        im = ax.imshow(channel_data, cmap='coolwarm', norm=norm)
        
        ax.set_title(f'Activation Heatmap for Channel {args.visualize_channel}\nSGF: {os.path.basename(args.sgf_file)} | Move: {args.move_number}')
        fig.colorbar(im, ax=ax, label="Activation Value")
        
        # Overlay Go board grid
        ax.set_xticks(np.arange(-.5, 18.5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 18.5, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Show stones
        board = game_state.board
        for r in range(19):
            for c in range(19):
                loc = board.loc(c, r)
                player = board.getPlayer(loc)
                if player == board.BLACK:
                    circle = plt.Circle((c, r), 0.45, color='black', zorder=3)
                    ax.add_patch(circle)
                elif player == board.WHITE:
                    circle = plt.Circle((c, r), 0.45, color='white', zorder=3, ec='black', linewidth=0.5)
                    ax.add_patch(circle)
        
        # Ensure aspect ratio is equal
        ax.set_aspect('equal', 'box')
        plt.show()

if __name__ == "__main__":
    main()
