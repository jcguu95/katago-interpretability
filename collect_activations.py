import argparse
import os
import sys
import subprocess
import json
import torch
import numpy as np

from sgfmill import sgf
from katago.game.board import Board
from katago.game.gamestate import GameState
from extract_katago_features import KataGoFeatureExtractor


def get_git_commit_hash(path='.'):
    """Gets the git commit hash for a given path."""
    try:
        # Use DEVNULL to hide git errors, e.g. when not in a git repo
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, stderr=subprocess.DEVNULL).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def process_sgf_file(sgf_path):
    """Yields a GameState for each move in an SGF file's main variation."""
    with open(sgf_path, "rb") as f:
        try:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError:
            print(f"Warning: Could not parse {sgf_path}", file=sys.stderr)
            return

    board_size = sgf_game.get_size()
    state = GameState(board_size=board_size, rules=GameState.RULES_TT)

    node = sgf_game.get_root()
    # Handle setup stones
    if node.has_property('AB'):
        for point in node.get('AB'):
            row, col = point
            state.play(Board.BLACK, state.board.loc(col, row))
    if node.has_property('AW'):
        for point in node.get('AW'):
            row, col = point
            state.play(Board.WHITE, state.board.loc(col, row))

    yield state

    # Follow the main variation
    while node.has_children():
        node = node[0]
        if node.has_property('B'):
            color, point = "B", node.get('B')
        elif node.has_property('W'):
            color, point = "W", node.get('W')
        else:
            continue

        player = Board.BLACK if color == 'B' else Board.WHITE
        if point is None:  # Pass
            loc = Board.PASS_LOC
        else:
            row, col = point
            loc = state.board.loc(col, row)
        
        # Check legality before playing
        if state.board.would_be_legal(player, loc):
            state.play(player, loc)
            yield state
        else:
            print(f"Warning: Illegal move in {sgf_path}, stopping processing for this file.", file=sys.stderr)
            break


def main():
    parser = argparse.ArgumentParser(description="Collect KataGo feature activations from SGF files.")
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files.")
    parser.add_argument("--model-path", required=True, help="Path to the KataGo model file.")
    parser.add_argument("--output-file", default="activations/activations.pt", help="Path to save the collected activations.")
    args = parser.parse_args()

    if not os.path.isdir(args.sgf_dir):
        print(f"Error: SGF directory not found at {args.sgf_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # We assume all SGFs are for 19x19 boards.
    print("Initializing feature extractor...")
    extractor = KataGoFeatureExtractor(args.model_path, pos_len=19)
    print("Initialization complete.")

    all_activations = []

    sgf_files = sorted([os.path.join(args.sgf_dir, f) for f in os.listdir(args.sgf_dir) if f.endswith(".sgf")])

    print(f"\nFound {len(sgf_files)} SGF files to process.")

    total_states = 0
    for sgf_path in sgf_files:
        print(f"Processing {sgf_path}...")
        states_to_process = list(process_sgf_file(sgf_path))
        if not states_to_process:
            continue

        total_states += len(states_to_process)
        activations_batch = extractor.extract_trunkfinal_output_batch(states_to_process)
        all_activations.append(activations_batch)

    if not all_activations:
        print("No activations were collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Concatenate all activations into a single tensor
    final_tensor = np.concatenate(all_activations, axis=0)
    final_tensor_torch = torch.from_numpy(final_tensor)

    print(f"\nCollected {final_tensor_torch.shape[0]} activations from {total_states} game states.")

    # Prepare metadata for reproducibility
    metadata = {
        "repo_git_hash": get_git_commit_hash(),
        "katago_submodule_hash": get_git_commit_hash(path="katago"),
        "model_path": os.path.abspath(args.model_path),
        "sgf_dir": os.path.abspath(args.sgf_dir),
        "num_activations": final_tensor_torch.shape[0],
        "activations_shape": list(final_tensor_torch.shape),
    }

    # Save the tensor and metadata
    torch.save({
        "metadata": metadata,
        "activations": final_tensor_torch
    }, args.output_file)

    print(f"Saved activations and metadata to {args.output_file}")
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
