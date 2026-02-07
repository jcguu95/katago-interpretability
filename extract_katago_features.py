import sys
import argparse
import os
import urllib.request
import zipfile

# Add submodules to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'katago', 'python'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sgfmill'))

from sgfmill import sgf
from katago.game import gamestate
from katago.train.load_model import load_model
import torch
from katago.game.board import Board
# from katago.game import rules


def get_state_from_sgf(sgf_file, move_number):
    """Loads a game state from an SGF file at a specific move number."""
    if not os.path.exists(sgf_file):
        raise FileNotFoundError(f"SGF file not found: {sgf_file}")

    with open(sgf_file, "rb") as f:
        sgf_content = f.read()

    game = sgf.Sgf_game.from_bytes(sgf_content)
    board_size = game.get_size()
    state = gamestate.GameState(board_size=board_size, rules=gamestate.GameState.RULES_TT)

    node = game.get_root()

    # Handle setup stones (handicap)
    if node.has_property('AB'):
        for point in node.get('AB'):
            row, col = point
            state.play(Board.BLACK, state.board.loc(col, row))
    if node.has_property('AW'):
        for point in node.get('AW'):
            row, col = point
            state.play(Board.WHITE, state.board.loc(col, row))

    # Replay moves to reach the desired move number
    for i in range(move_number):
        try:
            node = node[0]  # Get next node in main variation
        except IndexError:
            print(f"Warning: SGF file has fewer than {move_number} moves. Stopping at move {i}.")
            break

        if node.has_property('B'):
            color, point = "B", node.get('B')
        elif node.has_property('W'):
            color, point = "W", node.get('W')
        else:
            # Not a move node (e.g., commentary), so we skip it
            continue

        player = Board.BLACK if color == 'B' else Board.WHITE
        if point is None:  # Pass
            state.play_pass(player)
        else:
            row, col = point
            state.play(player, state.board.loc(col, row))

    return state


def initialize_game_state():
    """Initializes the game state with two stones."""
    state = gamestate.GameState(board_size=19, rules=gamestate.GameState.RULES_TT)
    state.play(Board.BLACK, state.board.loc(2, 2))  # Black
    state.play(Board.WHITE, state.board.loc(3, 3))  # White
    return state


class KataGoFeatureExtractor:
    """A class to handle KataGo model loading and feature extraction."""

    def __init__(self, model_path, pos_len):
        """Initializes the feature extractor and loads the model."""
        self.model = self._load_katago_model(model_path, pos_len)

    def _load_katago_model(self, model_path, pos_len):
        """Loads a PyTorch-native KataGo model (.ckpt), downloading and unzipping if necessary."""
        local_path = os.path.basename(model_path)

        model_filename = local_path
        if local_path.endswith(".zip"):
            # The zip is expected to extract to a directory named after the zip, containing 'model.ckpt'
            unzipped_dir = os.path.splitext(local_path)[0]
            model_filename = os.path.join(unzipped_dir, "model.ckpt")

        # Download if the target model file doesn't exist and a URL is provided
        if not os.path.exists(model_filename) and (
            model_path.startswith("http://") or model_path.startswith("https://")
        ):
            print(f"Model not found locally. Downloading from {model_path}...")

            # Add User-Agent header to avoid 403 Forbidden error
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(model_path, headers=headers)
            with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
                out_file.write(response.read())

            print(f"Downloaded to {local_path}.")

        # Unzip if we have a zip file and the target model doesn't exist yet
        if local_path.endswith(".zip") and not os.path.exists(model_filename):
            print(f"Extracting {local_path}...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Extraction complete.")

        if not os.path.exists(model_filename):
            raise FileNotFoundError(
                f"Model file not found at {model_filename}. "
                "This script requires a PyTorch-native model file (.ckpt)."
            )

        print(f"Loading PyTorch model from '{model_filename}'")
        model, _, _ = load_model(
            model_filename, use_swa=False, device="cpu", pos_len=pos_len
        )
        model.eval()  # Set the model to evaluation mode
        return model

    def extract_trunkfinal_output(self, state):
        """Extracts the 'trunkfinal' layer from KataGo's neural network."""
        extra_output_names = ["trunkfinal"]
        outputs = state.get_model_outputs(
            self.model, extra_output_names=extra_output_names
        )
        trunkfinal_output = outputs["trunkfinal"]
        return trunkfinal_output


def print_trunkfinal_output(trunkfinal_output):
    """Prints the trunkfinal output."""
    print("Trunkfinal output shape:", trunkfinal_output.shape)
    print("Trunkfinal output:", trunkfinal_output)


def extract_features(args):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """
    if args.sgf_file:
        print(f"--- Loading game state from {args.sgf_file} at move {args.move_number} ---")
        state = get_state_from_sgf(args.sgf_file, args.move_number)
    else:
        print("--- Using initial demo game state ---")
        state = initialize_game_state()

    pos_len = state.board_size if isinstance(state.board_size, int) else state.board_size[0]
    extractor = KataGoFeatureExtractor(args.model_path, pos_len)

    print("\n--- Extracting features for the specified game state ---")
    trunkfinal_output = extractor.extract_trunkfinal_output(state)
    print_trunkfinal_output(trunkfinal_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state."
    )
    parser.add_argument(
        "--model-path",
        default="https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s12374138624-d5703190512.zip",
        help="Path or URL to a KataGo PyTorch model file (.ckpt or .zip).",
    )
    parser.add_argument(
        "--sgf-file",
        help="Path to an SGF file to load the game state from.",
    )
    parser.add_argument(
        "--move-number",
        type=int,
        default=0,
        help="Move number to extract features from (default: 0 for the root position).",
    )
    args = parser.parse_args()

    extract_features(args)
