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
from katago.game import features
from katago.train.load_model import load_model
import torch
import numpy as np
from katago.game.board import Board
# from katago.game import rules


def get_state_from_sgf(sgf_file, variation_path_str):
    """Loads a game state from an SGF file at a specific node, specified by a variation path."""
    if not os.path.exists(sgf_file):
        raise FileNotFoundError(f"SGF file not found: {sgf_file}")

    with open(sgf_file, "rb") as f:
        sgf_content = f.read()

    game = sgf.Sgf_game.from_bytes(sgf_content)
    board_size = game.get_size()
    state = gamestate.GameState(board_size=board_size, rules=gamestate.GameState.RULES_TT)

    node = game.get_root()

    # Handle setup stones (handicap) from root node
    if node.has_property('AB'):
        for point in node.get('AB'):
            row, col = point
            state.play(Board.BLACK, state.board.loc(col, row))
    if node.has_property('AW'):
        for point in node.get('AW'):
            row, col = point
            state.play(Board.WHITE, state.board.loc(col, row))

    # Parse variation path
    path_indices = []
    if variation_path_str:
        try:
            path_indices = [int(i) for i in variation_path_str.split(',')]
        except ValueError:
            raise ValueError("Invalid variation path. Must be comma-separated integers.")

    # Replay moves along the path to reach the desired node
    for branch_index in path_indices:
        try:
            node = node[branch_index]
        except IndexError:
            raise ValueError(f"Invalid variation path: branch index {branch_index} is out of range for a node with {len(node)} children.")

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
        self.model, config = self._load_katago_model(model_path, pos_len)
        # If config is not returned by load_model, try to get it from the model object
        if config is None and hasattr(self.model, 'config'):
            config = self.model.config
        self.features_obj = features.Features(config, pos_len)

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
        model, config, _ = load_model(
            model_filename, use_swa=False, device="cpu", pos_len=pos_len
        )
        model.eval()  # Set the model to evaluation mode
        return model, config

    def extract_trunkfinal_output(self, state):
        """Extracts the 'trunkfinal' layer from KataGo's neural network."""
        extra_output_names = ["trunkfinal"]
        outputs = state.get_model_outputs(
            self.model, extra_output_names=extra_output_names
        )
        trunkfinal_output = outputs["trunkfinal"]
        return trunkfinal_output

    def extract_trunkfinal_output_batch(self, states):
        """Extracts 'trunkfinal' layer for a batch of game states."""
        if not states:
            return np.array([])

        batch_size = len(states)
        pos_len = self.features_obj.pos_len
        # Get shapes from the features object
        binary_input_data = np.zeros(shape=[batch_size] + self.features_obj.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[batch_size] + self.features_obj.global_input_shape, dtype=np.float32)

        # The fill_row_features function expects a specific data layout (N, HW, C).
        # We need to reshape the array before passing it to the function.
        # This creates a view, so modifications will be reflected in the original array.
        binary_input_data_to_fill = np.transpose(binary_input_data, axes=(0, 2, 3, 1))
        binary_input_data_to_fill = binary_input_data_to_fill.reshape([batch_size, pos_len * pos_len, -1])

        for i, state in enumerate(states):
            pla = state.board.pla
            opp = Board.get_opp(pla)
            self.features_obj.fill_row_features(
                state.board, pla, opp, state.boards, state.moves, len(state.moves), state.rules,
                binary_input_data_to_fill, global_input_data, i
            )

        extra_output_names = ["trunkfinal"]

        # Get device from model
        device = next(self.model.parameters()).device

        # Convert to tensors
        binary_input_data_tensor = torch.from_numpy(binary_input_data).to(device)
        global_input_data_tensor = torch.from_numpy(global_input_data).to(device)

        with torch.no_grad():
            outputs = self.model.forward(
                binary_input_data_tensor,
                global_input_data_tensor,
                extra_output_names=extra_output_names
            )

        trunkfinal_output_batch = outputs["trunkfinal"].cpu().numpy()
        return trunkfinal_output_batch


def print_trunkfinal_output(trunkfinal_output):
    """Prints the trunkfinal output."""
    print("Trunkfinal output shape:", trunkfinal_output.shape)
    print("Trunkfinal output:", trunkfinal_output)


def extract_features(args):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """
    if args.sgf_file:
        # Batch processing for all specified variation paths
        print(f"--- Loading game states from {args.sgf_file} ---")
        states = [get_state_from_sgf(args.sgf_file, path) for path in args.variation_path]

        if not states:
            print("No valid game states to process.")
            return

        pos_len = states[0].board_size if isinstance(states[0].board_size, int) else states[0].board_size[0]
        extractor = KataGoFeatureExtractor(args.model_path, pos_len)

        print("\n--- Extracting features for the specified game states in a batch ---")
        trunkfinal_outputs = extractor.extract_trunkfinal_output_batch(states)

        for i, path in enumerate(args.variation_path):
            print(f"\n--- Features for path '{path or 'root'}' ---")
            print_trunkfinal_output(trunkfinal_outputs[i])

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
        "--variation-path",
        type=str,
        nargs='*',
        default=[""],
        help="One or more comma-separated paths of variation indices (e.g., '0,1,0' '0,0,1'). Default is the root position.",
    )
    args = parser.parse_args()

    extract_features(args)
