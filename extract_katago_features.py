import sys
import argparse
import os
import urllib.request
import zipfile
import requests

from sgfmill import sgf
from katago.game import gamestate
from katago.game import features
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs
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
            state.play(player, Board.PASS_LOC)
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
        is_url = model_path.startswith("http://") or model_path.startswith("https://")

        # Determine the target local path for the model/zip file
        if is_url:
            models_dir = os.environ.get('KATAGO_MODELS_DIR', '.')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
            local_path = os.path.join(models_dir, os.path.basename(model_path))
        else:
            # If a local path is provided, use it as is.
            local_path = model_path

        # If a URL is provided, download the model.
        if is_url and not os.path.exists(local_path):
            print(f"Model not found locally. Downloading from {model_path}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://katagotraining.org/'
            }
            try:
                response = requests.get(model_path, headers=headers, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes
                with open(local_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download model: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Downloaded to {local_path}.")

        model_filename = local_path
        if local_path.endswith(".zip"):
            extract_dir = os.path.splitext(local_path)[0]
            # Unzip if the extract directory doesn't exist
            if not os.path.exists(extract_dir):
                print(f"Extracting {local_path}...")
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print("Extraction complete.")

            # Search for the model file in the extracted directory
            found_model_path = None
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        found_model_path = os.path.join(root, file)
                        break
                if found_model_path:
                    break
            
            if not found_model_path:
                raise FileNotFoundError(f"Could not find a model file (.ckpt) in the extracted contents of {local_path}")
            model_filename = found_model_path

        if not os.path.exists(model_filename):
            raise FileNotFoundError(
                f"Model file not found at {model_filename}. "
                "This script requires a KataGo model file (.ckpt)."
            )

        print(f"Loading PyTorch model from '{model_filename}'")
        model, config, _ = load_model(
            model_filename, use_swa=False, device="cpu", pos_len=pos_len
        )
        model.eval()  # Set the model to evaluation mode
        return model, config

    def extract_trunkfinal_output(self, state):
        """Extracts the 'trunkfinal' layer from KataGo's neural network for a single state."""
        # Process a single state by wrapping it in a batch of size 1
        batch_output = self.extract_trunkfinal_output_batch([state])
        return batch_output[0]

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
        extra_outputs = ExtraOutputs(extra_output_names)

        # Get device from model
        device = next(self.model.parameters()).device

        # Convert to tensors
        binary_input_data_tensor = torch.from_numpy(binary_input_data).to(device)
        global_input_data_tensor = torch.from_numpy(global_input_data).to(device)

        with torch.no_grad():
            self.model.forward(
                binary_input_data_tensor,
                global_input_data_tensor,
                extra_outputs=extra_outputs
            )

        trunkfinal_output_batch = extra_outputs.returned["trunkfinal"].cpu().numpy()
        return trunkfinal_output_batch


def print_trunkfinal_output(trunkfinal_output):
    """Prints the trunkfinal output."""
    print("Trunkfinal output shape:", trunkfinal_output.shape)
    print("Trunkfinal output:", trunkfinal_output)


def extract_features(args):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """
    if args.sgf_node:
        nodes_to_process = args.sgf_node
        
        states = []
        valid_nodes = []
        for sgf_filepath, path in nodes_to_process:
            try:
                states.append(get_state_from_sgf(sgf_filepath, path))
                valid_nodes.append((sgf_filepath, path))
            except (FileNotFoundError, ValueError) as e:
                print(f"Error processing {sgf_filepath} with path '{path}': {e}", file=sys.stderr)
                continue

        if not states:
            print("No valid game states to process.", file=sys.stderr)
            if nodes_to_process:
                sys.exit(1)
            return

        # Determine pos_len from the first valid state
        pos_len = states[0].board_size
        extractor = KataGoFeatureExtractor(args.model_path, pos_len)

        print(f"--- Extracting features for {len(states)} positions in a single batch ---")
        trunkfinal_outputs = extractor.extract_trunkfinal_output_batch(states)

        for i, (sgf_filepath, path) in enumerate(valid_nodes):
            print(f"\n--- Features for {sgf_filepath} at path '{path or 'root'}' ---")
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
        default="https://media.katagotraining.org/uploaded/networks/zips/misc/kata1-b6c96-s175439552-d46399393-checkpoint.zip",
        help="Path or URL to a KataGo model file (.ckpt) or a .zip archive containing one.",
    )
    parser.add_argument(
        '--sgf-node',
        nargs=2,
        action='append',
        metavar=('SGF_FILE', 'VARIATION_PATH'),
        help='Pair of SGF file and variation path to extract features from. Can be specified multiple times for batch processing.'
    )
    args = parser.parse_args()

    extract_features(args)
