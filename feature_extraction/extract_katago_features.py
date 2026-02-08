import sys
import argparse
import os
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

    def __init__(self, model_path, board_size):
        """Initializes the feature extractor and loads the model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, config = self._load_katago_model(model_path, board_size)
        # If config is not returned by load_model, try to get it from the model object
        if config is None and hasattr(self.model, 'config'):
            config = self.model.config
        self.features_obj = features.Features(config, board_size)

    def _load_katago_model(self, model_path, board_size):
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
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                model_file_in_zip = next((s for s in zip_ref.namelist() if s.endswith(".ckpt")), None)
                if not model_file_in_zip:
                    raise FileNotFoundError(f"Could not find a model file (.ckpt) in {local_path}")

                if not os.path.exists(model_file_in_zip):
                    print(f"Extracting {local_path}...")
                    # Extract to current directory, which is more robust
                    zip_ref.extractall(".")
                    print("Extraction complete.")
            model_filename = model_file_in_zip

        if not os.path.exists(model_filename):
            raise FileNotFoundError(
                f"Model file not found at {model_filename}. "
                "This script requires a KataGo model file (.ckpt)."
            )

        print(f"Loading PyTorch model from '{model_filename}'")
        model, config, _ = load_model(
            model_filename, use_swa=False, device=self.device, pos_len=board_size
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
        board_size = self.features_obj.board_size
        
        # 1. Create empty numpy arrays to hold the binary and global features for the batch.
        # These will be filled with data from each game state.
        binary_input_data = np.zeros(shape=[batch_size] + self.features_obj.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[batch_size] + self.features_obj.global_input_shape, dtype=np.float32)

        # The fill_row_features function expects a specific data layout (N, HW, C).
        # We need to reshape the array before passing it to the function.
        # This creates a view, so modifications will be reflected in the original array.
        binary_input_data_to_fill = np.transpose(binary_input_data, axes=(0, 2, 3, 1))
        binary_input_data_to_fill = binary_input_data_to_fill.reshape([batch_size, board_size * board_size, -1])

        # 2. Loop through each game state and use KataGo's `fill_row_features` to populate the numpy arrays.
        # This is where the game state is converted into the neural network's input format.
        for i, state in enumerate(states):
            # `pla` is the player to move, `opp` is the opponent.
            pla = state.board.pla
            opp = Board.get_opp(pla)
            self.features_obj.fill_row_features(
                state.board, pla, opp, state.boards, state.moves, len(state.moves), state.rules,
                binary_input_data_to_fill, global_input_data, i
            )

        # 3. Set up a hook to capture the 'trunkfinal' layer's output.
        # This doesn't use the `state` directly, but it tells the model what to give us back
        # when we run the forward pass with the state-derived feature tensors.
        extra_output_names = ["trunkfinal"]
        extra_outputs = ExtraOutputs(extra_output_names)

        # 4. Convert the numpy arrays to PyTorch tensors and move them to the correct device (CPU/GPU).
        binary_input_data_tensor = torch.from_numpy(binary_input_data).to(self.device)
        global_input_data_tensor = torch.from_numpy(global_input_data).to(self.device)

        # 5. Run the forward pass. The `extra_outputs` object will capture the intermediate 'trunkfinal' tensor.
        with torch.no_grad():
            self.model.forward(
                binary_input_data_tensor,
                global_input_data_tensor,
                extra_outputs=extra_outputs
            )

        # 6. Retrieve the captured tensor from the `extra_outputs` object.
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

        # Determine board_size from the first valid state
        board_size = states[0].board_size
        extractor = KataGoFeatureExtractor(args.model_path, board_size)

        print(f"--- Extracting features for {len(states)} positions in a single batch ---")
        trunkfinal_outputs = extractor.extract_trunkfinal_output_batch(states)

        for i, (sgf_filepath, path) in enumerate(valid_nodes):
            print(f"\n--- Features for {sgf_filepath} at path '{path or 'root'}' ---")
            print_trunkfinal_output(trunkfinal_outputs[i])

    else:
        print("--- Using initial demo game state ---")
        state = initialize_game_state()
        board_size = state.board_size if isinstance(state.board_size, int) else state.board_size[0]
        extractor = KataGoFeatureExtractor(args.model_path, board_size)

        print("\n--- Extracting features for the specified game state ---")
        trunkfinal_output = extractor.extract_trunkfinal_output(state)
        print_trunkfinal_output(trunkfinal_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state."
    )
    parser.add_argument(
        "--model-path",
        default="https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s12404017920-d5711392113.zip",
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
