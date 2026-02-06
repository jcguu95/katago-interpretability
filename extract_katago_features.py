import sys
import argparse
import os
import urllib.request
import zipfile

# Add katago submodule to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'katago', 'python'))

from katago.game import gamestate
from katago.train.load_model import load_model
import torch
from katago.game.board import Board
# from katago.game import rules


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


def extract_features(model_path):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """
    state = initialize_game_state()
    pos_len = state.board_size if isinstance(state.board_size, int) else state.board_size[0]
    extractor = KataGoFeatureExtractor(model_path, pos_len)

    print("--- Features for Initial Game State ---")
    trunkfinal_output_1 = extractor.extract_trunkfinal_output(state)
    print_trunkfinal_output(trunkfinal_output_1)

    # Add another move and extract features again
    print("\n--- Features for Game State After One More Move ---")
    state.play(Board.BLACK, state.board.loc(4, 4))
    trunkfinal_output_2 = extractor.extract_trunkfinal_output(state)
    print_trunkfinal_output(trunkfinal_output_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state."
    )
    parser.add_argument(
        "--model_path",
        default="https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s12374138624-d5703190512.zip",
        help="Path or URL to a KataGo PyTorch model file (.ckpt or .zip).",
    )
    args = parser.parse_args()

    extract_features(args.model_path)
