import sys
import argparse
import os
import gzip
import shutil
import urllib.request

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
        """Loads the KataGo model."""
        local_model_path = model_path
        if not os.path.exists(local_model_path) and (
            local_model_path.startswith("http://")
            or local_model_path.startswith("https://")
        ):
            print(f"Model not found locally. Downloading from {local_model_path}...")
            gz_filename = os.path.basename(local_model_path)
            uncompressed_filename = (
                gz_filename[:-3] if gz_filename.endswith(".gz") else gz_filename
            )

            urllib.request.urlretrieve(local_model_path, gz_filename)
            print(f"Downloaded to {gz_filename}.")

            if gz_filename.endswith(".gz"):
                print(f"Decompressing to {uncompressed_filename}...")
                with gzip.open(gz_filename, "rb") as f_in:
                    with open(uncompressed_filename, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("Decompression complete.")

            local_model_path = uncompressed_filename

        model, swa_model, other_state_dict = load_model(
            local_model_path, use_swa=False, device="cpu", pos_len=pos_len
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
    trunkfinal_output = extractor.extract_trunkfinal_output(state)
    print_trunkfinal_output(trunkfinal_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state."
    )
    parser.add_argument(
        "--model_path",
        default="https://media.katagotraining.org/uploaded/models/kata1/kata1-b15c192-s1673373440-d46379203.bin.gz",
        help="Path or URL to the KataGo model file.",
    )
    args = parser.parse_args()

    extract_features(args.model_path)
