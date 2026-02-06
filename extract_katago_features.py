import sys
import argparse
import os

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
        """Loads a PyTorch-native KataGo model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "This script requires a PyTorch-native (.pth) model file. "
                "Please convert the .bin.gz model outside this environment and place it here."
            )

        print(f"Loading PyTorch model from '{model_path}'")
        model, _, _ = load_model(
            model_path, use_swa=False, device="cpu", pos_len=pos_len
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
        default="kata1-b28c512nbt-s7944987392-d4526094999.pth",
        help="Path to a KataGo PyTorch model file (.pth).",
    )
    args = parser.parse_args()

    extract_features(args.model_path)
