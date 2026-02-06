import sys
import argparse
import os

# Add katago submodule to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'katago', 'python'))

from katago.game import gamestate
from katago.train.model_pytorch import Model
from katago.train.load_model import load_weights
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
        config = {
            "norm_kind": "fixup",
            "block_kind": [["rconv1", "regular"]],
            "trunk_num_channels": 256,
            "mid_num_channels": 128,
            "gpool_num_channels": 32,
            "p1_num_channels": 32,
            "g1_num_channels": 16,
            "v1_num_channels": 32,
            "v2_size": 32,
            "sbv2_num_channels": 32,
            "num_scorebeliefs": 61,
            "initial_conv_1x1": True,
            "has_intermediate_head": False,
            "activation": "relu",
            "bnorm_epsilon": 1e-5,
            "bnorm_running_avg_momentum": 0.1,
            "version": 15,
            "use_attention_pool": False,
            "num_attention_pool_heads": 1,
            "use_repvgg_init": False,
            "use_repvgg_linear": False,
            "metadata_encoder": None,
            "trunk_normless": False,
        }
        model = Model(config, pos_len=pos_len)
        load_weights(model, model_path)
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
    parser.add_argument("model_path", help="Path to the KataGo model file.")
    args = parser.parse_args()

    extract_features(args.model_path)
