import sys
import argparse
sys.path.append('./katago/python')

from katago.game import gamestate
from katago.train import model_pytorch
import torch
from katago.game.board import Board
# from katago.game import rules

def extract_features(model_path):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """

    # Initialize game state with two stones
    # Example: Black at (2, 2), White at (3, 3)
    # rule = rules.Rules()
    state = gamestate.GameState(board_size=19, rules=gamestate.GameState.RULES_TT)
    state.play(Board.BLACK, state.board.loc(2, 2))  # Black
    state.play(Board.WHITE, state.board.loc(3, 3)) # White

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
    pos_len = state.board_size if isinstance(state.board_size, int) else state.board_size[0]
    model = model_pytorch.Model(config, pos_len=pos_len)
    model_pytorch.load_weights(model, model_path)
    model.eval()  # Set the model to evaluation mode

    # Extract the trunkfinal layer
    extra_output_names = ["trunkfinal"]
    outputs = state.get_model_outputs(model, extra_output_names=extra_output_names)

    # Get the trunkfinal output
    trunkfinal_output = outputs["trunkfinal"]

    # Print the output (or do something else with it)
    print("Trunkfinal output shape:", trunkfinal_output.shape)
    print("Trunkfinal output:", trunkfinal_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract KataGo features from a game state.")
    parser.add_argument("model_path", help="Path to the KataGo model file.")
    args = parser.parse_args()

    extract_features(args.model_path)
