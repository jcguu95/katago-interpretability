import sys
import argparse
sys.path.append('./katago/python')

from katago.game import gamestate
from katago.train import model_pytorch
import torch
# from katago.game import rules

def extract_features(model_path):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """

    # Initialize game state with two stones
    # Example: Black at (2, 2), White at (3, 3)
    # rule = rules.Rules()
    state = gamestate.GameState(board_size=19, rules=gamestate.RULES_TT)
    state.play_move(2, 2, color=1)  # Black
    state.play_move(3, 3, color=-1) # White

    config = model_pytorch.ModelConfig()  # Use default config or load from a file
    model = model_pytorch.Model(config)
    model.load_weights(model_path)
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
