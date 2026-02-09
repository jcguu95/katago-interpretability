import sys
import os
import torch
import numpy as np

# Add KataGo submodule to path
sys.path.insert(0, os.path.abspath('katago/python'))

from katago.game import gamestate
from katago.game.board import Board
from katago.train.load_model import load_model
from katago.game import features
from katago.train.model_pytorch import ExtraOutputs

def initialize_game_state():
    """Initializes a simple game state for demonstration."""
    state = gamestate.GameState(board_size=19, rules=gamestate.GameState.RULES_TT)
    state.play(Board.BLACK, state.board.loc(3, 3))  # D4
    state.play(Board.WHITE, state.board.loc(15, 15)) # Q16
    return state

def main():
    """
    This script verifies that the modification to KataGo's source code
    successfully exposes the 'policy_penultimate' activation layer.

    It loads a KataGo model, creates a simple board state, and runs a forward
    pass while requesting both 'trunkfinal' and 'policy_penultimate' as extra
    outputs. It then prints the shapes of these tensors.

    A successful run will print the shapes for both tensors, confirming that
    the 'policy_penultimate' tensor is being correctly captured.
    """
    # This path assumes the model has been downloaded and extracted by 'make data-pipeline'
    model_dir = "kata1-b28c512nbt-s12404017920-d5711392113"
    model_path = os.path.join(model_dir, "model.ckpt")
    board_size = 19
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Verification Script for 'policy_penultimate' ---")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at '{model_path}'.")
        print("Please run 'make data-pipeline' first to download and extract the model.")
        sys.exit(1)

    try:
        model, config, _ = load_model(model_path, use_swa=False, device=device, pos_len=board_size)
        model.eval()
        print(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)


    features_obj = features.Features(config, board_size)
    state = initialize_game_state()

    # Prepare model inputs from game state
    binary_input_data = np.zeros(shape=[1] + features_obj.bin_input_shape, dtype=np.float32)
    global_input_data = np.zeros(shape=[1] + features_obj.global_input_shape, dtype=np.float32)

    binary_input_data_to_fill = np.transpose(binary_input_data, axes=(0, 2, 3, 1))
    binary_input_data_to_fill = binary_input_data_to_fill.reshape([1, board_size * board_size, -1])

    pla = state.board.pla
    opp = Board.get_opp(pla)
    features_obj.fill_row_features(
        state.board, pla, opp, state.boards, state.moves, len(state.moves), state.rules,
        binary_input_data_to_fill, global_input_data, 0
    )

    binary_input_data_tensor = torch.from_numpy(binary_input_data).to(device)
    global_input_data_tensor = torch.from_numpy(global_input_data).to(device)

    # Request both 'trunkfinal' and the new 'policy_penultimate'
    extra_output_names = ["trunkfinal", "policy_penultimate"]
    extra_outputs = ExtraOutputs(extra_output_names)

    print(f"\nRequesting extra outputs: {extra_output_names}")

    # Run forward pass
    with torch.no_grad():
        model.forward(
            binary_input_data_tensor,
            global_input_data_tensor,
            extra_outputs=extra_outputs
        )

    print("\n--- Results ---")
    success = True
    if "trunkfinal" in extra_outputs.returned:
        trunkfinal_shape = extra_outputs.returned["trunkfinal"].shape
        print(f"Successfully captured 'trunkfinal' with shape: {trunkfinal_shape}")
    else:
        print("ERROR: Failed to capture 'trunkfinal'.")
        success = False


    if "policy_penultimate" in extra_outputs.returned:
        penultimate_shape = extra_outputs.returned["policy_penultimate"].shape
        print(f"Successfully captured 'policy_penultimate' with shape: {penultimate_shape}")
    else:
        print("ERROR: Failed to capture 'policy_penultimate'.")
        print("Please ensure the change in 'katago/python/katago/train/model_pytorch.py' has been applied correctly.")
        success = False

    if success:
        print("\nSUCCESS: The KataGo modification is working as expected.")
    else:
        print("\nFAILURE: Verification failed.")


if __name__ == "__main__":
    main()
