import sys
import argparse
import os
import gzip
import shutil

# Add katago submodule to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'katago', 'python'))

from katago.game import gamestate
from katago.train.load_model import load_model
from katago.cpp_model import Model as CppModel
from katago.train.model_pytorch import Model as PyTorchModel
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
        """Loads the KataGo model, converting it to PyTorch format if necessary."""
        # Define the path for the PyTorch model
        base_name = os.path.basename(model_path)
        if base_name.endswith(".bin.gz"):
            pytorch_model_path = base_name[:-7] + ".pth"
        else:
            pytorch_model_path = os.path.splitext(base_name)[0] + ".pth"

        # Convert the model if the PyTorch version doesn't exist
        if not os.path.exists(pytorch_model_path):
            print(f"PyTorch model not found at '{pytorch_model_path}'.")
            print(f"Converting '{model_path}' to PyTorch format...")

            local_model_path = model_path
            if model_path.endswith(".gz"):
                uncompressed_filename = model_path[:-3]
                if not os.path.exists(uncompressed_filename):
                    print(f"Decompressing {model_path} to {uncompressed_filename}...")
                    with gzip.open(model_path, "rb") as f_in:
                        with open(uncompressed_filename, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print("Decompression complete.")
                local_model_path = uncompressed_filename

            print(f"Loading C++ model from '{local_model_path}'")
            cpp_model = CppModel(local_model_path)
            config = cpp_model.get_model_config()

            print("Creating PyTorch model from C++ model weights")
            pytorch_model = PyTorchModel(config, cpp_model.pos_len)
            pytorch_model.initialize()
            pytorch_model.load_state_dict(cpp_model.get_sd())

            print(f"Saving PyTorch model to '{pytorch_model_path}'")
            data_to_save = {
                "config": config,
                "model": pytorch_model.state_dict(),
                "train_state": {},
            }
            torch.save(data_to_save, pytorch_model_path)
            print("Conversion successful.")

        print(f"Loading PyTorch model from '{pytorch_model_path}'")
        model, _, _ = load_model(
            pytorch_model_path, use_swa=False, device="cpu", pos_len=pos_len
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
    print(f"Using Python: {sys.executable}")
    # The script will fail with an ImportError if torch is not available.
    # This print statement will only be reached if torch was imported successfully.
    print("torch is available.")
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state."
    )
    parser.add_argument(
        "--model_path",
        default="kata1-b28c512nbt-s7944987392-d4526094999.bin.gz",
        help="Path to the KataGo model file.",
    )
    args = parser.parse_args()

    extract_features(args.model_path)
