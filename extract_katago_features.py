import sys
import argparse
import os
import json
import subprocess
import time
from threading import Thread
from typing import Tuple, List, Optional, Union, Literal, Any, Dict

# Add katago submodule to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'katago', 'python'))

from katago.game import gamestate
import torch
from katago.game.board import Board
# from katago.game import rules

Color = Union[Literal["b"],Literal["w"]]
Move = Union[None,Literal["pass"],Tuple[int,int]]

def sgfmill_to_str(move: Move) -> str:
    if move is None:
        return "pass"
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)


def initialize_game_state():
    """Initializes the game state with two stones."""
    state = gamestate.GameState(board_size=19, rules=gamestate.GameState.RULES_TT)
    state.play(Board.BLACK, state.board.loc(2, 2))  # Black
    state.play(Board.WHITE, state.board.loc(3, 3))  # White
    return state


class KataGo:
    """A class to manage the KataGo analysis engine subprocess."""

    def __init__(self, katago_path: str, config_path: str, model_path: str, additional_args: List[str] = []):
        self.query_counter = 0
        katago_command = [katago_path, "analysis", "-config", config_path, "-model", model_path, *additional_args]
        print(f"Running KataGo: {' '.join(katago_command)}")
        katago = subprocess.Popen(
            katago_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago
        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                time.sleep(0)
                if data:
                    print("KataGo stderr: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo stderr: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

    def close(self):
        if self.katago.stdin and not self.katago.stdin.closed:
            self.katago.stdin.close()
        try:
            self.katago.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.katago.kill()
            self.katago.wait()
        self.stderrthread.join(timeout=5)

    def query_raw(self, query: Dict[str,Any]):
        if not self.katago.stdin or self.katago.stdin.closed:
             raise IOError("KataGo process stdin is closed")
        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        line = ""
        while line == "" and self.katago.poll() is None:
            if not self.katago.stdout or self.katago.stdout.closed:
                raise IOError("KataGo process stdout is closed")
            line = self.katago.stdout.readline()
            line = line.decode().strip()

        if line == "" and self.katago.poll() is not None:
             raise Exception("Unexpected katago exit")

        response = json.loads(line)
        return response


class KataGoFeatureExtractor:
    """A class to handle KataGo feature extraction via the analysis engine."""

    def __init__(self, katago_path, config_path, model_path):
        """Initializes the feature extractor and starts the KataGo engine."""
        self.katago = KataGo(katago_path, config_path, model_path)

    def close(self):
        """Shuts down the KataGo engine."""
        self.katago.close()

    def extract_trunkfinal_output(self, state):
        """Extracts the 'trunkfinal' layer from KataGo's neural network."""
        moves = []
        for player, loc in state.moves:
            color = "b" if player == Board.BLACK else "w"
            move = (state.board.loc_y(loc), state.board.loc_x(loc)) if loc is not None else "pass"
            moves.append((color, move))

        query = {
            "id": "extract_features_query",
            "moves": [(color, sgfmill_to_str(move)) for color, move in moves],
            "initialStones": [],
            "rules": "Chinese",
            "komi": state.komi,
            "boardXSize": state.board.x_size,
            "boardYSize": state.board.y_size,
            "includePolicy": False,
            "extraOutputNames": ["trunkfinal"],
        }
        response = self.katago.query_raw(query)

        if "extraOutputs" not in response or "trunkfinal" not in response["extraOutputs"]:
            raise ValueError(f"Failed to get trunkfinal output. Response: {response}")

        trunkfinal_output_list = response["extraOutputs"]["trunkfinal"]
        # The output is likely (C, H, W). We add a batch dimension to match the old output shape.
        trunkfinal_output = torch.tensor(trunkfinal_output_list, dtype=torch.float32).unsqueeze(0)
        return trunkfinal_output


def print_trunkfinal_output(trunkfinal_output):
    """Prints the trunkfinal output."""
    print("Trunkfinal output shape:", trunkfinal_output.shape)
    print("Trunkfinal output:", trunkfinal_output)


def extract_features(katago_path, config_path, model_path):
    """
    Extracts the 'trunkfinal' layer from KataGo's neural network for a given game state.
    """
    state = initialize_game_state()
    extractor = KataGoFeatureExtractor(katago_path, config_path, model_path)
    try:
        trunkfinal_output = extractor.extract_trunkfinal_output(state)
        print_trunkfinal_output(trunkfinal_output)
    finally:
        extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract KataGo features from a game state using the analysis engine."
    )
    parser.add_argument(
        "-katago-path",
        help="Path to katago executable",
        required=True,
    )
    parser.add_argument(
        "-config-path",
        help="Path to KataGo analysis config (e.g. katago/cpp/configs/analysis_example.cfg)",
        required=True,
    )
    parser.add_argument(
        "-model-path",
        help="Path to neural network .bin.gz file",
        default="kata1-b28c512nbt-s7944987392-d4526094999.bin.gz",
    )
    args = parser.parse_args()

    extract_features(args.katago_path, args.config_path, args.model_path)
