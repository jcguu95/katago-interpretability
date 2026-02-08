import argparse
import sys
import torch

from sgfmill import sgf
from katago.game.board import Board
from katago.game.gamestate import GameState
from feature_extraction.extract_katago_features import KataGoFeatureExtractor

def get_state_at_move(sgf_path, move_number):
    """
    Replays an SGF file to a specific move number and returns the GameState.
    Move 0 is the empty board.
    """
    with open(sgf_path, "rb") as f:
        try:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError:
            print(f"Error: Could not parse {sgf_path}", file=sys.stderr)
            return None

    board_size = sgf_game.get_size()
    state = GameState(board_size=board_size, rules=GameState.RULES_TT)
    
    current_move = 0

    # Handle setup stones
    node = sgf_game.get_root()
    if node.has_property('AB'):
        for point in node.get('AB'):
            row, col = point
            state.play(Board.BLACK, state.board.loc(col, row))
    if node.has_property('AW'):
        for point in node.get('AW'):
            row, col = point
            state.play(Board.WHITE, state.board.loc(col, row))
            
    if move_number == 0:
        return state

    # Follow the main variation
    while node:
        if not node[0]:
            break
        node = node[0]
            
        current_move += 1
        
        if node.has_property('B'):
            color, point = "B", node.get('B')
        elif node.has_property('W'):
            color, point = "W", node.get('W')
        else:
            continue

        player = Board.BLACK if color == 'B' else Board.WHITE
        if point is None:
            loc = Board.PASS_LOC
        else:
            row, col = point
            loc = state.board.loc(col, row)
        
        if state.board.would_be_legal(player, loc):
            state.play(player, loc)
            if current_move == move_number:
                return state
        else:
            print(f"Warning: Illegal move found at move {current_move} in {sgf_path}", file=sys.stderr)
            return None
            
    print(f"Error: Move number {move_number} not found in SGF. The game has only {current_move} moves.", file=sys.stderr)
    return None

def main():
    parser = argparse.ArgumentParser(description="Verify the activation extraction for a single game state.")
    parser.add_argument("--sgf-file", required=True, help="Path to the SGF file.")
    parser.add_argument("--move-number", type=int, required=True, help="The move number to analyze (0 for empty board).")
    parser.add_argument("--model-path", required=True, help="Path to the KataGo model file.")
    args = parser.parse_args()

    print("Initializing feature extractor...")
    extractor = KataGoFeatureExtractor(args.model_path, pos_len=19)
    print("Initialization complete.")
    
    print(f"\nLoading game state from {args.sgf_file} at move {args.move_number}...")
    game_state = get_state_at_move(args.sgf_file, args.move_number)
    
    if game_state is None:
        sys.exit(1)
        
    print("Game state loaded successfully.")
    print("Board:")
    game_state.board.show()

    print("\nExtracting activations...")
    activations_np = extractor.extract_trunkfinal_output(game_state)
    activations = torch.from_numpy(activations_np)
    print("Extraction complete.")

    print("\n--- Activation Tensor Summary ---")
    print(f"  - Shape: {list(activations.shape)}")
    print(f"  - DType: {activations.dtype}")
    print(f"  - Mean:  {activations.mean().item():.6f}")
    print(f"  - Std:   {activations.std().item():.6f}")
    print(f"  - Min:   {activations.min().item():.6f}")
    print(f"  - Max:   {activations.max().item():.6f}")

if __name__ == "__main__":
    main()
