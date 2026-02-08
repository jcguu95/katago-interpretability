import os
import random
import argparse
from katago.game.board import Board
from katago.game.gamestate import GameState


def generate_plausible_game(board_size, num_moves):
    """Generates a list of moves for a plausible-looking game."""
    state = GameState(board_size=board_size, rules=GameState.RULES_TT)
    moves = []

    # Common opening points (hoshis and komokus)
    opening_points = [
        (3,3), (3,15), (15,3), (15,15), # hoshi
        (2,3), (3,2), (16,3), (3,16), # komoku
        (2,15), (15,2), (16,15), (15,16) # komoku
    ]
    random.shuffle(opening_points)

    # Play first few moves on opening points
    opening_moves = min(len(opening_points), 4) # play up to 4 opening moves
    for _ in range(opening_moves):
        if not opening_points:
            break
        # Use pop to not repeat points
        x, y = opening_points.pop(0)
        loc = state.board.loc(x, y)
        if state.board.would_be_legal(state.board.pla, loc):
            moves.append((state.board.pla, loc))
            state.play(state.board.pla, loc)
        else: # Should not happen with this list on an empty board, but good practice
            break

    # Continue with random moves for the rest of the game
    for _ in range(num_moves - len(moves)):
        possible_moves = []
        for y in range(state.board.y_size):
            for x in range(state.board.x_size):
                loc = state.board.loc(x, y)
                if state.board.would_be_legal(state.board.pla, loc):
                    possible_moves.append(loc)
        
        if not possible_moves:
            possible_moves.append(Board.PASS_LOC)
            
        move_loc = random.choice(possible_moves)
        moves.append((state.board.pla, move_loc))
        state.play(state.board.pla, move_loc)
    
    return moves


def format_sgf(board_size, moves):
    """Formats a list of moves into an SGF string."""
    sgf_string = f"(;GM[1]FF[4]CA[UTF-8]AP[KataGoFeatureExtractor]RU[Chinese]SZ[{board_size}]KM[7.5]\n"
    current_node = ""
    for pla, loc in moves:
        player_char = 'B' if pla == Board.BLACK else 'W'
        col = loc % (board_size + 1) - 1
        row = loc // (board_size + 1) - 1
        col_char = chr(ord('a') + col)
        row_char = chr(ord('a') + row)
        current_node += f";{player_char}[{col_char}{row_char}]"
    sgf_string += current_node
    sgf_string += ")"
    return sgf_string


def main():
    """Main function to generate SGF files."""
    parser = argparse.ArgumentParser(description="Generate random SGF files for testing.")
    parser.add_argument("--output-dir", default="generated_sgfs", help="Directory to save SGF files.")
    parser.add_argument("--num-files", type=int, default=5, help="Number of SGF files to generate.")
    parser.add_argument("--num-moves", type=int, default=10, help="Number of random moves per game.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_files):
        moves = generate_plausible_game(19, args.num_moves)
        sgf_content = format_sgf(19, moves)
        filepath = os.path.join(args.output_dir, f"random_game_{i+1}.sgf")
        with open(filepath, "w") as f:
            f.write(sgf_content)
        print(f"Generated {filepath}")


if __name__ == "__main__":
    main()
