import os
import random
import argparse
from katago.game.board import Board
from katago.game.gamestate import GameState


def generate_random_game(board_size, num_moves):
    """Generates a list of moves for a random game."""
    state = GameState(board_size=board_size, rules=GameState.RULES_TT)
    moves = []
    for _ in range(num_moves):
        possible_moves = []
        for y in range(state.board.y_size):
            for x in range(state.board.x_size):
                loc = state.board.loc(x, y)
                if state.board.would_be_legal(state.board.pla, loc):
                    possible_moves.append(loc)
        
        if not possible_moves:
            break
            
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
        moves = generate_random_game(19, args.num_moves)
        sgf_content = format_sgf(19, moves)
        filepath = os.path.join(args.output_dir, f"random_game_{i+1}.sgf")
        with open(filepath, "w") as f:
            f.write(sgf_content)
        print(f"Generated {filepath}")


if __name__ == "__main__":
    main()
