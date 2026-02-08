import sys
from sgfmill import sgf
from katago.game.board import Board
from katago.game.gamestate import GameState

def get_state_at_move(sgf_path, move_number):
    """
    Replays an SGF file to a specific move number and returns the GameState.
    Move 0 is the empty board.
    """
    try:
        with open(sgf_path, "rb") as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
    except (ValueError, FileNotFoundError):
        print(f"Error: Could not parse or find {sgf_path}", file=sys.stderr)
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
    node = sgf_game.get_root()
    while node:
        if not node[0]:
            break
        node = node[0]
            
        if node.has_property('B'):
            color, point = "B", node.get('B')
        elif node.has_property('W'):
            color, point = "W", node.get('W')
        else:
            continue
            
        current_move += 1

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
