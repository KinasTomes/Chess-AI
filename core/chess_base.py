import numpy as np
import chess
from typing import Tuple
from collections import deque
import os
import sys
from six import StringIO
import gym
from gym.spaces import Box, Discrete

from core.coords_converter import ChessCoordsConverter

# Chess piece representation
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

# Chess piece symbols for rendering
PIECE_SYMBOLS = {
    EMPTY: '.',
    PAWN: 'P',
    KNIGHT: 'N',
    BISHOP: 'B',
    ROOK: 'R',
    QUEEN: 'Q',
    KING: 'K'
}

# Chess piece values for evaluation
PIECE_VALUES = {
    EMPTY: 0,
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000
}

class ChessEnv(gym.Env):
    """Chess environment."""

    def __init__(self, num_stack: int = 8, black_player_id: int = 2, white_player_id: int = 1, has_resign_move: bool = True, id: str = 'Chess') -> None:
        """
        Initialize the chess environment.

        Args:
            num_stack: Number of history states to stack, default 8.
            white_player_id: ID for white player, default 1.
            black_player_id: ID for black player, default 2.
            has_resign_move: Whether to allow resigning, default True.
            id: Environment ID or name.

        Note:
            - If white_player_id = 2, black will play first
            - If white_player_id = 1, white will play first (traditional chess)
        """
        assert black_player_id != white_player_id != 0, 'player ids can not be the same, and can not be zero'

        super().__init__()

        # Set player IDs
        self.black_player = black_player_id
        self.white_player = white_player_id
        self.opponent_player = None

        # Set environment parameters
        self.num_stack = num_stack
        self.has_resign_move = has_resign_move
        self.id = id
        self.board_size = 8

        # Initialize chess board
        self.chess_board = chess.Board()

        # Initialize coordinate converter
        self.chess_coords = ChessCoordsConverter()

        # Initialize piece boards (one for each piece type)
        self.piece_boards = np.zeros((7, self.board_size, self.board_size), dtype=np.int8)

        # Set up action and observation spaces
        self.action_dim = 4864  # Total number of possible moves in chess
        self.action_space = Discrete(self.action_dim + (1 if has_resign_move else 0))
        self.observation_space = Box(low=0, high=1, shape=(num_stack * 2 + 1, self.board_size, self.board_size), dtype=np.int8)

        # Initialize game state
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board_deltas = deque(maxlen=self.num_stack)  # Đảm bảo maxlen bằng với num_stack
        self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)
        self.last_move = None
        self.last_player = None
        self.steps = 0
        self.winner = None

        # Initialize the board with the starting position
        self._initialize_board()

        if (self.white_player == 2):
            self.chess_board.turn = chess.BLACK
            self.to_play = self.black_player
            self.opponent_player = self.white_player
        else:
            self.chess_board.turn = chess.WHITE
            self.to_play = self.white_player
            self.opponent_player = self.black_player

        # Track game state
        self.is_check = False
        self.is_checkmate = False
        self.is_stalemate = False
        self.is_insufficient_material = False
        self.is_fifty_move_rule = False
        self.is_threefold_repetition = False

        # Track captures
        self.captures = {'white': 0, 'black': 0}

        # Set up resign move
        self.resign_move = self.action_dim if has_resign_move else None

        # Update legal actions
        self._update_legal_actions()

    def _initialize_board(self):
        """Initialize the chess board with the starting position."""
        # Clear the board
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.piece_boards = np.zeros((7, 8, 8), dtype=np.int8)

        # Set up the initial position
        # White pieces
        self._place_piece(ROOK, 7, 0, self.white_player)
        self._place_piece(KNIGHT, 7, 1, self.white_player)
        self._place_piece(BISHOP, 7, 2, self.white_player)
        self._place_piece(QUEEN, 7, 3, self.white_player)
        self._place_piece(KING, 7, 4, self.white_player)
        self._place_piece(BISHOP, 7, 5, self.white_player)
        self._place_piece(KNIGHT, 7, 6, self.white_player)
        self._place_piece(ROOK, 7, 7, self.white_player)
        for i in range(8):
            self._place_piece(PAWN, 6, i, self.white_player)

        # Black pieces
        self._place_piece(ROOK, 0, 0, self.black_player)
        self._place_piece(KNIGHT, 0, 1, self.black_player)
        self._place_piece(BISHOP, 0, 2, self.black_player)
        self._place_piece(QUEEN, 0, 3, self.black_player)
        self._place_piece(KING, 0, 4, self.black_player)
        self._place_piece(BISHOP, 0, 5, self.black_player)
        self._place_piece(KNIGHT, 0, 6, self.black_player)
        self._place_piece(ROOK, 0, 7, self.black_player)
        for i in range(8):
            self._place_piece(PAWN, 1, i, self.black_player)

    def _place_piece(self, piece_type: int, rank: int, file: int, player: int):
        """Place a piece on the board."""
        self.board[rank, file] = player
        self.piece_boards[piece_type, rank, file] = 1

    def _update_legal_actions(self):
        """Update the legal actions based on the current chess board."""
        self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)
        legal_moves = list(self.chess_board.legal_moves)
        for move in legal_moves:
            try:
                move_index = self.chess_coords.move_to_index(move)
                if 0 <= move_index < self.action_dim:
                    self.legal_actions[move_index] = 1
                else:
                    print(f"Warning: Move index {move_index} is out of bounds for legal_actions with size {self.action_dim}")
            except ValueError as e:
                print(f"Warning: Could not convert move {move.uci()} to index: {e}")
                continue

    def reset(self, **kwargs) -> np.ndarray:
        self.chess_board = chess.Board()
        self._initialize_board()
        self.is_check = False
        self.is_checkmate = False
        self.is_stalemate = False
        self.is_insufficient_material = False
        self.is_fifty_move_rule = False
        self.is_threefold_repetition = False
        self.captures = {'white': 0, 'black': 0}
        self.board_deltas = deque(maxlen=self.num_stack)
        self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)
        self.last_move = None
        self.last_player = None
        self.steps = 0
        self.winner = None
        self.to_play = self.white_player
        self.opponent_player = self.black_player
        self._update_legal_actions()
        return self._observation()

    def step(self, action: int, debug: bool = True) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step within the environment.

        Args:
            action: The action to take.

        Returns:
            observation: The observation of the environment.
            reward: The reward from the action.
            done: Whether the episode is done.
            info: Additional information.
        """

        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and action != self.resign_move and not 0 <= int(action) <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')

        chess_move = self.chess_coords.index_to_move(action)
        if action is not None and action != self.resign_move:
            if not self.chess_board.is_legal(chess_move):
                if debug: print(f"[DEBUG] Chess board: {self.chess_board.fen()}")
                if debug: print(f"[DEBUG] Illegal move: {chess_move.uci()}")
                raise ValueError(f'Illegal action {action}.')

        # Handle resign move
        if action == self.resign_move:
            self.winner = self.opponent_player
            self.last_move = action
            self.last_player = self.to_play
            self.steps += 1
            self.add_to_history(self.last_player, self.last_move)
            return self._observation(), 0, True, {'result': 'resignation'}

        # Make the move on the chess board
        self.chess_board.push(chess_move)

        # Cập nhật các hành động hợp lệ mới sau khi thực hiện nước đi
        self._update_legal_actions()

        # Update the board representation
        self._update_board_from_chess_board()

        # Append board to history
        if len(self.board_deltas) >= self.num_stack:
            self.board_deltas.pop()
        self.board_deltas.appendleft(np.copy(self.board))

        # Record the move and player *before* switching
        self.last_move = action
        self.last_player = self.to_play  # Player who made the move
        self.steps += 1
        self.add_to_history(self.last_player, self.last_move)

        # Switch players
        self.to_play, self.opponent_player = self.opponent_player, self.to_play

        # Update legal actions and game state
        self._update_game_state()

        # Check if the game is over
        done = self.is_game_over()

        # Calculate reward
        reward = self._calculate_reward()

        # Prepare info dictionary
        info = {
            'is_check': self.is_check,
            'is_checkmate': self.is_checkmate,
            'is_stalemate': self.is_stalemate,
            'is_insufficient_material': self.is_insufficient_material,
            'is_fifty_move_rule': self.is_fifty_move_rule,
            'is_threefold_repetition': self.is_threefold_repetition,
            'captures': self.captures,
            'result': self.get_result_string()
        }

        return self._observation(), reward, done, info

    def _update_board_from_chess_board(self):
        """Update the board representation from the chess.Board object."""
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.piece_boards = np.zeros((7, 8, 8), dtype=np.int8)

        for square in chess.SQUARES:
            piece = self.chess_board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)  # 0-7
                file = chess.square_file(square)  # 0-7
                board_row = 7 - rank  # Convert to numpy row
                board_col = file
                piece_type = self._chess_piece_to_piece_type(piece)
                player = self.white_player if piece.color == chess.WHITE else self.black_player
                self.board[board_row, board_col] = player
                self.piece_boards[piece_type, board_row, board_col] = 1

    def _chess_piece_to_piece_type(self, piece) -> int:
        """Convert a chess.Piece to a piece type."""
        if piece.piece_type == chess.PAWN:
            return PAWN
        elif piece.piece_type == chess.KNIGHT:
            return KNIGHT
        elif piece.piece_type == chess.BISHOP:
            return BISHOP
        elif piece.piece_type == chess.ROOK:
            return ROOK
        elif piece.piece_type == chess.QUEEN:
            return QUEEN
        elif piece.piece_type == chess.KING:
            return KING
        else:
            raise ValueError(f'Unknown piece type: {piece.piece_type}')

    def _update_game_state(self):
        """Update the game state based on the current position."""
        # Check if the king is in check
        self.is_check = self.chess_board.is_check()

        # Check if the game is over
        self.is_checkmate = self.chess_board.is_checkmate()
        self.is_stalemate = self.chess_board.is_stalemate()
        self.is_insufficient_material = self.chess_board.is_insufficient_material()
        self.is_fifty_move_rule = self.chess_board.is_fifty_moves()
        self.is_threefold_repetition = self.chess_board.is_repetition(3)

        # Set winner if game is over
        if self.is_checkmate:
            self.winner = self.opponent_player  # Player who gave checkmate wins
        elif any([self.is_stalemate, self.is_insufficient_material,
                self.is_fifty_move_rule, self.is_threefold_repetition]):
            self.winner = 0  # Draw

        # Update captures
        self._update_captures()

    def _update_captures(self):
        """Track captured pieces."""
        if len(self.chess_board.move_stack) > 0:
            last_move = self.chess_board.move_stack[-1]

            if self.chess_board.is_capture(last_move):
                captured_square = last_move.to_square

                # Create a copy with full move_stack
                board_before_move = self.chess_board.copy()
                board_before_move.pop()  # Undo the last move to get the captured piece

                captured_piece = board_before_move.piece_at(captured_square)
                if captured_piece:
                    # Credit the capture to the opponent of the captured piece's color
                    captured_color = 'white' if captured_piece.color == chess.BLACK else 'black'
                    self.captures[captured_color] += PIECE_VALUES[self._chess_piece_to_piece_type(captured_piece)]


    def _calculate_reward(self) -> float:
        """Calculate the reward for the current position."""
        # If the game is over, return a large reward or penalty
        if self.is_checkmate:
            return 1.0 if self.winner == self.to_play else -1.0

        # If the game is a draw, return a small reward or penalty
        if self.is_stalemate or self.is_insufficient_material or self.is_fifty_move_rule or self.is_threefold_repetition:
            return 0.0

        # Otherwise, calculate a material-based evaluation
        return self._evaluate_position()

    def _evaluate_position(self) -> float:
        """Evaluate the current position based on material."""
        evaluation = 0.0

        # Sum up the material value for each piece
        for piece_type in range(1, 7):  # Skip EMPTY
            # Count pieces where piece_boards indicates a piece and board indicates the player
            white_pieces = np.sum((self.piece_boards[piece_type] == 1) & (self.board == self.white_player))
            black_pieces = np.sum((self.piece_boards[piece_type] == 1) & (self.board == self.black_player))
            evaluation += white_pieces * PIECE_VALUES[piece_type]
            evaluation -= black_pieces * PIECE_VALUES[piece_type]

        # Normalize the evaluation
        return evaluation / 10000.0

    def render(self, mode='terminal'):
        """Render the chess board to the terminal."""
        board = np.copy(self.board)
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if mode == 'human':
            # Clearing the Screen
            if os.name == 'posix':  # posix is os name for Linux or mac
                os.system('clear')
            else:  # else screen will be cleared for windows
                os.system('cls')

        # Head information
        outfile.write(f'{self.id} (8x8)')
        outfile.write('\n')
        outfile.write(f'Black: B, White: W')
        outfile.write('\n')
        outfile.write('\n')

        game_over_label = 'Yes' if self.is_game_over() else 'No'
        outfile.write(f'Game over: {game_over_label}, Result: {self.get_result_string()}')
        outfile.write('\n')
        outfile.write(
            f'Steps: {self.steps}, Current player: {"B" if self.to_play == self.black_player else "W"}'
        )
        outfile.write('\n')
        outfile.write('\n')

        # Add top column label
        outfile.write('     ')
        for y in range(8):
            outfile.write('{0:3}'.format(chr(ord('a') + y)))
        outfile.write('\n')
        outfile.write('   +' + '-' * 8 * 3 + '+\n')

        # Each row
        for r in range(8):
            # Add left row label
            outfile.write('{0:2} |'.format(8 - r))
            # Each column
            for c in range(0, 8):
                # Single cell.
                our_str = '.'
                if board[r, c] == self.black_player:
                    our_str = 'X'
                elif board[r, c] == self.white_player:
                    our_str = 'O'
                if (r, c) == self.action_to_coords(self.last_move) if self.last_move is not None else False:
                    our_str = f'({our_str})'
                outfile.write(f'{our_str}'.center(3))
            # Add right row label
            outfile.write('| {0:2}'.format(8 - r))
            outfile.write('\r\n')

        # Add bottom column label
        outfile.write('   +' + '-' * 8 * 3 + '+\n')
        outfile.write('     ')
        for y in range(8):
            outfile.write('{0:3}'.format(chr(ord('a') + y)))

        outfile.write('\n\n')

        # Print the FEN representation of the position
        outfile.write(f'FEN: {self.chess_board.fen()}')
        outfile.write('\n')

        return outfile

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over if there is a winner
        if self.winner is not None:
            return True

        # Game is over if there is a checkmate
        if self.is_checkmate:
            self.winner = self.opponent_player
            return True

        # Game is over if there is a stalemate
        if self.is_stalemate:
            self.winner = 0  # Draw
            return True

        # Game is over if there is insufficient material
        if self.is_insufficient_material:
            self.winner = 0  # Draw
            return True

        # Game is over if there is a fifty-move rule
        if self.is_fifty_move_rule:
            self.winner = 0  # Draw
            return True

        # Game is over if there is a threefold repetition
        if self.is_threefold_repetition:
            self.winner = 0  # Draw
            return True

        return False

    def get_result_string(self) -> str:
        """Get a string representation of the game result."""
        if self.winner is None:
            return 'In Progress'
        elif self.winner == 0:
            if self.is_stalemate:
                return 'Draw by Stalemate'
            elif self.is_insufficient_material:
                return 'Draw by Insufficient Material'
            elif self.is_fifty_move_rule:
                return 'Draw by Fifty-Move Rule'
            elif self.is_threefold_repetition:
                return 'Draw by Threefold Repetition'
            else:
                return 'Draw'
        elif self.winner == self.black_player:
            return 'Black Wins'
        elif self.winner == self.white_player:
            return 'White Wins'
        else:
            return 'Unknown'

    def _observation(self) -> np.ndarray:
        """
        Return AlphaZero-style observation: (119, 8, 8) float32 tensor.
        Includes 8-step history of positions (each with 14 planes) + 7 constant planes.
        """
        T = 8
        M = 14
        L = 7
        planes = []

        # Lấy T-1 trạng thái quá khứ từ board_deltas, cộng với hiện tại
        history_boards = []
        for b in list(self.board_deltas)[:T - 1]:
            if isinstance(b, chess.Board):
                history_boards.append(b.copy())
            else:
                history_boards.append(None)
        history_boards.insert(0, self.chess_board.copy())  # thêm trạng thái hiện tại
        while len(history_boards) < T:
            history_boards.append(None)

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        for board in history_boards:
            if board is None or not isinstance(board, chess.Board):
                planes.extend([np.zeros((8, 8), dtype=np.float32) for _ in range(M)])
                continue

            p1_planes = []
            p2_planes = []
            for piece_type in piece_types:
                p1 = np.zeros((8, 8), dtype=np.float32)
                p2 = np.zeros((8, 8), dtype=np.float32)
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece is not None and piece.piece_type == piece_type:
                        row = 7 - chess.square_rank(square)
                        col = chess.square_file(square)
                        if piece.color == board.turn:
                            p1[row, col] = 1
                        else:
                            p2[row, col] = 1
                p1_planes.append(p1)
                p2_planes.append(p2)

            rep_planes = [np.zeros((8, 8), dtype=np.float32) for _ in range(2)]
            if board.is_repetition(2):
                rep_planes[0][:, :] = 1
            if board.is_repetition(3):
                rep_planes[1][:, :] = 1

            planes.extend(p1_planes + p2_planes + rep_planes)

        # Constant planes
        color_plane = np.ones((8, 8), dtype=np.float32) if self.chess_board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
        move_count_plane = np.full((8, 8), self.chess_board.fullmove_number, dtype=np.float32)

        # Castling
        p1_castle = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]
        p2_castle = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]

        if self.chess_board.turn == chess.WHITE:
            if self.chess_board.has_kingside_castling_rights(chess.WHITE): p1_castle[0][:, :] = 1
            if self.chess_board.has_queenside_castling_rights(chess.WHITE): p1_castle[1][:, :] = 1
            if self.chess_board.has_kingside_castling_rights(chess.BLACK): p2_castle[0][:, :] = 1
            if self.chess_board.has_queenside_castling_rights(chess.BLACK): p2_castle[1][:, :] = 1
        else:
            if self.chess_board.has_kingside_castling_rights(chess.BLACK): p1_castle[0][:, :] = 1
            if self.chess_board.has_queenside_castling_rights(chess.BLACK): p1_castle[1][:, :] = 1
            if self.chess_board.has_kingside_castling_rights(chess.WHITE): p2_castle[0][:, :] = 1
            if self.chess_board.has_queenside_castling_rights(chess.WHITE): p2_castle[1][:, :] = 1

        no_progress = np.full((8, 8), self.chess_board.halfmove_clock, dtype=np.float32)
        constant_planes = [color_plane, move_count_plane] + p1_castle + p2_castle + [no_progress]
        planes.extend(constant_planes)

        obs_tensor = np.stack(planes, axis=0)
        return obs_tensor

    def add_to_history(self, player: int, action: int):
        """Add a move to the history."""
        # This method is called after each move to record the history
        # In this implementation, we don't need to do anything special
        # as the board state is already tracked in self.board and self.board_deltas
        pass

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert an action to coordinates."""
        if action == self.resign_move:
            return (-1, -1)

        # Convert action to chess move
        chess_move = self.chess_coords.index_to_move(action)

        # Return destination square coordinates
        to_square = chess_move.to_square
        return divmod(to_square, 8)  # (rank, file)

    def coords_to_action(self, coords: Tuple[int, int]) -> int:
        """Convert coordinates to an action."""
        rank, file = coords

        # Check if the coordinates are on the board
        if not self.is_coords_on_board(coords):
            return -1

        # Convert to square
        square = rank * 8 + file

        # Find a legal move that ends at this square
        for move in self.chess_board.legal_moves:
            if move.to_square == square:
                return self.chess_coords.move_to_index(move)

        return -1