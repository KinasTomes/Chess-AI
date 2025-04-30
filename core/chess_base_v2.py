import gym
import chess
import chess.polyglot
import numpy as np

from typing import Tuple
from collections import deque

from core.coords_converter import ChessCoordsConverter

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
    _fen_piece_map = {
        'P': (1, 0), 'N': (1, 1), 'B': (1, 2),
        'R': (1, 3), 'Q': (1, 4), 'K': (1, 5),
        'p': (0, 0), 'n': (0, 1), 'b': (0, 2),
        'r': (0, 3), 'q': (0, 4), 'k': (0, 5),
    }

    def __init__(self, num_stack: int = 8, first_player: bool = chess.WHITE):
        super(ChessEnv, self).__init__()

        self.num_stack = num_stack
        self.to_play = first_player

        self.board = chess.Board()
        if not first_player:
            self.board.turn = chess.BLACK
        
        self.hash_dict = dict()
        self._update_hash(self.current_hash(), +1)

        self.converter = ChessCoordsConverter()

        self.action_dim = 4864
        self.board_deltas = deque(maxlen=num_stack)
        self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)
        
        self._update_legal_actions()

    def current_hash(self) -> np.uint64:
        return np.uint64(chess.polyglot.zobrist_hash(self.board))

    def _update_hash(self, h, delta):
        self.hash_dict[h] = self.hash_dict.get(h, 0) + delta
        if self.hash_dict[h] == 0:
            del self.hash_dict[h]

    def _update_legal_actions(self) -> None:
        self.legal_actions.fill(0)
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            try:
                action = self.converter.move_to_index(move)
                if 0 <= action < self.action_dim:
                    self.legal_actions[action] = 1
                else:
                    print(f"Invalid action index: {action}")
            except ValueError as e:
                print(f"Warning: Could not convert move {move.uci()} to index: {e}")
                continue

    def _get_legal_moves(self) -> np.ndarray:
        return list(self.board.legal_moves)

    def reset(self, to_play: bool = None) -> None:
        self.board.reset()
        if to_play is not None:
            self.to_play = to_play
            self.opponent = not to_play
            self.board.turn = chess.WHITE if to_play else chess.BLACK

        self.board_deltas.clear()
        self._update_legal_actions()

    def _repetition_count(self) -> int:
        return self.hash_dict.get(self.current_hash(), 0)
    
    def _is_repetition(self, num: int = 3) -> bool:
        return self._repetition_count() >= num

    def _is_game_over(self) -> bool:
        return self.board.is_game_over()
    
    def _calculate_reward(self) -> float:
        board = self.board

        if board.is_checkmate():
            return 1.0 if board.result() == ("1-0" if board.turn else "0-1") else -1.0

        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            return 0.0

        return self._evaluate_position()

    def _evaluate_position(self) -> float:
        evaluation = sum(
            (len(self.board.pieces(pt, chess.WHITE)) - len(self.board.pieces(pt, chess.BLACK))) * val
            for pt, val in PIECE_VALUES.items() if pt != EMPTY
        )
        return evaluation / 10000.0

    def step(self, action: chess.Move) -> Tuple[np.ndarray, float, bool]:
        if self._is_game_over():
            raise RuntimeError("Game is over, call reset() before using step method.")
        if action is not None and action not in list(self.board.legal_moves):
            raise ValueError(f"Invalid action: {action} is not legal.")
        
        self.board.push(action)
        self._update_legal_actions()
        self._update_hash(self.current_hash(), +1)

        if len(self.board_deltas) == self.num_stack:
            self.board_deltas.popleft()
        self.board_deltas.append(self.board.fen())

        self.to_play = not self.to_play

        done = self._is_game_over()

        reward = self._calculate_reward()

        return self._observation(), reward, done
    
    def _fen_to_presentation(self, fen: str, turn_color: bool) -> np.ndarray:
        """
        Chuyển FEN thành 14 input planes:
        - 6 planes quân người chơi hiện tại
        - 6 planes quân đối thủ
        - 2 planes repetition
        """
        planes = np.zeros((14, 8, 8), dtype=np.float32)
        board_fen = fen.split(' ')[0]

        row = 0
        col = 0
        for ch in board_fen:
            if ch == '/':
                row += 1
                col = 0
            elif ch.isdigit():
                col += int(ch)
            else:
                player, piece_idx = ChessEnv._fen_piece_map[ch]
                # Nếu player == turn_color → là người chơi hiện tại → group 0
                # Nếu player != turn_color → đối thủ → group 1
                group = 0 if player == turn_color else 1
                plane_idx = piece_idx + 6 * group
                planes[plane_idx, row, col] = 1
                col += 1

        if self._is_repetition(2):
            planes[12, :, :] = 1
        if self._is_repetition(3):
            planes[13, :, :] = 1

        return planes

    
    def _observation(self):
        planes = []
        player = self.to_play

        for fen in self.board_deltas:
            planes.append(self._fen_to_presentation(fen, player := not player))
            
        while len(planes) < self.num_stack:
            planes.append(np.zeros((14, 8, 8), dtype=np.float32))

        color_planes = np.zeros((8, 8), dtype=np.float32)
        if self.board.turn == chess.BLACK:
            color_planes.fill(1)
        move_count_plane = np.full((8, 8), self.board.fullmove_number, dtype=np.float32)

        p1_castle = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]
        p2_castle = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]

        if self.board.turn == chess.WHITE:
            if self.board.has_kingside_castling_rights(chess.WHITE): p1_castle[0][:, :] = 1
            if self.board.has_queenside_castling_rights(chess.WHITE): p1_castle[1][:, :] = 1
            if self.board.has_kingside_castling_rights(chess.BLACK): p2_castle[0][:, :] = 1
            if self.board.has_queenside_castling_rights(chess.BLACK): p2_castle[1][:, :] = 1
        else:
            if self.board.has_kingside_castling_rights(chess.BLACK): p1_castle[0][:, :] = 1
            if self.board.has_queenside_castling_rights(chess.BLACK): p1_castle[1][:, :] = 1
            if self.board.has_kingside_castling_rights(chess.WHITE): p2_castle[0][:, :] = 1
            if self.board.has_queenside_castling_rights(chess.WHITE): p2_castle[1][:, :] = 1

        no_progress = np.full((8, 8), self.board.halfmove_clock, dtype=np.float32)
        constant_planes = [color_planes, move_count_plane] + p1_castle + p2_castle + [no_progress]
        
        planes = np.concatenate(planes, axis=0) 

        return np.concatenate([planes, constant_planes], axis=0)