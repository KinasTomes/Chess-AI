import chess
import torch
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import List
import time

from core.chess_base import PIECE_VALUES

class MCTSNode:
    def __init__(self, board: chess.Board, env, parent=None, move=None):
        self.board = board
        self.env = env
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.prior = 0
        self.is_expanded = False
        self.virtual_loss = 0  # Thêm virtual loss cho parallel search

    def value(self):
        if self.visits + self.virtual_loss == 0:
            return 0
        return self.value_sum / (self.visits + self.virtual_loss)

    def add_virtual_loss(self):
        self.virtual_loss += 3  # Tăng virtual loss để giảm khả năng thread khác chọn node này

    def revert_virtual_loss(self):
        self.virtual_loss = max(0, self.virtual_loss - 3)

    @property
    def is_terminal(self):
        return self.board.is_game_over()

def _worker_search(args):
    """
    Worker function for parallel search using model predictions.

    Args:
        args (tuple): A tuple containing:
            - board_fen (str): The FEN string representing the chess board state.
            - env: The environment object used for simulations.
            - max_depth (int): The maximum depth for the simulation.
            - neural_net: The neural network model for predictions.
            - converter: The move converter.
            - device: The device to run the model on.

    Returns:
        tuple: A tuple containing:
            - path (list): A list of moves made during the simulation.
            - value (int): The result of the simulation (1 for white win, -1 for black win, 0 for draw).
    """
    board_fen, env, max_depth, neural_net, converter, device = args
    board = chess.Board(board_fen)
    sim_env = deepcopy(env)
    sim_env.chess_board = board.copy()
    path = []
    current_depth = 0

    while not board.is_game_over() and current_depth < max_depth:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # Get model prediction
        state = sim_env._observation()
        mask = np.zeros(env.action_dim, dtype=np.float32)
        for move in legal_moves:
            idx = converter.move_to_index(move)
            mask[idx] = 1

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)

        with torch.no_grad():
            policy, _ = neural_net(state_tensor, mask_tensor)
            policy = policy.squeeze().cpu().numpy()

        # Filter policy for legal moves
        legal_policy = np.zeros_like(policy)
        for move in legal_moves:
            idx = converter.move_to_index(move)
            legal_policy[idx] = policy[idx]

        # Normalize policy
        if np.sum(legal_policy) > 0:
            legal_policy /= np.sum(legal_policy)

        # Select move based on policy
        move_idx = np.random.choice(len(legal_policy), p=legal_policy)
        move = converter.index_to_move(move_idx)

        board.push(move)
        current_depth += 1
        path.append(move)

    # Return simulation result
    result = board.result()
    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    return path, value

def _worker_search_v2(args):
    """
    Worker function for parallel search using capture-first strategy.

    Args:
        args (tuple): A tuple containing:
            - board_fen (str): The FEN string representing the chess board state.
            - env: The environment object used for simulations.
            - max_depth (int): The maximum depth for the simulation.

    Returns:
        tuple: A tuple containing:
            - path (list): A list of moves made during the simulation.
            - value (int): The result of the simulation (1 for white win, -1 for black win, 0 for draw).
    """
    board_fen, env, max_depth = args
    board = chess.Board(board_fen)
    sim_env = deepcopy(env)
    sim_env.chess_board = board.copy()
    path = []
    current_depth = 0

    while not board.is_game_over() and current_depth < max_depth:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # Find captures and their values
        captures = []
        for move in legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_value = PIECE_VALUES[env._chess_piece_to_piece_type(captured_piece)]
                    captures.append((move, piece_value))

        if captures:
            # Sort captures by piece value (highest first)
            captures.sort(key=lambda x: x[1], reverse=True)
            move = captures[0][0]  # Select highest value capture
        else:
            # Random move if no captures
            move = np.random.choice(legal_moves)

        board.push(move)
        current_depth += 1
        path.append(move)

    # Return simulation result
    result = board.result()
    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    return path, value

class MCTS:
    def __init__(self, neural_net, converter, env, c_puct=1.0, simulations=100,
                 max_depth=50, device='cpu', num_processes=None, use_model=False):
        self.neural_net = neural_net
        self.converter = converter
        self.env = env
        self.c_puct = c_puct
        self.simulations = simulations
        self.max_depth = max_depth
        self.device = device
        self.use_model = use_model

        # Khởi tạo process pool
        if num_processes is None:
            num_processes = cpu_count()
        self.num_processes = num_processes
        self.pool = Pool(processes=num_processes)

        # Cache cho neural network predictions
        self.prediction_cache = {}

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def run(self, root_board: chess.Board):
        root = MCTSNode(root_board, self.env)

        # Prepare args for parallel search based on use_model
        if self.use_model:
            search_args = [(root_board.fen(), self.env, self.max_depth, 
                          self.neural_net, self.converter, self.device)
                          for _ in range(self.simulations)]
            worker_func = _worker_search
        else:
            search_args = [(root_board.fen(), self.env, self.max_depth)
                          for _ in range(self.simulations)]
            worker_func = _worker_search_v2

        # Perform parallel search
        t_start = time.time()
        results = self.pool.map(worker_func, search_args)

        # Xử lý kết quả và cập nhật statistics
        move_stats = {}  # {move: (visits, value_sum)}
        for path, value in results:
            if path:  # Nếu path không rỗng
                first_move = path[0]
                if first_move not in move_stats:
                    move_stats[first_move] = [0, 0]  # [visits, value_sum]
                move_stats[first_move][0] += 1
                move_stats[first_move][1] += value

        # Tính probabilities
        move_probs = np.zeros(self.env.action_dim)
        total_visits = sum(stats[0] for stats in move_stats.values())

        if total_visits > 0:
            for move, (visits, _) in move_stats.items():
                idx = self.converter.move_to_index(move)
                move_probs[idx] = visits / total_visits

            # Thêm Dirichlet noise cho root node
            legal_moves = list(root_board.legal_moves)
            noise_probs = np.zeros_like(move_probs)
            noise = np.random.dirichlet([0.3] * len(legal_moves))

            for i, move in enumerate(legal_moves):
                idx = self.converter.move_to_index(move)
                noise_probs[idx] = noise[i]

            # Mix với tỷ lệ 75-25
            move_probs = 0.75 * move_probs + 0.25 * noise_probs

            # Normalize lại
            if np.sum(move_probs) > 0:
                move_probs /= np.sum(move_probs)

        return move_probs

    def _evaluate_batch(self, states: List[np.ndarray], masks: List[np.ndarray]):
        """Đánh giá batch states với neural network."""
        if not states:
            return [], []

        # Convert to tensors
        batch_states = torch.from_numpy(np.stack(states)).float().to(self.device)
        batch_masks = torch.from_numpy(np.stack(masks)).float().to(self.device)

        with torch.no_grad():
            policies, values = self.neural_net(batch_states, batch_masks)

        return policies.cpu().numpy(), values.cpu().numpy()

    def _legal_moves_mask(self, board: chess.Board) -> np.ndarray:
        """Create mask for legal moves."""
        mask = np.zeros(self.env.action_dim, dtype=np.float32)
        for move in board.legal_moves:
            idx = self.converter.move_to_index(move)
            mask[idx] = 1
        return mask