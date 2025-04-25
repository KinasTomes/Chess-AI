import sys
import chess
import torch
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple
import time
import threading
from queue import Queue
import asyncio
from collections import OrderedDict
from threading import Event
from collections import deque

from core.model import ChessNet
from core.chess_base import ChessEnv

class MCTSNode:
    _id_counter = 0  # Class variable to keep track of node IDs
    
    def __init__(self, board: chess.Board, env, parent=None, move=None):
        self.id = MCTSNode._id_counter
        MCTSNode._id_counter += 1
        self.board = board
        self.env = env
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.prior = 0
        self.is_expanded = False
        self.virtual_loss = 0
        self.value = 0
        self.lock = threading.Lock()  # Lock cho mỗi node

    def get_value(self):
        with self.lock:
            if self.visits + self.virtual_loss == 0:
                return 0
            return self.value_sum / (self.visits + self.virtual_loss)

    def add_virtual_loss(self):
        with self.lock:
            self.virtual_loss += 3

    def revert_virtual_loss(self):
        with self.lock:
            self.virtual_loss = max(0, self.virtual_loss - 3)

    @property
    def is_terminal(self):
        return self.board.is_game_over()

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

class MCTS:
    def __init__(self, neural_net, converter, env, c_puct=1.0, simulations=100,
                 max_depth=50, device='cpu', num_processes=None, use_model=False,
                 temperature=1.0):
        self.neural_net = neural_net
        self.converter = converter
        self.env = env
        self.c_puct = c_puct
        self.simulations = simulations
        self.max_depth = max_depth
        self.device = device
        self.use_model = use_model
        self.temperature = temperature
        self.root = None  # Lưu root node để reuse

        # Khởi tạo process pool và thread pool
        if num_processes is None:
            num_processes = cpu_count()
        self.num_processes = num_processes
        self.pool = Pool(processes=num_processes)
        
        # Thread pool cho async inference
        self.thread_pool = []
        self.inference_queue = Queue()
        self.result_queue = Queue()
        self.lock = threading.Lock()
        
        # Cache cho neural network predictions
        self.prediction_cache = {}

    def __del__(self):
        self.pool.close()
        self.pool.join()
        for thread in self.thread_pool:
            thread.join()

    def run(self, root_board: chess.Board):
        # Reuse root node nếu có
        if self.root is not None:
            # Tìm child node tương ứng với nước đi vừa thực hiện
            for child in self.root.children:
                if child.board == root_board:
                    self.root = child
                    break
                else:
                    # Nếu không tìm thấy, tạo root mới
                    self.root = MCTSNode(root_board, self.env)
        else:
            self.root = MCTSNode(root_board, self.env)

        # Khởi tạo thread pool nếu chưa có
        if not self.thread_pool:
            for _ in range(self.num_processes):
                thread = threading.Thread(target=self._inference_worker)
                thread.daemon = True
                thread.start()
                self.thread_pool.append(thread)

        # Thực hiện simulations lần
        for _ in range(self.simulations):
            try:
                # 1. Select: Chọn node để mở rộng
                node, path = self._select(self.root)
                
                # 2. Expand: Mở rộng node đã chọn
                if not node.is_terminal:
                    self._expand(node)
                
                # 3. Simulate: Mô phỏng từ node đã mở rộng
                value = self._simulate(node)
                
                # 4. Backprop: Cập nhật thông tin ngược lên
                self._backprop(node, value)
            except Exception as e:
                print(f"Error in MCTS simulation: {str(e)}")
                continue
            finally:
                # Revert virtual loss cho tất cả nodes trong path
                if 'path' in locals():  # Kiểm tra xem path có được định nghĩa không
                    for node in path:
                        node.revert_virtual_loss()

        # Tính xác suất cho các nước đi
        move_probs = self._get_move_probs(root_board)
        return move_probs

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Chọn node để mở rộng dựa trên UCB.
        
        Returns:
            Tuple[MCTSNode, List[MCTSNode]]: (node được chọn, path từ root đến node đó)
        """
        path = []  # Lưu lại path để revert virtual loss nếu cần
        while node.is_expanded and not node.is_terminal:
            path.append(node)
            node.add_virtual_loss()  # Thêm virtual loss khi đi xuống
            
            # Tính UCB cho mỗi child
            ucb_values = []
            for child in node.children:
                # UCB = Q + U
                # Q = value
                # U = c_puct * P * sqrt(N) / (1 + n)
                Q = child.get_value()
                U = self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                ucb_values.append(Q + U)
            
            # Chọn child có UCB cao nhất
            best_child_idx = np.argmax(ucb_values)
            node = node.children[best_child_idx]
        
        path.append(node)  # Thêm node cuối cùng vào path
        return node, path

    def _expand(self, node: MCTSNode):
        """Mở rộng node bằng cách thêm các children."""
        if not node.is_expanded:
            # Tạo children cho mỗi nước đi hợp lệ
            for move in node.board.legal_moves:
                # Tạo bản sao hoàn chỉnh của môi trường
                child_env = deepcopy(self.env)
                child_env.chess_board = node.board.copy()
                child_env.chess_board.push(move)
                
                # Cập nhật board_deltas cho child_env
                child_env.board_deltas = deque(maxlen=8)
                child_env.board_deltas.appendleft(np.copy(child_env.board))
                
                # Tạo node con với môi trường mới
                child = MCTSNode(child_env.chess_board, child_env, parent=node, move=move)
                node.children.append(child)
            
            # Lấy policy và value từ neural network cho node hiện tại
            state = node.env._observation()
            mask = self._legal_moves_mask(node.board)
            
            # Thêm vào inference queue
            self.inference_queue.put((state, mask))
            
            # Lấy kết quả từ result queue
            policy, value = self.result_queue.get()
            
            # Cập nhật prior và value cho tất cả children
            for child in node.children:
                child.prior = policy[self.converter.move_to_index(child.move)]
                child.value = value  # Lưu value prediction
            
            node.is_expanded = True

    def _simulate(self, node: MCTSNode) -> float:
        """Mô phỏng từ node đến khi kết thúc game hoặc sử dụng value prediction."""
        if self.use_model:
            # Sử dụng neural network để đánh giá trạng thái
            state = node.env._observation()
            mask = self._legal_moves_mask(node.board)
            
            # Thêm vào inference queue
            self.inference_queue.put((state, mask))
            
            # Lấy kết quả từ result queue
            _, value = self.result_queue.get()
            return float(value)
        else:
            # Rollout đến cuối game
            board = node.board.copy()
            current_depth = 0
            
            while not board.is_game_over() and current_depth < self.max_depth:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                    
                move = np.random.choice(legal_moves)
                board.push(move)
                current_depth += 1
            
            # Trả về kết quả game
            result = board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return -1.0
            else:
                return 0.0

    def _backprop(self, node: MCTSNode, value: float):
        """Cập nhật thông tin ngược lên cây."""
        while node is not None:
            with self.lock:
                node.visits += 1
                node.value_sum += value
            value = -value  # Đảo ngược giá trị cho người chơi khác
            node = node.parent

    def _inference_worker(self):
        """Worker thread cho async inference."""
        while True:
            state, mask = self.inference_queue.get()
            
            # Convert to tensors
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
            
            # Get prediction from model
            with torch.no_grad():
                policy, value = self.neural_net(state_tensor, mask_tensor)
                policy = policy.squeeze().cpu().numpy()
                value = value.squeeze().cpu().numpy()
            
            # Put result back
            self.result_queue.put((policy, value))

    def _legal_moves_mask(self, board: chess.Board) -> np.ndarray:
        """Tạo mask cho các nước đi hợp lệ."""
        mask = np.zeros(self.env.action_dim, dtype=np.float32)
        for move in board.legal_moves:
            idx = self.converter.move_to_index(move)
            mask[idx] = 1
        return mask
    
    def _revert_virtual_loss_path(self, node: MCTSNode):
        """
        Revert virtual loss cho tất cả nodes trong path từ node hiện tại lên root.
        
        Args:
            node: Node hiện tại cần revert virtual loss
        """
        current = node
        while current is not None:
            current.revert_virtual_loss()
            current = current.parent

    def _get_move_probs(self, root_board: chess.Board) -> np.ndarray:
        """
        Tính xác suất cho các nước đi dựa trên số lần thăm của mỗi node.
        
        Args:
            root_board: Bàn cờ hiện tại
            
        Returns:
            np.ndarray: Mảng xác suất cho mỗi nước đi
        """
        move_probs = np.zeros(self.env.action_dim)
        total_visits = sum(child.visits for child in self.root.children)

        if total_visits > 0:
            for child in self.root.children:
                idx = self.converter.move_to_index(child.move)
                move_probs[idx] = child.visits / total_visits

            # Thêm Dirichlet noise cho root node
            legal_moves = list(root_board.legal_moves)
            noise_probs = np.zeros_like(move_probs)
            noise = np.random.dirichlet([0.3] * len(legal_moves))

            for i, move in enumerate(legal_moves):
                idx = self.converter.move_to_index(move)
                noise_probs[idx] = noise[i]

            # Mix với tỷ lệ 75-25
            move_probs = 0.75 * move_probs + 0.25 * noise_probs

            # Apply temperature
            if self.temperature != 1.0:
                move_probs = np.power(move_probs, 1.0 / self.temperature)
                move_probs /= np.sum(move_probs)

            # Normalize
            if np.sum(move_probs) > 0:
                move_probs /= np.sum(move_probs)

        return move_probs

def load_predict_model(path: str, model: ChessNet, device: str = "cuda"):
    """
    Load model chỉ để predict, không cần load optimizer và scheduler.
    Hàm này nhẹ hơn và nhanh hơn load_training_model.
    
    Args:
        path: Đường dẫn tới file checkpoint
        model: Instance của model cần load weights vào
        device: Device để load model (cuda/cpu)
    
    Returns:
        ChessNet: Model đã load weights
    """
    try:
        # Load với map_location để có thể load model từ GPU sang CPU hoặc ngược lại
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Chuyển model sang device phù hợp
        model = model.to(device)
        
        # Chuyển sang eval mode
        model.eval()
        
        print(f"✅ Loaded model for prediction from {path}")
        if 'loss' in checkpoint:
            print(f"   └── Model loss: {checkpoint['loss']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model for prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Khởi tạo device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Khởi tạo và load model
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.to(device)
    model.eval()

    # Khởi tạo môi trường
    env = ChessEnv()
    env.reset()

    # Khởi tạo MCTS với model đã load
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=400,  # Tăng số lượt mô phỏng để có kết quả tốt hơn
        max_depth=50,     # Độ sâu tối đa cho mỗi mô phỏng
        device=device,
        num_processes=4,  # Số process cho parallel search
        use_model=True,    # Sử dụng model để dự đoán nước đi
        temperature=1.0    # Không sử dụng temperature
    )

    move_count = 0
    print("🎮 Bắt đầu game tự đánh...")

    while not env.is_game_over():
        # In trạng thái bàn cờ
        print("\n" + str(env.chess_board))
        
        # Chạy MCTS để tìm nước đi tốt nhất
        pi = mcts.run(env.chess_board)
        
        # Chọn nước đi dựa trên policy từ MCTS
        valid_moves = env.legal_actions
        pi_valid = pi * valid_moves
        
        if np.sum(pi_valid) > 0:
            if move_count < 30:  # Temperature = 1 cho 30 nước đầu
                pi_valid = pi_valid / np.sum(pi_valid)
                action = np.random.choice(len(pi), p=pi_valid)
            else:  # Temperature = 0 (greedy) sau 30 nước
                action = np.argmax(pi_valid)
        else:
            action = np.random.choice(np.where(valid_moves)[0])

        # Thực hiện nước đi
        move_uci = env.chess_coords.index_to_move(action)
        print(f"Move {move_count+1}: {move_uci} (policy: {pi[action]:.4f})")
        
        env.step(action)
        move_count += 1
        break

    # In kết quả game
    result = env.chess_board.result()
    print(f"\n🏁 Game kết thúc sau {move_count} nước đi")
    print(f"Kết quả: {result}")