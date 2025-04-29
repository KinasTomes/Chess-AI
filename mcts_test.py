import torch
import numpy as np

from typing import List, Tuple, Iterable

from core.chess_base_v2 import ChessEnv
from core.coords_converter import ChessCoordsConverter
from core.mcts_v2_fake import MCTSNode, uct_search, DummyNode, parallel_uct_search
from core.model import ChessNet
from training.utils import load_model

model = ChessNet()
model = load_model(r'model_checkpoint\best_model.pth', model)
model.to('cuda')
model.eval()

device = 'cuda'

def _evaluate(obs: np.ndarray, mask: np.ndarray) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
    with torch.no_grad():
        state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)

        policy, value = model(state_tensor, mask_tensor)
        policy = policy.squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()

        return policy, value

def _parallel_evaluate(obs_batch: List[np.ndarray], mask_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched evaluation using a neural network model.

    Args:
        obs_batch: List of observations (np.ndarray), shape [C, 8, 8] per item
        mask_batch: List of legal move masks (np.ndarray), shape [A] per item
        model: The model to evaluate, returns (policy, value)
        device: 'cuda' or 'cpu'

    Returns:
        policies: np.ndarray of shape [B, A]
        values: np.ndarray of shape [B]
    """
    with torch.no_grad():
        # Stack input batches
        states = np.stack(obs_batch)         # [B, C, 8, 8]
        masks = np.stack(mask_batch)         # [B, A]

        # Convert to tensors
        state_tensor = torch.from_numpy(states).float().to(device)
        mask_tensor = torch.from_numpy(masks).float().to(device)

        # Run model
        policy, value = model(state_tensor, mask_tensor)  # policy [B, A], value [B, 1] or [B]

        # Ensure shapes are right
        policies = policy.cpu().numpy()                   # [B, A] (already softmaxed)
        values = value.squeeze(-1).cpu().tolist()         # [B]

    return policies, values

if __name__ == '__main__':
    env = ChessEnv()
    root_node = MCTSNode(int(env.to_play), parent=DummyNode())
    c_puct_base = 2.0
    c_puct_init = 1.0
    num_sims = 800

    move, search_pi, root_q, best_child_q, next_root_node = parallel_uct_search(
        env=env,
        eval_func=_parallel_evaluate,
        root_node=root_node,
        c_puct_base=c_puct_base,
        c_puct_init=c_puct_init,
        num_sims=num_sims
    )

    print(f"Best move: {move}")
    print(f"Search policy (action probabilities): {search_pi}")
    print(f"Root Q value: {root_q}")
    print(f"Best child Q value: {best_child_q}")

    # Kiểm tra nếu trạng thái tiếp theo là hợp lệ
    converter = ChessCoordsConverter()
    env.step(converter.index_to_move(move))
    print(f"New board state: {env.board.fen()}")
    print(f"Game over: {env._is_game_over()}")