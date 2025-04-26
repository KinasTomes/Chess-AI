import chess
import numpy as np
from core.mcts import MCTS, MCTSNode
from core.chess_base import ChessEnv
from core.model import ChessNet
from training.utils import load_predict_model

def print_mcts_tree(node, depth=0, max_depth=3, prefix=""):
    if depth > max_depth:
        return

    # In thông tin node
    move_str = node.move.uci() if node.move else "root"
    value = node.get_value()
    visits = node.visits
    parent_id = node.parent.id if node.parent else "None"

    if node.visits == 0:
        return

    # In thông tin của node
    print(f"{prefix}Node ID: {node.id} (Parent: {parent_id})")
    print(f"{prefix}    Move: {move_str}")
    print(f"{prefix}    Visits: {visits}")
    print(f"{prefix}    Value: {value:.3f}")
    print(f"{prefix}    Prior: {node.prior:.3f}")
    print(f"{prefix}    Virtual Loss: {node.virtual_loss}")
    print(f"{prefix}    {'-'*30}")  # Dùng dấu '-' để phân tách các node

    # In các node con
    for child in node.children:
        print_mcts_tree(child, depth + 1, max_depth, prefix)  # Giữ nguyên prefix mà không thay đổi

def test_full_mcts_run():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r'model_checkpoint\best_model.pth', model)
    model.to('cuda')
    model.eval()

    # Initialize MCTS
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=200,
        max_depth=50,
        device='cuda',
        num_processes=4,
        use_model=True,
        temperature=1.0
    )

    # Test full MCTS run
    print("\n=== Testing Full MCTS Run ===")
    print("\nInitial board state:")
    print(env.chess_board)
    
    move_probs = mcts.run(env.chess_board)
    
    print("\nMCTS Tree after search:")
    print_mcts_tree(mcts.root, max_depth=2)  # In ra cây với độ sâu tối đa là 2
    
    print("\nMove probabilities:")
    legal_moves = list(env.chess_board.legal_moves)
    for move in legal_moves:
        idx = env.chess_coords.move_to_index(move)
        print(f"Move {move.uci()}: {move_probs[idx]:.3f}")

if __name__ == "__main__":
    # Run all tests
    # test_mcts_selection()
    # test_mcts_expansion()
    # test_mcts_simulation()
    # test_mcts_backpropagation()
    test_full_mcts_run()