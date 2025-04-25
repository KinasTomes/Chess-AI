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
    print(f"{prefix}    {'-'*30}")  # Dùng dấu '-' để phân tách các node

    # In các node con
    for child in node.children:
        print_mcts_tree(child, depth + 1, max_depth, prefix)  # Giữ nguyên prefix mà không thay đổi

def test_mcts_selection():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.eval()

    # Initialize MCTS
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=400,
        max_depth=50,
        device='cuda',
        num_processes=4,
        use_model=True,
        temperature=1.0
    )

    # Create root node
    root_node = MCTSNode(env.chess_board, env)
    
    # Test selection
    print("\n=== Testing MCTS Selection ===")
    selected_node, path = mcts._select(root_node)
    
    print("\nSelected path:")
    for i, node in enumerate(path):
        move_str = node.move.uci() if node.move else "root"
        print(f"Step {i}: Move {move_str}, Visits: {node.visits}, Value: {node.get_value():.3f}")
        print(f"Virtual Loss: {node.virtual_loss}")
        print(f"Prior: {node.prior:.3f}")
        print("-" * 30)

def test_mcts_expansion():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.eval()

    # Initialize MCTS
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=400,
        max_depth=50,
        device='cuda',
        num_processes=4,
        use_model=True,
        temperature=1.0
    )

    # Create root node
    root_node = MCTSNode(env.chess_board, env)
    
    # Test expansion
    print("\n=== Testing MCTS Expansion ===")
    mcts._expand(root_node)
    
    print("\nExpanded children:")
    for child in root_node.children:
        print(f"Move: {child.move.uci()}")
        print(f"Prior: {child.prior:.3f}")
        print(f"Value: {child.value:.3f}")
        print(f"Visits: {child.visits}")
        print(f"Virtual Loss: {child.virtual_loss}")
        print("-" * 30)

def test_mcts_simulation():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.eval()

    # Initialize MCTS
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=400,
        max_depth=50,
        device='cuda',
        num_processes=4,
        use_model=True,
        temperature=1.0
    )

    # Create root node
    root_node = MCTSNode(env.chess_board, env)
    
    # Test simulation
    print("\n=== Testing MCTS Simulation ===")
    value = mcts._simulate(root_node)
    print(f"Simulation value: {value:.3f}")
    print(f"Using model prediction: {mcts.use_model}")

def test_mcts_backpropagation():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.eval()

    # Initialize MCTS
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=400,
        max_depth=50,
        device='cuda',
        num_processes=4,
        use_model=True,
        temperature=1.0
    )

    # Create root node and expand it
    root_node = MCTSNode(env.chess_board, env)
    mcts._expand(root_node)
    
    # Test backpropagation
    print("\n=== Testing MCTS Backpropagation ===")
    value = 1.0  # Simulate a win
    mcts._backprop(root_node.children[0], value)
    
    print("\nNode statistics after backpropagation:")
    print(f"Root visits: {root_node.visits}")
    print(f"Root value sum: {root_node.value_sum}")
    print(f"Root value: {root_node.get_value():.3f}")
    print(f"Child visits: {root_node.children[0].visits}")
    print(f"Child value sum: {root_node.children[0].value_sum}")
    print(f"Child value: {root_node.children[0].get_value():.3f}")

def test_full_mcts_run():
    # Initialize environment and model
    env = ChessEnv()
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
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