import torch
import chess
import numpy as np
from core.model import ChessNet
from training.utils import load_predict_model
from core.chess_base import ChessEnv
from core.mcts import MCTS
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    
    # Khởi tạo device
    model = ChessNet()
    model = load_predict_model(r'model_checkpoint\best_model.pth', model)
    model.to('cuda')
    model.eval()
    
    # Khởi tạo môi trường
    env = ChessEnv()
    env.reset()

    # Khởi tạo MCTS không sử dụng neural network
    mcts = MCTS(
        neural_net=model,  # Không sử dụng neural network
        converter=env.chess_coords,
        env=env,
        simulations=200,  # Số lượt mô phỏng cho mỗi nước đi
        max_depth=30,     # Độ sâu tối đa cho mỗi mô phỏng
        device='cuda',
        num_processes=4,  # Số process cho parallel search
        use_model=False   # Không sử dụng model để dự đoán nước đi
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

    # In kết quả game
    result = env.chess_board.result()
    print(f"\n🏁 Game kết thúc sau {move_count} nước đi")
    print(f"Kết quả: {result}")