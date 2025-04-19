import torch
import chess
import numpy as np
from core.model import ChessNet
from training.utils import load_predict_model
from core.chess_base import ChessEnv  # Lớp môi trường cờ

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet()
    model = load_predict_model(r"model_checkpoint\best_model.pth", model)
    model.to(device)
    model.eval()

    env = ChessEnv()
    env.reset()

    for step_num in range(10):
        state = env._observation()
        legal_moves = list(env.chess_board.legal_moves)

        # Tạo mask
        mask = np.zeros(env.action_dim, dtype=np.float32)
        move_idx_map = {}  # Map index -> move (để chọn move tốt nhất)
        for move in legal_moves:
            idx = env.chess_coords.move_to_index(move)
            mask[idx] = 1
            move_idx_map[idx] = move

        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            policy, _ = model(input_tensor, mask_tensor)
            policy = policy.squeeze().cpu().numpy()

        # Zero-out các action không hợp lệ
        legal_policy = policy * mask
        best_move_idx = np.argmax(legal_policy)
        best_move = move_idx_map[best_move_idx]  # Lấy move từ index

        # Thực hiện nước đi
        env.step(env.chess_coords.move_to_index(chess.Move.from_uci(best_move.uci())))

        # In thông tin
        print(f"\nStep {step_num + 1}: Move {best_move.uci()} made.")
        # print(env.chess_board)  # In ra board hiện tại
