import torch
from core.model import ChessNet
import training.utils as utils

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet().to(device)

    utils.collect_and_save_games(
        model,
        games_per_batch=100,    # Số game mỗi lần collect
        total_samples=50000,    # Tổng số samples cần thu thập
        samples_per_file=10000, # Số samples trong mỗi file
        mcts_sims=400,         # Số MCTS simulations mỗi nước đi
        device=device,
        save_dir="buffer_data", # Thư mục lưu buffer files
        batch_size=8,          # Batch size cho MCTS
        num_processes=8          # Số threads cho parallel MCTS
    )