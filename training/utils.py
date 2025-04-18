import os
import glob
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from core.mcts import MCTS
from core.model import ChessNet, AlphaLoss
from core.chess_base import ChessEnv
from training.replay_buffer import ReplayBuffer

from torch.utils.data import Dataset, DataLoader, TensorDataset

class ChessDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]
    
def collect_self_play_games(model: ChessNet, num_games: int = 100, mcts_sims: int = 800,
                          max_depth = 30, device: str = "cuda", save_dir: str = "game_logs",
                          batch_size: int = 8, num_processes: int = 4, use_model: bool = False):
    """
    Thu thập dữ liệu từ các ván tự chơi và lưu vào replay buffer.

    Args:
        model: Mô hình neural network
        num_games: Số ván chơi cần thu thập
        mcts_sims: Số lượt mô phỏng MCTS cho mỗi nước đi
        device: Thiết bị tính toán (cuda/cpu)
        save_dir: Thư mục lưu log
        batch_size: Kích thước batch cho MCTS
        num_processes: Số nhân CPU
        use_model: Có sử dụng model để dự đoán nước đi hay không

    Returns:
        replay_buffer: Buffer chứa dữ liệu game
    """
    env = ChessEnv()
    replay_buffer = ReplayBuffer()

    total_moves = 0
    total_time = 0

    for game_idx in tqdm(range(num_games), desc="Playing games"):
        print(f"\n🎮 Game {game_idx + 1}/{num_games}")
        game_start = time.time()
        state = env.reset()
        game_history = []
        move_count = 0

        mcts = MCTS(
            neural_net=model,
            converter=env.chess_coords,
            env=env,
            simulations=mcts_sims,
            max_depth=max_depth,
            device=device,
            num_processes=num_processes,
            use_model=use_model
        )

        while not env.is_game_over():
            # MCTS simulation với batch processing
            # print(f"    🔍 Move {move_count + 1}...", end='', flush=True)
            pi = mcts.run(env.chess_board)
            # print(" ✅")  # kết thúc move

            # Lưu state và policy
            game_history.append({
                'state': env._observation(),  # AlphaZero style observation
                'policy': pi,
                'player': env.to_play
            })

            # Chọn action với temperature scheduling
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

            move_uci = env.chess_coords.index_to_move(action)
            print(f"        ➡️ Action: {move_uci}")

            # Thực hiện nước đi
            _, reward, done, info = env.step(action)
            move_count += 1

        # Game kết thúc, tính kết quả
        game_time = time.time() - game_start
        print(f"🏁 Game finished in {move_count} moves, time: {game_time:.2f}s")
        total_time += game_time
        total_moves += move_count

        if env.winner == env.white_player:
            game_result = 1
        elif env.winner == env.black_player:
            game_result = -1
        else:
            game_result = 0

        # Tạo training samples và thêm vào buffer
        for hist in game_history:
            # Value từ góc nhìn của người chơi tại thời điểm đó
            if hist['player'] == env.white_player:
                value = game_result
            else:
                value = -game_result

            replay_buffer.add_game([(hist['state'], hist['policy'], value)])

    return replay_buffer

def collect_and_save_games(model: ChessNet,
                          games_per_batch: int = 100,
                          total_samples: int = 50000,
                          samples_per_file: int = 10000,
                          mcts_sims: int = 800,
                          device: str = "cuda",
                          save_dir: str = "buffer_data",
                          batch_size: int = 8,
                          num_processes: int = 8,
                          use_model: bool = False):
    """
    Thu thập dữ liệu từ self-play và lưu thành các file buffer riêng biệt.

    Args:
        model: Mô hình neural network
        games_per_batch: Số game mỗi lần collect
        total_samples: Tổng số samples cần thu thập
        samples_per_file: Số samples trong mỗi file buffer
        mcts_sims: Số lượt mô phỏng MCTS cho mỗi nước đi
        device: Thiết bị tính toán (cuda/cpu)
        save_dir: Thư mục lưu buffer files
        batch_size: Kích thước batch cho MCTS
        num_threads: Số luồng cho parallel MCTS
        use_model: Có sử dụng model để dự đoán nước đi hay không
    """
    os.makedirs(save_dir, exist_ok=True)

    # Buffer tạm thời để gom samples
    temp_buffer = ReplayBuffer(max_size=total_samples)
    current_part = 0

    pbar = tqdm(total=total_samples, desc="Collecting samples")
    initial_samples = len(temp_buffer)
    pbar.update(initial_samples)

    while len(temp_buffer) < total_samples:
        print(f"\n🚀 [INFO] Collecting games... Current samples: {len(temp_buffer)}/{total_samples}")
        # Thu thập batch games mới
        new_buffer = collect_self_play_games(
            model,
            num_games=games_per_batch,
            mcts_sims=mcts_sims,
            device=device,
            batch_size=batch_size,
            num_processes=num_processes,
            use_model=use_model
        )

        # Thêm vào buffer tạm thời
        samples_before = len(temp_buffer)
        for state, policy, value in zip(*new_buffer.sample_batch(len(new_buffer))):
            temp_buffer.add_game([(state, policy, value)])
        samples_added = len(temp_buffer) - samples_before

        # Update progress
        pbar.update(samples_added)

        # Kiểm tra xem có đủ samples để lưu file mới không
        while len(temp_buffer) >= (current_part + 1) * samples_per_file:
            # Lấy samples_per_file samples từ buffer
            states, policies, values = temp_buffer.sample_batch(samples_per_file)

            # Lưu thành file riêng
            save_path = os.path.join(save_dir, f"buffer_part_{current_part}.pt")
            torch.save({
                'states': states,
                'policies': policies,
                'values': values
            }, save_path)

            current_part += 1

    pbar.close()

    # Lưu phần samples còn lại nếu có
    remaining_samples = len(temp_buffer) - (current_part * samples_per_file)
    if remaining_samples > 0:
        states, policies, values = temp_buffer.sample_batch(remaining_samples)
        save_path = os.path.join(save_dir, f"buffer_part_{current_part}.pt")
        torch.save({
            'states': states,
            'policies': policies,
            'values': values
        }, save_path)

def load_buffer_part(file_path: str, device: str = "cuda"):
    """
    Load một phần của buffer từ file.
    """
    data = torch.load(file_path)
    return (
        torch.FloatTensor(data['states']).to(device),
        torch.FloatTensor(data['policies']).to(device),
        torch.FloatTensor(data['values']).to(device)
    )

def train_model_lazyload(model: ChessNet = None, optimizer = None, scheduler = None,
                         buffer_dir: str = "buffer_data", num_epochs: int = 10,
                         batch_size: int = 128, device: str = "cuda", save_dir: str = ".",
                         initial_lr: float = 0.001, min_lr: float = 1e-6):
    try:
        bess_loss = float('inf')

        if model is None:
            print("🔄 Creating new model...")
            model = ChessNet()

        if torch.cuda.device_count() > 1:
            print(f"🖥️ Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = torch.nn.DataParallel(model)

        model = model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2,
                min_lr=min_lr, verbose=True
            )

        criterion = AlphaLoss()

        buffer_files = sorted(glob.glob(os.path.join(buffer_dir, "buffer_part_*.pt")))
        if not buffer_files:
            raise FileNotFoundError(f"No buffer files found in {buffer_dir}")
        print(f"📂 Found {len(buffer_files)} buffer files.")

        no_improve_count = 0

        for part_idx, file_path in enumerate(buffer_files):
            print(f"\n🔄 Training on {file_path}...")

            try:
                data = torch.load(file_path, map_location='cpu')
                states = torch.FloatTensor(data['states'])
                policies = torch.FloatTensor(data['policies'])
                values = torch.FloatTensor(data['values'])

                del data  # Giải phóng bộ nhớ
                if device == 'cuda':
                    torch.cuda.empty_cache()

                dataset = TensorDataset(states, policies, values)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True if device == 'cuda' else False
                )

                part_best_loss = float('inf')

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0
                    pbar = tqdm(dataloader, desc=f"[Part {part_idx+1}/{len(buffer_files)}] Epoch {epoch+1}/{num_epochs}")

                    for batch in pbar:
                        state_batch, policy_batch, value_batch = [b.to(device) for b in batch]
                        mask = torch.zeros_like(policy_batch).to(device)
                        for i in range(len(policy_batch)):
                            mask[i] = (policy_batch[i] > 0).float()
                        
                        pred_policy, pred_value = model(state_batch, mask)
                        loss = criterion(
                            pred_value, value_batch.view(-1, 1),
                            pred_policy, policy_batch
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_loss += loss.item()
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                        )

                        del state_batch, policy_batch, value_batch, mask, pred_policy, pred_value, loss
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        
                    avg_loss = total_loss / len(dataloader)
                    print(f"✅ Done epoch {epoch+1} | Avg loss: {avg_loss:.4f}")
                    scheduler.step(avg_loss)

                    if avg_loss < part_best_loss:
                        part_best_loss = avg_loss
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            no_improve_count = 0
                            save_path = os.path.join(save_dir, "best_model.pth")
                            torch.save({
                                'epoch': epoch,
                                'part': part_idx,
                                'model_state_dict': model.module.state_dict()
                                    if isinstance(model, torch.nn.DataParallel)
                                    else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': best_loss,
                            }, save_path)
                            print(f"💾 Saved best model: {save_path}")
                        else:
                            no_improve_count += 1

                    if no_improve_count >= 5:
                        print("⚠️ Early stopping triggered!")
                        return model
                    
                    if device == "cuda":
                        torch.cuda.empty_cache()

                checkpoint_path = os.path.join(save_dir, f"model_after_part{part_idx}.pth")
                torch.save({
                    'part': part_idx,
                    'model_state_dict': model.module.state_dict()
                        if isinstance(model, torch.nn.DataParallel)
                        else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': part_best_loss,
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            except Exception as e:
                print(f"❌ Error processing part {part_idx}: {str(e)}")
                continue

            del states, policies, values, dataset, dataloader
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"\n🏁 Training completed. Best loss: {best_loss:.4f}")
        return model
    
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise
