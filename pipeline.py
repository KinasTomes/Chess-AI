import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import random_split
from typing import Tuple, List, Dict
import traceback

from core.mcts_v2_fake import MCTSNode, uct_search, parallel_uct_search
from core.model import ChessNet, AlphaLoss
from core.chess_base_v2 import ChessEnv
from training.replay_buffer import ReplayBuffer
from training.utils import load_model
from torch.utils.data import Dataset, DataLoader, TensorDataset

def get_gpu_id(process_id: int, num_gpus: int) -> int:
    """
    Get the GPU ID for a given process ID.
    
    Args:
        process_id (int): The process ID.
        num_gpu (int): The number of GPUs available.
        
    Returns:
        int: The GPU ID for the process.
    """
    return process_id % num_gpus

def self_play(model: ChessNet, num_games: int, max_move_limit: int, replay_buffer: ReplayBuffer) -> None:
    """
    Self-play function to generate training data.
    
    Args:
        model (ChessNet): The chess model.
        num_games (int): Number of games to play.
        replay_buffer (ReplayBuffer): Thread-safe replay buffer for storing game data.
    """
    try:
        device = next(model.parameters()).device

        def _parallel_evaluate(obs_batch: List[np.ndarray], mask_batch: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
            """
            Batched evaluation using a neural network model.

            Args:
                obs_batch: List of observations (np.ndarray), shape [C, 8, 8] per item
                mask_batch: List of legal move masks (np.ndarray), shape [A] per item
                model: The model to evaluate, returns (policy, value)
                device: 'cuda' or 'cpu'

            Returns:
                policies: np.ndarray of shape [B, A]
                values: List[float] of length B
            """
            with torch.no_grad():
                # Handle both single input and batch input
                if not isinstance(obs_batch, list):
                    obs_batch = [obs_batch]
                if not isinstance(mask_batch, list):
                    mask_batch = [mask_batch]

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

                # If single input, return single policy but keep values as list
                if len(obs_batch) == 1:
                    return policies[0], values[0]

            return policies, values
            
        for game_idx in range(num_games):
            env = ChessEnv()
            env.reset()
            root_node = None
            game_history = []
            move_count = 0

            while not env._is_game_over() and move_count < max_move_limit:
                # Use parallel MCTS for better performance
                move, pi, root_value, best_child_value, next_root = parallel_uct_search(
                    env=env,
                    eval_func=_parallel_evaluate,
                    root_node=root_node,
                    c_puct_base=19652,
                    c_puct_init=1.25,
                    num_sims=200,
                    num_parallel=8,
                    root_noise=True if move_count < 10 else False,
                    warm_up=True if move_count < 10 else False,
                    deterministic=False
                )

                game_history.append({
                    'state': env._observation(),
                    'policy': pi,
                    'player': env.to_play
                })

                selected_move = env.converter.index_to_move(move)
                env.step(selected_move)
                root_node = next_root
                move_count += 1

            if env.board.result() == '1-0':
                game_result = 1
            elif env.board.result() == '0-1':
                game_result = -1
            else:
                game_result = 0

            game_data = []
            for history in game_history:
                value = game_result if history['player'] else -game_result
                game_data.append((history['state'], history['policy'], value))

            replay_buffer.add_game(game_data)

            if game_idx % 10 == 0:
                print(f"Process {mp.current_process().name}: Completed {game_idx}/{num_games} games")

            # Clean up resources
            del env
            torch.cuda.empty_cache()

        print("Self play complete.")

    except Exception as e:
        print(f"Error in self_play: {str(e)}\n{traceback.format_exc()}")
        raise

def get_model_for_pipeline(model_dir: str) -> ChessNet:
    """
    Load the model from the specified directory into CPU.
    
    Args:
        model_dir (str): Directory containing the model checkpoint.
        
    Returns:
        ChessNet: The loaded chess model if exists, otherwise a new model is created.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_file = [file for file in os.listdir(model_dir) if file.endswith('.pth')]
    model = ChessNet()
    if len(model_file) == 0:
        print(f"No model found in: {model_dir}, creating a new model.")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': 10
        }, os.path.join(model_dir, 'best_model.pth'))
        return model
    
    return load_model(os.path.join(model_dir, model_file[0]), model)

def prepare_data_loaders(states: torch.Tensor, policies: torch.Tensor, values: torch.Tensor, 
                        batch_size: int, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation data loaders.
    
    Args:
        states (torch.Tensor): Game states
        policies (torch.Tensor): Policy targets
        values (torch.Tensor): Value targets
        batch_size (int): Batch size for training
        val_split (float): Validation split ratio
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders
    """
    dataset = TensorDataset(states, policies, values)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: str) -> float:
    """
    Validate the model on validation set.
    
    Args:
        model (nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (str): Device to run validation on
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            states_batch, policies_batch, values_batch = [b.to(device) for b in batch]
            mask = torch.zeros_like(policies_batch).to(device)
            for i in range(len(policies_batch)):
                mask[i] = (policies_batch[i] > 0).float()
            
            pred_policies, pred_values = model(states_batch, mask)
            loss = criterion(pred_values, values_batch.view(-1, 1), pred_policies, policies_batch)
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)

def training_pipeline(num_iterations: int, num_games_per_iteration: int, model_dir: str, 
                     num_epoch: int = 10, batch_size: int = 128, device: str = 'cuda', 
                     initial_lr: float = 0.2, min_lr: float = 0.0002):
    """
    Main training pipeline function.
    """
    try:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        mp.set_start_method('spawn', force=True)
        NUM_PROCESSES = 4 * num_gpus
        best_loss = float('inf')

        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")

            # Create thread-safe replay buffer
            replay_buffer = ReplayBuffer()
            replay_buffer_lock = mp.Lock()

            mcts_model = []
            for gpu_id in range(num_gpus):
                model = get_model_for_pipeline(model_dir)
                model = model.to(f'cuda:{gpu_id}')
                model.share_memory()
                model.eval()
                mcts_model.append(model)

            print(f"Starting self-play with {NUM_PROCESSES} processes...")    
            mcts_processes = []
            for i in range(NUM_PROCESSES):
                gpu_id = get_gpu_id(i, num_gpus)
                p = mp.Process(target=self_play, args=(mcts_model[gpu_id], num_games_per_iteration // 2, 200, replay_buffer))
                mcts_processes.append(p)
                p.start()
            
            for p in mcts_processes:
                p.join()

            del mcts_model
            torch.cuda.empty_cache()

            states, policies, values = replay_buffer.take_all()
            
            if len(states) == 0:
                print("No data collected in this iteration. Skipping training.")
                continue

            # Convert to tensors
            states = torch.FloatTensor(states)
            policies = torch.FloatTensor(policies)
            values = torch.FloatTensor(values)

            # Prepare data loaders
            train_loader, val_loader = prepare_data_loaders(states, policies, values, batch_size)

            # Training setup
            model = get_model_for_pipeline(model_dir)
            model = model.to(device)
            criterion = AlphaLoss()
            optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=min_lr)

            # Training loop
            for epoch in range(num_epoch):
                model.train()
                total_loss = 0.0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"):
                    states_batch, policies_batch, values_batch = [b.to(device) for b in batch]
                    mask = torch.zeros_like(policies_batch).to(device)
                    for i in range(len(policies_batch)):
                        mask[i] = (policies_batch[i] > 0).float()
                    
                    optimizer.zero_grad()
                    pred_policies, pred_values = model(states_batch, mask)
                    loss = criterion(pred_values, values_batch.view(-1, 1), pred_policies, policies_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                val_loss = validate_model(model, val_loader, criterion, device)
                
                print(f"Epoch {epoch + 1}/{num_epoch} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                scheduler.step()

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'loss': val_loss
                }, os.path.join(model_dir, 'best_model.pth'))
                print(f"New best model saved with validation loss: {val_loss:.4f}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        print("Training pipeline completed successfully.")

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    training_pipeline(
        num_iterations=100,
        num_games_per_iteration=100,
        model_dir="model_checkpoint",
        num_epoch=10,
        batch_size=128,
        device='cuda',
        initial_lr=0.2,
        min_lr=0.0002
    )