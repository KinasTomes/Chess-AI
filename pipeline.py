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

from core.mcts import MCTS
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

def self_play(model: ChessNet, num_games: int, replay_buffer: ReplayBuffer) -> None:
    """
    Self-play function to generate training data.
    
    Args:
        model (ChessNet): The chess model.
        num_games (int): Number of games to play.
        replay_buffer (ReplayBuffer): Thread-safe replay buffer for storing game data.
    """
    try:
        for game_idx in range(num_games):
            env = ChessEnv()
            env.reset()
            mcts = MCTS(
                neural_net=model,
                converter=env.converter,
                env=env,
                simulations=100,
                max_depth=5,
                device='cuda',
                num_processes=4,
                use_model=True,
                temperature=1.0
            )

            game_history = []
            move_count = 0

            while not env._is_game_over():
                pi = mcts.run(env.board)

                game_history.append({
                    'state': env._observation(),
                    'policy': pi,
                    'player': env.to_play
                })

                valid_move = env.legal_actions
                pi_valid = pi * valid_move

                if move_count < 10:
                    pi_valid = pi_valid / (np.sum(pi_valid) + 1e-8)
                    action = np.random.choice(len(pi), p=pi_valid)
                else:
                    action = np.argmax(pi_valid)

                selected_move = env.converter.index_to_move(action)
                env.step(selected_move)
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

            # Clean up resources after each game
            mcts.cleanup()
            del mcts
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
        NUM_PROCESSES = 2
        best_loss = float('inf')

        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")

            # Create thread-safe replay buffer
            replay_buffer = ReplayBuffer()
            replay_buffer_lock = mp.Lock()

            mcts_model = []
            for gpu_id in range(num_gpus):
                model = get_model_for_pipeline(model_dir)
                model = model.to(f'cuda:{gpu_id}')  # Explicitly move model to specific GPU
                model.share_memory()
                model.eval()
                mcts_model.append(model)

            print(f"Starting self-play with {NUM_PROCESSES} processes...")    
            mcts_processes = []
            for i in range(NUM_PROCESSES):
                gpu_id = get_gpu_id(i, num_gpus)
                p = mp.Process(target=self_play, args=(mcts_model[gpu_id], num_games_per_iteration // 2, replay_buffer))
                mcts_processes.append(p)
                p.start()
            
            for p in mcts_processes:
                p.join()

            del mcts_model
            torch.cuda.empty_cache()

            states, policies, values = replay_buffer.take_all()
            del replay_buffer

            # Move all data to the same device
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            # Prepare data loaders with validation split
            train_loader, val_loader = prepare_data_loaders(states, policies, values, batch_size)

            train_model = get_model_for_pipeline(model_dir)
            train_model = nn.DataParallel(train_model)
            train_model = train_model.to(device)  # Move model to device before training
            train_model.train()

            optimizer = optim.SGD(train_model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, min_lr=min_lr)
            criterion = AlphaLoss()

            for epoch in range(num_epoch):
                train_model.train()
                total_loss = 0.0
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}")
                for batch in pbar:
                    states_batch, policies_batch, values_batch = [b.to(device) for b in batch]
                    mask = torch.zeros_like(policies_batch).to(device)
                    for i in range(len(policies_batch)):
                        mask[i] = (policies_batch[i] > 0).float()

                    pred_policies, pred_values = train_model(states_batch, mask)
                    loss = criterion(pred_values, values_batch.view(-1, 1), pred_policies, policies_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                    )

                    del states_batch, policies_batch, values_batch 
                    del mask, pred_policies, pred_values, loss
                    torch.cuda.empty_cache()

                avg_train_loss = total_loss / len(train_loader)
                
                # Validate model
                avg_val_loss = validate_model(train_model, val_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{num_epoch} - "
                      f"Train Loss: {avg_train_loss:.4f} - "
                      f"Val Loss: {avg_val_loss:.4f}")
                
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    print(f"Saving best model with validation loss: {best_loss:.4f}")
                    torch.save({
                        'model_state_dict': train_model.state_dict(),
                        'loss': best_loss,
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(model_dir, 'best_model.pth'))
        
            del states, policies, values, train_loader, val_loader
            torch.cuda.empty_cache()

        print(f"Training completed. Best validation loss: {best_loss:.4f}")

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        torch.cuda.empty_cache()