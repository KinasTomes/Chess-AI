import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from typing import Tuple, List, Dict

from core.model import ChessNet, AlphaLoss
from training.utils import load_model

def get_model_for_training(model_dir: str) -> ChessNet:
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

def prepare_dataloader_from_file(file_path: str, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Prepare a DataLoader from a single file.
    
    Args:
        file_path (str): Path to the data file
        batch_size (int): Batch size for the data loader
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: DataLoader for the data in the file
    """
    data = torch.load(file_path)
    
    # Convert numpy arrays to tensors if needed
    states = torch.FloatTensor(data['states']) if isinstance(data['states'], np.ndarray) else data['states']
    policies = torch.FloatTensor(data['policies']) if isinstance(data['policies'], np.ndarray) else data['policies']
    values = torch.FloatTensor(data['values']) if isinstance(data['values'], np.ndarray) else data['values']
    
    dataset = TensorDataset(states, policies, values)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader

def prepare_validation_data(val_files: List[str], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DataLoader]:
    """
    Prepare validation data by loading all validation files.
    
    Args:
        val_files (List[str]): List of validation file paths
        batch_size (int): Batch size for the validation data loader
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DataLoader]: Combined validation data tensors and DataLoader
    """
    all_states = []
    all_policies = []
    all_values = []
    
    for file_path in val_files:
        data = torch.load(file_path)
        
        # Convert numpy arrays to tensors if needed
        states = torch.FloatTensor(data['states']) if isinstance(data['states'], np.ndarray) else data['states']
        policies = torch.FloatTensor(data['policies']) if isinstance(data['policies'], np.ndarray) else data['policies']
        values = torch.FloatTensor(data['values']) if isinstance(data['values'], np.ndarray) else data['values']
        
        all_states.append(states)
        all_policies.append(policies)
        all_values.append(values)
    
    # Concatenate data from all validation files
    states = torch.cat(all_states, dim=0)
    policies = torch.cat(all_policies, dim=0)
    values = torch.cat(all_values, dim=0)
    
    dataset = TensorDataset(states, policies, values)
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return states, policies, values, val_loader

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
            mask = (policies_batch > 0).float()
            
            pred_policies, pred_values = model(states_batch, mask)
            loss = criterion(pred_values, values_batch.view(-1, 1), pred_policies, policies_batch)
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)

def train_from_saved_data(model_dir: str, data_dir: str, num_epoch: int = 10, 
                         batch_size: int = 128, device: str = 'cuda', 
                         initial_lr: float = 0.2, min_lr: float = 0.0002,
                         val_split: float = 0.1):
    """
    Train the model using saved data from .pt files.
    
    Args:
        model_dir (str): Directory containing the model checkpoint.
        data_dir (str): Directory containing the saved data (.pt files).
        num_epoch (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to run training on ('cuda' or 'cpu').
        initial_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        val_split (float): Proportion of files to use for validation.
    """
    # Check if device is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    # Load model
    model = get_model_for_training(model_dir)
    model = model.to(device)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all data file paths
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in data directory: {data_dir}")
    
    # Shuffle and split files into training and validation sets
    random.shuffle(data_files)
    val_size = max(1, int(len(data_files) * val_split))
    train_files = data_files[val_size:]
    val_files = data_files[:val_size]
    
    print(f"Found {len(data_files)} data files: {len(train_files)} for training, {len(val_files)} for validation")
    
    # Prepare validation data (load all validation files at once)
    print("Loading validation data...")
    _, _, _, val_loader = prepare_validation_data(val_files, batch_size)
    
    # Training setup
    criterion = AlphaLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=min_lr)
    
    # Track best model
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Shuffle training files at the beginning of each epoch
        random.shuffle(train_files)
        
        # Process one file at a time in each epoch
        for file_idx, file_path in enumerate(train_files):
            print(f"Epoch {epoch + 1}/{num_epoch}, File {file_idx + 1}/{len(train_files)}: {os.path.basename(file_path)}")
            
            # Load data from a single file
            train_loader = prepare_dataloader_from_file(file_path, batch_size, shuffle=True)
            
            # Train on this file
            file_loss = 0.0
            file_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Training"):
                states_batch, policies_batch, values_batch = [b.to(device) for b in batch]
                mask = (policies_batch > 0).float()
                
                optimizer.zero_grad()
                pred_policies, pred_values = model(states_batch, mask)
                loss = criterion(pred_values, values_batch.view(-1, 1), pred_policies, policies_batch)
                loss.backward()
                optimizer.step()
                
                file_loss += loss.item()
                file_batches += 1
            
            avg_file_loss = file_loss / file_batches if file_batches > 0 else 0
            print(f"File average loss: {avg_file_loss:.4f}")
            
            total_loss += file_loss
            total_batches += file_batches
        
        # Calculate average loss over all files in this epoch
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        
        # Validate the model
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
    
    print("Training completed successfully.")

if __name__ == "__main__":
    # Example usage
    train_from_saved_data(
        model_dir="model_checkpoint",
        data_dir="saved_data",
        num_epoch=10,
        batch_size=128,
        device='cuda',
        initial_lr=0.2,
        min_lr=0.0002,
        val_split=0.1
    ) 