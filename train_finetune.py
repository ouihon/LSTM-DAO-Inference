"""
Model Fine-tuning Training Script

This module handles fine-tuning of pre-trained magnetic component models.
It supports:
- Sliding window data preparation
- Multiple loss functions (MSE, RMSE, Energy)
- Layer freezing for transfer learning
- Real-time training status tracking
- PDF report generation

Author: Magnetic Design Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
import zipfile
import shutil
from datetime import datetime
from threading import Thread
import time

# Add models directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.modelA import Hybrid_PI_Model as ModelA
from models.modelB import Hybrid_PI_Model as ModelB
from models.modelC import Hybrid_PI_Model as ModelC
from models.modelD import Hybrid_PI_Model as ModelD
from models.modelE import Hybrid_PI_Model as ModelE

# ========== Model Class Mapping ==========

# Map model IDs to their corresponding model classes
MODEL_CLASSES = {
    'A': ModelA,
    'B': ModelB,
    'C': ModelC,
    'D': ModelD,
    'E': ModelE
}

# Device configuration: use GPU if available, otherwise CPU
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global dictionary to store training status for all tasks
# Structure: {task_id: {status, progress, metrics, ...}}
training_status = {}


# ========== Evaluation Metrics ==========

def compute_sequence_relative_error(H_pred, H_true):
    """
    Compute Sequence Relative Error (RMSE percentage)
    
    This metric measures the relative error between predicted and true H sequences.
    Formula: RMSE_pm = (RMS_diff / RMS_true) * 100%
    
    Args:
        H_pred: numpy array, shape (num_seq, T) - predicted H sequences
        H_true: numpy array, shape (num_seq, T) - true H sequences
    
    Returns:
        numpy array: RMSE percentage for each sequence [%]
    """
    assert H_pred.shape == H_true.shape, "H_pred and H_true must have same shape [num_seq, T]"
    
    diff = H_pred - H_true
    rms_diff = np.sqrt(np.mean(diff ** 2, axis=1))
    rms_true = np.sqrt(np.mean(H_true ** 2, axis=1))
    eps = 1e-12
    rms_pm = rms_diff / (rms_true + eps) * 100.0
    
    return rms_pm


def compute_energy_density_normalized_error(H_pred, H_true, B_true):
    """
    Compute Energy Density Normalized Relative Error (percentage)
    
    This metric measures the error in energy density calculation, which is important
    for magnetic component design as it relates to core loss.
    Formula: ene_error = ((W_pred - W_true) / W_true) * 100%
    where W = ∫ H dB (energy density)
    
    Args:
        H_pred: numpy array, shape (num_seq, T) - predicted H sequences
        H_true: numpy array, shape (num_seq, T) - true H sequences
        B_true: numpy array, shape (num_seq, T) - true B sequences
    
    Returns:
        numpy array: Energy error percentage for each sequence [%]
    """
    assert H_pred.shape == H_true.shape == B_true.shape, \
        f"Shape mismatch: H_pred{H_pred.shape}, H_true{H_true.shape}, B_true{B_true.shape}"
    
    # Calculate dB (change in B)
    dB = np.diff(B_true, axis=1)
    
    # Use midpoint values for H (trapezoidal integration)
    H_true_mid = 0.5 * (H_true[:, :-1] + H_true[:, 1:])
    H_pred_mid = 0.5 * (H_pred[:, :-1] + H_pred[:, 1:])
    
    # Calculate energy density: W = ∫ H dB ≈ sum(H_mid * dB)
    W_true = np.sum(H_true_mid * dB, axis=1)
    W_pred = np.sum(H_pred_mid * dB, axis=1)
    
    # Avoid division by zero
    eps = 1e-12
    denom = np.where(np.abs(W_true) < eps, np.sign(W_true) * eps + eps, W_true)
    ene_error = (W_pred - W_true) / denom * 100.0
    
    return ene_error


# ========== Data Preparation ==========

def create_sliding_windows(B, H, T, window_stride):
    """
    Create sliding windows from time series data
    
    Window size is fixed at 1000 time steps. This function creates overlapping
    windows with a specified stride for training data augmentation.
    
    Args:
        B: numpy array, shape (N, M) - N samples, each with M time steps
        H: numpy array, shape (N, M) - N samples, each with M time steps
        T: numpy array, shape (N, 1) - N samples, each with 1 temperature value
        window_stride: int - stride between windows (window size is fixed at 1000)
    
    Returns:
        list: List of tuples (B_window, H_window, T_value) for each window
    """
    N, M = B.shape
    WINDOW_SIZE = 1000  # Fixed window size
    windows = []
    
    print(f"[DEBUG] create_sliding_windows: N={N}, M={M}, window_stride={window_stride}, WINDOW_SIZE={WINDOW_SIZE}")
    
    for i in range(N):
        seq_B = B[i]  # Sequence of length M
        seq_H = H[i]  # Sequence of length M
        seq_T = T[i, 0]  # Scalar temperature value
        
        # If sequence length is less than window size, pad to 1000
        if M < WINDOW_SIZE:
            pad_B = np.pad(seq_B, (0, WINDOW_SIZE - M), mode='edge')
            pad_H = np.pad(seq_H, (0, WINDOW_SIZE - M), mode='edge')
            windows.append((pad_B, pad_H, seq_T))
            print(f"[DEBUG] Sample {i}: M={M} < {WINDOW_SIZE}, added 1 window after padding")
        else:
            # Sliding window: start from 0, step by window_stride, until can't fit 1000 points
            start = 0
            window_count = 0
            
            while start + WINDOW_SIZE <= M:
                end = start + WINDOW_SIZE
                windows.append((seq_B[start:end], seq_H[start:end], seq_T))
                window_count += 1
                start += window_stride
            
            # If there's remaining data, take the last 1000 points (avoid duplicates)
            if start < M and start + WINDOW_SIZE > M:
                # Only add if this segment is different from the last one
                last_start = M - WINDOW_SIZE
                if last_start >= 0 and last_start != start - window_stride:
                    windows.append((seq_B[-WINDOW_SIZE:], seq_H[-WINDOW_SIZE:], seq_T))
                    window_count += 1
            
            print(f"[DEBUG] Sample {i}: M={M}, created {window_count} windows")
    
    print(f"[DEBUG] Total windows created: {len(windows)}")
    return windows


# ========== Main Training Function ==========

def train_model(task_id, base_model, data_dir, epochs, learning_rate, batch_size, 
                validation_split, window_stride, loss_type, frozen_layers):
    """
    Main training function for model fine-tuning
    
    This function:
    1. Loads and validates training data
    2. Creates sliding windows
    3. Loads pre-trained base model
    4. Freezes specified layers
    5. Trains the model with specified hyperparameters
    6. Tracks training progress and metrics
    7. Saves trained weights and generates PDF report
    
    Args:
        task_id: str - Unique identifier for this training task
        base_model: str - Base model ID ('A', 'B', 'C', 'D', or 'E')
        data_dir: str - Directory containing H.csv, B.csv, T.csv
        epochs: int - Number of training epochs
        learning_rate: float - Learning rate for optimizer
        batch_size: int - Batch size for training
        validation_split: float - Fraction of data to use for validation (0.1-0.4)
        window_stride: int - Stride for sliding windows (window size fixed at 1000)
        loss_type: str - Loss function type ('MSE', 'RMSE', or 'energy')
        frozen_layers: int - Number of layers to freeze (0-3)
    """
    global training_status
    
    try:
        # Initialize training status
        training_status[task_id] = {
            'status': 'training',
            'progress': 0,
            'current_epoch': 0,
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_energy_loss': [],
            'val_energy_loss': [],
            'initial_rmse': None,
            'initial_energy_loss': None,
            'epoch_times': [],
            'message': 'Starting training...'
        }
        
        # ========== Load and Validate Data ==========
        
        H_data = np.loadtxt(os.path.join(data_dir, 'H.csv'), delimiter=',')
        T_data = np.loadtxt(os.path.join(data_dir, 'T.csv'), delimiter=',')
        B_data = np.loadtxt(os.path.join(data_dir, 'B.csv'), delimiter=',')
        
        print(f"[DEBUG] Raw data shapes: H={H_data.shape}, B={B_data.shape}, T={T_data.shape}")
        
        # Handle dimension reshaping (support 1D arrays)
        if len(H_data.shape) == 1:
            # 1D array: assume single sample, reshape to (1, M)
            H_data = H_data.reshape(1, -1)
            print(f"[DEBUG] H reshaped to: {H_data.shape}")
        if len(B_data.shape) == 1:
            B_data = B_data.reshape(1, -1)
            print(f"[DEBUG] B reshaped to: {B_data.shape}")
        if len(T_data.shape) == 1:
            # T is 1D array, reshape to (N, 1)
            T_data = T_data.reshape(-1, 1)
            print(f"[DEBUG] T reshaped to: {T_data.shape}")
        
        # Ensure all arrays are 2D
        if len(H_data.shape) != 2:
            raise ValueError(f"H data dimension error: {H_data.shape}, should be 2D")
        if len(B_data.shape) != 2:
            raise ValueError(f"B data dimension error: {B_data.shape}, should be 2D")
        if len(T_data.shape) != 2:
            raise ValueError(f"T data dimension error: {T_data.shape}, should be 2D")
        
        # Validate data dimensions
        N_H, M_H = H_data.shape
        N_B, M_B = B_data.shape
        N_T, M_T = T_data.shape
        
        print(f"[DEBUG] Processed data shapes: H=({N_H}, {M_H}), B=({N_B}, {M_B}), T=({N_T}, {M_T})")
        
        # Check if data might be transposed and try to fix
        if N_H != N_B or N_H != N_T:
            # Check if it's a transpose issue
            if M_H == N_B and M_B == N_H:
                print(f"[DEBUG] Detected transpose, transposing H and B")
                H_data = H_data.T
                B_data = B_data.T
                N_H, M_H = H_data.shape
                N_B, M_B = B_data.shape
                print(f"[DEBUG] After transpose: H=({N_H}, {M_H}), B=({N_B}, {M_B})")
        
        # Final validation checks
        if N_H != N_T or N_H != N_B:
            raise ValueError(f"Sample count mismatch: H={N_H}, T={N_T}, B={N_B}")
        
        if M_H != M_B:
            raise ValueError(f"Sequence length mismatch: H={M_H}, B={M_B}")
        
        if M_T != 1:
            raise ValueError(f"T should be (N, 1), got (N, {M_T})")
        
        # Check if data is empty
        if N_H == 0:
            raise ValueError(f"Data is empty: N_H={N_H}")
        
        print(f"[DEBUG] Final data shape: N={N_H}, M={M_H}, window_stride={window_stride}")
        
        # Create sliding windows (window size fixed at 1000, window_stride is step size)
        windows = create_sliding_windows(B_data, H_data, T_data, window_stride)
        
        print(f"[DEBUG] Total windows created: {len(windows)}")
        
        if len(windows) == 0:
            raise ValueError(
                f"Window creation resulted in 0 windows. "
                f"Check data format and window stride. "
                f"Data shapes: B={B_data.shape}, H={H_data.shape}, T={T_data.shape}, "
                f"window_stride={window_stride}"
            )
        
        # Split into training and validation sets
        split_idx = int(len(windows) * (1 - validation_split))
        train_windows = windows[:split_idx]
        val_windows = windows[split_idx:]
        
        # Update training status with window counts
        training_status[task_id]['total_windows'] = len(windows)
        training_status[task_id]['train_windows'] = len(train_windows)
        training_status[task_id]['val_windows'] = len(val_windows)
        
        # ========== Load Base Model ==========
        
        base_dir = '.'
        weights_path = os.path.join(base_dir, 'weights', f'{base_model}.sd')
        norm_params_path = os.path.join(base_dir, 'weights', f'Normalization_Params_{base_model}.json')
        
        # Load normalization parameters
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        
        # Create model instance
        ModelClass = MODEL_CLASSES.get(base_model, ModelA)
        
        # Set model parameters based on model type
        if base_model in ['A', 'B', 'C', 'D']:
            # Models A, B, C, D use: lstm_size=40, mlp_size=16, K=4
            model = ModelClass(lstm_size=40, mlp_size=16, num_layers=1, K=4).to(_device)
        else:
            # Other models (e.g., E) use default parameters
            model = ModelClass(lstm_size=40, mlp_size=32, num_layers=1, K=4).to(_device)
        
        # Load pre-trained weights
        checkpoint = torch.load(weights_path, map_location=_device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        # Freeze specified layers for transfer learning
        if frozen_layers > 0:
            # Freeze LSTM layer
            for param in model.lstm.parameters():
                param.requires_grad = False
            
            # If frozen_layers > 1, freeze first layer of pi_head
            if frozen_layers > 1:
                for param in model.pi_head[0].parameters():
                    param.requires_grad = False
        
        # ========== Setup Optimizer and Loss Function ==========
        
        # Optimizer: only optimize parameters that require gradients
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        # Loss function selection
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'RMSE':
            criterion = lambda pred, true: torch.sqrt(nn.MSELoss()(pred, true))
        elif loss_type == 'energy':
            # Custom energy loss function
            def energy_loss(pred, true, B):
                dB = B[:, 1:] - B[:, :-1]
                H_true_mid = 0.5 * (true[:, :-1] + true[:, 1:])
                H_pred_mid = 0.5 * (pred[:, :-1] + pred[:, 1:])
                W_true = torch.sum(H_true_mid * dB, dim=1)
                W_pred = torch.sum(H_pred_mid * dB, dim=1)
                eps = 1e-12
                return torch.mean(((W_pred - W_true) / (torch.abs(W_true) + eps)) ** 2)
            criterion = energy_loss
        else:
            criterion = nn.MSELoss()
        
        # ========== Compute Initial Metrics ==========
        
        # Calculate baseline metrics before training (using validation set)
        model.eval()
        with torch.no_grad():
            sample_windows = val_windows[:min(100, len(val_windows))]
            H_pred_list = []
            H_true_list = []
            B_true_list = []
            
            for win_B, win_H, win_T in sample_windows:
                # Normalize data
                B_norm = (win_B - norm_params['mean_B']) / (norm_params['std_B'] + 1e-12)
                H_norm = (win_H - norm_params['mean_H']) / (norm_params['std_H'] + 1e-12)
                T_norm = (win_T - norm_params['mean_T']) / (norm_params['std_T'] + 1e-12)
                
                # Convert to tensors
                B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                H_tensor = torch.tensor(H_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                T_tensor = torch.tensor([T_norm], dtype=torch.float32).unsqueeze(0).to(_device)
                
                # Calculate dB and ddB (derivatives)
                B_prep = 2 * B_tensor[:, 0:1] - B_tensor[:, 1:2]
                dB = torch.cat([B_prep, B_tensor], dim=1)[:, 1:] - B_tensor
                dB[dB == 0] = 1e-4
                
                dB_prep = 2 * dB[:, 0:1] - dB[:, 1:2]
                ddB = torch.cat([dB_prep, dB], dim=1)[:, 1:] - dB
                ddB[ddB == 0] = 1e-4
                
                # H mask: first value is known, rest are unknown
                H_value = H_tensor.clone()
                H_value[:, 1:] = 0
                H_mask = torch.zeros_like(H_tensor)
                H_mask[:, 0] = 1
                
                split_point = torch.tensor([1], dtype=torch.long).to(_device)
                
                # Forward pass
                H_pred = model(B_tensor, dB, ddB, H_value, H_mask, T_tensor, split_point)
                
                # Denormalize for metric calculation
                H_pred_denorm = (H_pred.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                H_true_denorm = (H_tensor.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                B_true_denorm = (B_tensor.cpu().numpy()[0] * norm_params['std_B'] + norm_params['mean_B'])
                
                H_pred_list.append(H_pred_denorm)
                H_true_list.append(H_true_denorm)
                B_true_list.append(B_true_denorm)
            
            # Calculate initial metrics
            H_pred_array = np.array(H_pred_list)
            H_true_array = np.array(H_true_list)
            B_true_array = np.array(B_true_list)
            
            initial_rmse_values = compute_sequence_relative_error(H_pred_array, H_true_array)
            initial_energy_values = compute_energy_density_normalized_error(H_pred_array, H_true_array, B_true_array)
            
            training_status[task_id]['initial_rmse'] = float(np.mean(initial_rmse_values))
            training_status[task_id]['initial_energy_loss'] = float(np.mean(np.abs(initial_energy_values)))
        
        # ========== Training Loop ==========
        
        model.train()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            training_status[task_id]['current_epoch'] = epoch + 1
            training_status[task_id]['progress'] = int((epoch + 1) / epochs * 100)
            training_status[task_id]['message'] = f'Training Epoch {epoch+1}/{epochs}'
            
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            # ========== Training Phase ==========
            
            model.train()
            np.random.shuffle(train_windows)  # Shuffle for better training
            
            # Calculate number of batches (ensure at least 1)
            num_batches = max(1, len(train_windows) // batch_size)
            actual_batches = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_windows))
                batch_windows = train_windows[start_idx:end_idx]
                
                if len(batch_windows) == 0:
                    break
                
                # Prepare batch data
                batch_B = []
                batch_H = []
                batch_T = []
                
                for win_B, win_H, win_T in batch_windows:
                    # Normalize
                    B_norm = (win_B - norm_params['mean_B']) / (norm_params['std_B'] + 1e-12)
                    H_norm = (win_H - norm_params['mean_H']) / (norm_params['std_H'] + 1e-12)
                    T_norm = (win_T - norm_params['mean_T']) / (norm_params['std_T'] + 1e-12)
                    
                    batch_B.append(B_norm)
                    batch_H.append(H_norm)
                    batch_T.append(T_norm)
                
                # Convert to tensors
                batch_B = torch.tensor(np.array(batch_B), dtype=torch.float32).to(_device)
                batch_H = torch.tensor(np.array(batch_H), dtype=torch.float32).to(_device)
                batch_T = torch.tensor(np.array(batch_T), dtype=torch.float32).unsqueeze(1).to(_device)
                
                # Calculate dB and ddB
                B_prep = 2 * batch_B[:, 0:1] - batch_B[:, 1:2]
                dB = torch.cat([B_prep, batch_B], dim=1)[:, 1:] - batch_B
                dB[dB == 0] = 1e-4
                
                dB_prep = 2 * dB[:, 0:1] - dB[:, 1:2]
                ddB = torch.cat([dB_prep, dB], dim=1)[:, 1:] - dB
                ddB[ddB == 0] = 1e-4
                
                # H mask: first value is known, rest are unknown
                H_value = batch_H.clone()
                H_value[:, 1:] = 0
                H_mask = torch.zeros_like(batch_H)
                H_mask[:, 0] = 1
                
                split_point = torch.tensor([1] * batch_B.shape[0], dtype=torch.long).to(_device)
                
                # Forward pass
                optimizer.zero_grad()
                H_pred = model(batch_B, dB, ddB, H_value, H_mask, batch_T, split_point)
                
                # Calculate loss
                if loss_type == 'energy':
                    loss = criterion(H_pred, batch_H, batch_B)
                else:
                    loss = criterion(H_pred, batch_H)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                actual_batches += 1
            
            # Average training loss
            if actual_batches > 0:
                epoch_train_loss /= actual_batches
            else:
                epoch_train_loss = 0.0
            
            # ========== Validation Phase ==========
            
            model.eval()
            with torch.no_grad():
                val_batch_size = min(batch_size, len(val_windows))
                num_val_batches = max(1, len(val_windows) // val_batch_size) if val_batch_size > 0 else 0
                actual_val_batches = 0
                
                for batch_idx in range(num_val_batches):
                    start_idx = batch_idx * val_batch_size
                    end_idx = min((batch_idx + 1) * val_batch_size, len(val_windows))
                    batch_windows = val_windows[start_idx:end_idx]
                    
                    if len(batch_windows) == 0:
                        break
                    
                    batch_B = []
                    batch_H = []
                    batch_T = []
                    
                    for win_B, win_H, win_T in batch_windows:
                        B_norm = (win_B - norm_params['mean_B']) / (norm_params['std_B'] + 1e-12)
                        H_norm = (win_H - norm_params['mean_H']) / (norm_params['std_H'] + 1e-12)
                        T_norm = (win_T - norm_params['mean_T']) / (norm_params['std_T'] + 1e-12)
                        
                        batch_B.append(B_norm)
                        batch_H.append(H_norm)
                        batch_T.append(T_norm)
                    
                    batch_B = torch.tensor(np.array(batch_B), dtype=torch.float32).to(_device)
                    batch_H = torch.tensor(np.array(batch_H), dtype=torch.float32).to(_device)
                    batch_T = torch.tensor(np.array(batch_T), dtype=torch.float32).unsqueeze(1).to(_device)
                    
                    # Calculate dB and ddB
                    B_prep = 2 * batch_B[:, 0:1] - batch_B[:, 1:2]
                    dB = torch.cat([B_prep, batch_B], dim=1)[:, 1:] - batch_B
                    dB[dB == 0] = 1e-4
                    
                    dB_prep = 2 * dB[:, 0:1] - dB[:, 1:2]
                    ddB = torch.cat([dB_prep, dB], dim=1)[:, 1:] - dB
                    ddB[ddB == 0] = 1e-4
                    
                    H_value = batch_H.clone()
                    H_value[:, 1:] = 0
                    H_mask = torch.zeros_like(batch_H)
                    H_mask[:, 0] = 1
                    
                    split_point = torch.tensor([1] * batch_B.shape[0], dtype=torch.long).to(_device)
                    
                    H_pred = model(batch_B, dB, ddB, H_value, H_mask, batch_T, split_point)
                    
                    if loss_type == 'energy':
                        loss = criterion(H_pred, batch_H, batch_B)
                    else:
                        loss = criterion(H_pred, batch_H)
                    
                    epoch_val_loss += loss.item()
                    actual_val_batches += 1
                
                # Average validation loss
                if actual_val_batches > 0:
                    epoch_val_loss /= actual_val_batches
                else:
                    epoch_val_loss = 0.0
            
            # ========== Calculate Metrics ==========
            
            # Calculate RMSE and Energy Loss for both training and validation sets
            model.eval()
            with torch.no_grad():
                # Validation set metrics
                val_sample_windows = val_windows[:min(100, len(val_windows))]
                H_pred_val_list = []
                H_true_val_list = []
                B_true_val_list = []
                
                for win_B, win_H, win_T in val_sample_windows:
                    B_norm = (win_B - norm_params['mean_B']) / (norm_params['std_B'] + 1e-12)
                    H_norm = (win_H - norm_params['mean_H']) / (norm_params['std_H'] + 1e-12)
                    T_norm = (win_T - norm_params['mean_T']) / (norm_params['std_T'] + 1e-12)
                    
                    B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                    H_tensor = torch.tensor(H_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                    T_tensor = torch.tensor([T_norm], dtype=torch.float32).unsqueeze(0).to(_device)
                    
                    B_prep = 2 * B_tensor[:, 0:1] - B_tensor[:, 1:2]
                    dB = torch.cat([B_prep, B_tensor], dim=1)[:, 1:] - B_tensor
                    dB[dB == 0] = 1e-4
                    
                    dB_prep = 2 * dB[:, 0:1] - dB[:, 1:2]
                    ddB = torch.cat([dB_prep, dB], dim=1)[:, 1:] - dB
                    ddB[ddB == 0] = 1e-4
                    
                    H_value = H_tensor.clone()
                    H_value[:, 1:] = 0
                    H_mask = torch.zeros_like(H_tensor)
                    H_mask[:, 0] = 1
                    
                    split_point = torch.tensor([1], dtype=torch.long).to(_device)
                    
                    H_pred = model(B_tensor, dB, ddB, H_value, H_mask, T_tensor, split_point)
                    
                    H_pred_denorm = (H_pred.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                    H_true_denorm = (H_tensor.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                    B_true_denorm = (B_tensor.cpu().numpy()[0] * norm_params['std_B'] + norm_params['mean_B'])
                    
                    H_pred_val_list.append(H_pred_denorm)
                    H_true_val_list.append(H_true_denorm)
                    B_true_val_list.append(B_true_denorm)
                
                H_pred_val_array = np.array(H_pred_val_list)
                H_true_val_array = np.array(H_true_val_list)
                B_true_val_array = np.array(B_true_val_list)
                
                val_rmse_values = compute_sequence_relative_error(H_pred_val_array, H_true_val_array)
                val_energy_values = compute_energy_density_normalized_error(H_pred_val_array, H_true_val_array, B_true_val_array)
                
                avg_val_rmse = np.mean(val_rmse_values)
                avg_val_energy = np.mean(np.abs(val_energy_values))
                
                # Training set metrics
                train_sample_windows = train_windows[:min(100, len(train_windows))]
                H_pred_train_list = []
                H_true_train_list = []
                B_true_train_list = []
                
                for win_B, win_H, win_T in train_sample_windows:
                    B_norm = (win_B - norm_params['mean_B']) / (norm_params['std_B'] + 1e-12)
                    H_norm = (win_H - norm_params['mean_H']) / (norm_params['std_H'] + 1e-12)
                    T_norm = (win_T - norm_params['mean_T']) / (norm_params['std_T'] + 1e-12)
                    
                    B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                    H_tensor = torch.tensor(H_norm, dtype=torch.float32).unsqueeze(0).to(_device)
                    T_tensor = torch.tensor([T_norm], dtype=torch.float32).unsqueeze(0).to(_device)
                    
                    B_prep = 2 * B_tensor[:, 0:1] - B_tensor[:, 1:2]
                    dB = torch.cat([B_prep, B_tensor], dim=1)[:, 1:] - B_tensor
                    dB[dB == 0] = 1e-4
                    
                    dB_prep = 2 * dB[:, 0:1] - dB[:, 1:2]
                    ddB = torch.cat([dB_prep, dB], dim=1)[:, 1:] - dB
                    ddB[ddB == 0] = 1e-4
                    
                    H_value = H_tensor.clone()
                    H_value[:, 1:] = 0
                    H_mask = torch.zeros_like(H_tensor)
                    H_mask[:, 0] = 1
                    
                    split_point = torch.tensor([1], dtype=torch.long).to(_device)
                    
                    H_pred = model(B_tensor, dB, ddB, H_value, H_mask, T_tensor, split_point)
                    
                    H_pred_denorm = (H_pred.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                    H_true_denorm = (H_tensor.cpu().numpy()[0] * norm_params['std_H'] + norm_params['mean_H'])
                    B_true_denorm = (B_tensor.cpu().numpy()[0] * norm_params['std_B'] + norm_params['mean_B'])
                    
                    H_pred_train_list.append(H_pred_denorm)
                    H_true_train_list.append(H_true_denorm)
                    B_true_train_list.append(B_true_denorm)
                
                H_pred_train_array = np.array(H_pred_train_list)
                H_true_train_array = np.array(H_true_train_list)
                B_true_train_array = np.array(B_true_train_list)
                
                train_rmse_values = compute_sequence_relative_error(H_pred_train_array, H_true_train_array)
                train_energy_values = compute_energy_density_normalized_error(H_pred_train_array, H_true_train_array, B_true_train_array)
                
                avg_train_rmse = np.mean(train_rmse_values)
                avg_train_energy = np.mean(np.abs(train_energy_values))
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            training_status[task_id]['epoch_times'].append(float(epoch_time))
            
            # Update training status with metrics
            training_status[task_id]['train_loss'].append(float(epoch_train_loss))
            training_status[task_id]['val_loss'].append(float(epoch_val_loss))
            training_status[task_id]['train_rmse'].append(float(avg_train_rmse))
            training_status[task_id]['val_rmse'].append(float(avg_val_rmse))
            training_status[task_id]['train_energy_loss'].append(float(avg_train_energy))
            training_status[task_id]['val_energy_loss'].append(float(avg_val_energy))
        
        # ========== Save Model and Generate Report ==========
        
        output_dir = os.path.join('.', 'uploads', task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trained weights
        weights_file = os.path.join(output_dir, 'model_weights.sd')
        torch.save(model.state_dict(), weights_file)
        
        # Generate training report PDF
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_file = os.path.join(output_dir, 'training_report.pdf')
            with PdfPages(pdf_file) as pdf:
                # Loss curves
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(training_status[task_id]['train_loss'], label='Training Loss')
                ax.plot(training_status[task_id]['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True)
                pdf.savefig(fig)
                plt.close()
                
                # RMSE curves
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(training_status[task_id]['train_rmse'], label='Training RMSE (%)')
                ax.plot(training_status[task_id]['val_rmse'], label='Validation RMSE (%)')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('RMSE (%)')
                ax.set_title('RMSE Progress')
                ax.legend()
                ax.grid(True)
                pdf.savefig(fig)
                plt.close()
                
                # Energy Loss curves
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(training_status[task_id]['train_energy_loss'], label='Training Energy Loss (%)')
                ax.plot(training_status[task_id]['val_energy_loss'], label='Validation Energy Loss (%)')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Energy Loss (%)')
                ax.set_title('Energy Loss Progress')
                ax.legend()
                ax.grid(True)
                pdf.savefig(fig)
                plt.close()
                
                # Training Summary
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                summary_text = f"""
Training Summary
================

Task ID: {task_id}
Base Model: {base_model}
Epochs: {epochs}
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Window Stride: {window_stride} (Fixed window size: 1000)
Loss Function: {loss_type}
Frozen Layers: {frozen_layers}

Final Metrics:
- Training Loss: {training_status[task_id]['train_loss'][-1]:.6f}
- Validation Loss: {training_status[task_id]['val_loss'][-1]:.6f}
- Training RMSE: {training_status[task_id]['train_rmse'][-1]:.2f}%
- Validation RMSE: {training_status[task_id]['val_rmse'][-1]:.2f}%
- Training Energy Loss: {training_status[task_id]['train_energy_loss'][-1]:.2f}%
- Validation Energy Loss: {training_status[task_id]['val_energy_loss'][-1]:.2f}%

Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', 
                       family='monospace')
                pdf.savefig(fig)
                plt.close()
        except Exception as e:
            print(f"Failed to generate PDF report: {e}")
        
        # Mark training as completed
        training_status[task_id]['status'] = 'completed'
        training_status[task_id]['message'] = 'Training completed'
        
    except Exception as e:
        # Mark training as failed
        training_status[task_id]['status'] = 'failed'
        training_status[task_id]['message'] = f'Training failed: {str(e)}'
        import traceback
        traceback.print_exc()
