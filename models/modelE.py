"""
Model E Definition and Prediction Functions

This module contains the model architecture and prediction logic for Model E.
It does not include training code - only inference functionality.

Model E is designed for other materials (not 3C90, 3C94, 3E6, or 3F4).
It uses a SmoothPI operator instead of ExtendedPI for faster computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


# ========== Model Architecture ==========

class SmoothPI(nn.Module):
    """
    Fast smooth PI operator using Sigmoid for softening
    
    This is a simplified version of the PI operator that uses Sigmoid functions
    for smooth transitions instead of cumulative smoothing operations.
    It's faster than ExtendedPI but may be less accurate for some materials.
    
    Args:
        K: int - Number of PI operators to combine
        smoothness: float - Smoothness parameter for Sigmoid/Tanh functions
    """
    def __init__(self, K=4, smoothness=3.0):
        super().__init__()
        self.K = K
        self.smoothness = smoothness
    
    def forward(self, B, params):
        batch, T = B.shape
        K = self.K
        
        # Unpack parameters: reshape from (batch, T, 3*K) to (batch, T, 3, K)
        params = params.reshape(batch, T, 3, K)
        r_up   = params[:, :, 0, :]  # Radius parameter for rising
        r_down = params[:, :, 1, :]  # Radius parameter for falling
        w      = params[:, :, 2, :]  # Weight parameter (shared for both directions)
        
        # Expand input B to match K operators: (batch, T) -> (batch, T, K)
        B3 = B.unsqueeze(2).repeat(1, 1, K)
        
        # Calculate time derivative dB
        dB = torch.zeros_like(B)
        dB[:, 1:] = B[:, 1:] - B[:, :-1]
        dB3 = dB.unsqueeze(2).repeat(1, 1, K)
        
        # Use Sigmoid for smooth switching between rising and falling parameters
        switch_weight = torch.sigmoid(self.smoothness * dB3)
        r_eff = switch_weight * r_up + (1 - switch_weight) * r_down
        
        # Use Tanh as soft limiter for smooth hysteresis behavior
        center = 0.5 * (r_up + r_down)
        scale = torch.abs(r_up - r_down) + 1e-6
        
        Y_norm = torch.tanh(self.smoothness * (B3 - center) / scale)
        Y = center + 0.5 * scale * Y_norm
        
        # Weighted sum of all K operators to get final H
        H = torch.sum(w * Y, dim=2)
        return H


class Hybrid_PI_Model(nn.Module):
    def __init__(self, lstm_size=40, mlp_size=32, num_layers=1, K=4):
        super().__init__()

        self.hidden_size = lstm_size
        self.K = K

        # LSTM backbone: processes time series of magnetic properties
        # Input: (B, dB, ddB, H_value, H_mask) - 5 features per time step
        self.lstm = nn.LSTM(
            input_size=5,        # B, dB, ddB, H_value, H_mask
            hidden_size=lstm_size,
            num_layers=num_layers,
            batch_first=True
        )

        # PI parameter head: generates parameters for SmoothPI operator
        # Input: LSTM hidden state + temperature, Output: 3*K parameters
        # Note: SmoothPI uses 3 parameters (r_up, r_down, w) vs ExtendedPI's 6
        self.pi_head = nn.Sequential(
            nn.Linear(lstm_size + 1, mlp_size),  # +1 for temperature
            nn.ReLU(),
            nn.Linear(mlp_size, 3 * K)  # 3 parameters per K operators
        )

        # SmoothPI module: faster but potentially less accurate than ExtendedPI
        self.pi = SmoothPI(K=K, smoothness=3.0)

    def forward(self, B, dB, ddB,
                H_value_full, H_mask,
                Temp, split_point):
        """
        B:            [batch, 1000]
        dB:           [batch, 1000]
        ddB:          [batch, 1000]
        H_value_full: [batch, 1000]   (known H only; unknown = 0)
        H_mask:       [batch, 1000]   (1 for known region)
        Temp:         [batch, 1]
        split_point:  [batch]         (100, 500, 900)
        """
        batch, T = B.shape
        Temp = Temp.view(batch, 1)

        # LSTM input: (batch, 1000, 5)
        x_full = torch.stack([B, dB, ddB, H_value_full, H_mask], dim=2)

        lstm_seq, _ = self.lstm(x_full)  # [batch,1000,hidden]

        # Build PI parameters for ALL time steps (vectorized)
        hidden = lstm_seq.shape[-1]

        # Flatten (batch*T, hidden)
        lstm_flat = lstm_seq.reshape(batch * T, hidden)

        # Repeat temperature for all T steps
        Temp_rep = Temp.repeat_interleave(T, dim=0)

        # Conditioning
        cond = torch.cat([lstm_flat, Temp_rep], dim=1)

        # PI head generates parameters, reshape back to (batch, T, 3*K)
        pi_raw = self.pi_head(cond).reshape(batch, T, 3 * self.K)

        # Apply parameter constraints: radii must be positive
        r_up   = F.softplus(pi_raw[:, :, :self.K]) + 1e-4
        r_down = F.softplus(pi_raw[:, :, self.K:2*self.K]) + 1e-4
        w      = pi_raw[:, :, 2*self.K:]  # Weights can be positive or negative

        pi_params_full = torch.cat([r_up, r_down, w], dim=2)

        # PI operator produces full prediction curve [batch,1000]
        H_PI_full = self.pi(B, pi_params_full)

        # Prediction region mask: t >= sp
        time_idx = torch.arange(T, device=B.device).unsqueeze(0)
        pred_mask = (time_idx >= split_point.unsqueeze(1))    # [batch,1000]

        # Only keep valid region
        H_PI = H_PI_full * pred_mask.float()

        return H_PI


# ========== Global Variables: Model and Normalization Parameters ==========

# Global model instance (loaded lazily on first prediction)
_model_E = None

# Global normalization parameters
_norm_params_E = None

# Device configuration: use GPU if available, otherwise CPU
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_E(weights_path=None, norm_params_path=None):
    """
    Load Model E weights and normalization parameters
    
    This function loads the pre-trained model weights and normalization parameters
    needed for prediction. It uses default paths if not specified.
    
    Args:
        weights_path: str, optional - Path to model weights file (.sd format)
                     Defaults to weights/E.sd relative to project root
        norm_params_path: str, optional - Path to normalization parameters JSON file
                         Defaults to weights/Normalization_Params_E.json
    """
    global _model_E, _norm_params_E
    
    # 默认路径
    if weights_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base_dir, 'weights', 'E.sd')
    
    if norm_params_path is None:
        # 尝试查找归一化参数文件
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        norm_params_path = os.path.join(base_dir, 'weights', 'Normalization_Params_E.json')
        if not os.path.exists(norm_params_path):
            # 如果不存在，使用默认值（需要根据实际情况调整）
            print(f"Warning: Normalization params file not found at {norm_params_path}")
            print("Using default normalization parameters")
            _norm_params_E = {
                "mean_B": 0.0, "std_B": 1.0,
                "mean_dB": 0.0, "std_dB": 1.0,
                "mean_ddB": 0.0, "std_ddB": 1.0,
                "mean_H": 0.0, "std_H": 1.0,
                "mean_T": 0.0, "std_T": 1.0
            }
            # 不要提前返回，继续创建模型
    
    # 加载归一化参数
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r') as f:
            _norm_params_E = json.load(f)
    else:
        print(f"Warning: Normalization params file not found at {norm_params_path}")
        _norm_params_E = {
            "mean_B": 0.0, "std_B": 1.0,
            "mean_dB": 0.0, "std_dB": 1.0,
            "mean_ddB": 0.0, "std_ddB": 1.0,
            "mean_H": 0.0, "std_H": 1.0,
            "mean_T": 0.0, "std_T": 1.0
        }
    
    # 创建模型
    _model_E = Hybrid_PI_Model(lstm_size=40, mlp_size=32, num_layers=1, K=4).to(_device)
    
    # 加载权重
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=_device)
        if 'model_state' in checkpoint:
            _model_E.load_state_dict(checkpoint['model_state'])
        else:
            _model_E.load_state_dict(checkpoint)
        _model_E.eval()
        print(f"Model E loaded from {weights_path}")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Model initialized with random weights")


def predict_E(B, H, T, split_point=500, real_time=None):
    """
    Predict magnetic field strength H using Model E
    
    This function takes magnetic flux density B and temperature T as inputs,
    and predicts the corresponding magnetic field strength H. It also calculates
    permeability and core loss.
    
    Args:
        B: numpy array, shape (1000,) - Magnetic flux density sequence
        H: numpy array, shape (1000,) - Magnetic field strength sequence
           Known values should be provided, unknown values can be NaN or 0
        T: float - Temperature value
        split_point: int, optional - Split point indicating end of known H region (default: 500)
        real_time: float, optional - Real time period in seconds for core loss calculation
                   If None, defaults to 1.0 second
    
    Returns:
        dict: {
            'H_pred': list - Predicted H sequence (1000 values)
            'mu': list - Permeability sequence (1000 values)
            'core_loss': float - Calculated core loss
        }
    """
    global _model_E, _norm_params_E, _device
    
    if _model_E is None:
        load_model_E()
    
    if _model_E is None:
        raise ValueError("Model E failed to load. Please check the weights file.")
    
    if _norm_params_E is None:
        raise ValueError("Normalization parameters not loaded")
    
    # Convert to numpy arrays
    B = np.array(B, dtype=np.float32)
    H = np.array(H, dtype=np.float32)
    
    # Ensure length is 1000 (model requirement)
    if len(B) != 1000:
        raise ValueError(f"B must have length 1000, got {len(B)}")
    if len(H) != 1000:
        raise ValueError(f"H must have length 1000, got {len(H)}")
    
    # Handle NaN values in H (unknown regions)
    H_known = H.copy()
    H_mask = np.ones(1000, dtype=np.float32)
    
    # Find known H values (non-NaN parts)
    known_indices = ~np.isnan(H)
    if known_indices.sum() > 0:
        # Use last known position as split_point
        split_point = np.where(known_indices)[0][-1] + 1
    else:
        split_point = 0
    
    # Set unknown parts to 0
    H_known[~known_indices] = 0.0
    H_mask[~known_indices] = 0.0
    
    # Calculate dB (first derivative) and ddB (second derivative)
    # Use extrapolation for first point to maintain sequence length
    B_prep = 2 * B[0] - B[1] if len(B) > 1 else B[0]
    dB = np.diff(np.concatenate([[B_prep], B])).astype(np.float32)
    dB[dB == 0] = 1e-4  # Avoid exact zeros
    
    dB_prep = 2 * dB[0] - dB[1] if len(dB) > 1 else dB[0]
    ddB = np.diff(np.concatenate([[dB_prep], dB])).astype(np.float32)
    ddB[ddB == 0] = 1e-4  # Avoid exact zeros
    
    # Normalize inputs using stored normalization parameters
    mean_B = _norm_params_E["mean_B"]
    std_B = _norm_params_E["std_B"]
    mean_dB = _norm_params_E["mean_dB"]
    std_dB = _norm_params_E["std_dB"]
    mean_ddB = _norm_params_E["mean_ddB"]
    std_ddB = _norm_params_E["std_ddB"]
    mean_H = _norm_params_E["mean_H"]
    std_H = _norm_params_E["std_H"]
    mean_T = _norm_params_E["mean_T"]
    std_T = _norm_params_E["std_T"]
    
    B_norm = (B - mean_B) / (std_B + 1e-12)
    dB_norm = (dB - mean_dB) / (std_dB + 1e-12)
    ddB_norm = (ddB - mean_ddB) / (std_ddB + 1e-12)
    H_known_norm = (H_known - mean_H) / (std_H + 1e-12)
    T_norm = (np.array([T], dtype=np.float32) - mean_T) / (std_T + 1e-12)
    
    # Convert to PyTorch tensors and move to appropriate device
    B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    dB_tensor = torch.tensor(dB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    ddB_tensor = torch.tensor(ddB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_known_tensor = torch.tensor(H_known_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_mask_tensor = torch.tensor(H_mask, dtype=torch.float32).unsqueeze(0).to(_device)
    T_tensor = torch.tensor(T_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    split_point_tensor = torch.tensor([split_point], dtype=torch.long).to(_device)
    
    # Perform prediction (no gradient computation needed)
    with torch.no_grad():
        H_pred_norm = _model_E(
            B_tensor, dB_tensor, ddB_tensor,
            H_known_tensor, H_mask_tensor,
            T_tensor, split_point_tensor
        )
    
    # Denormalize prediction
    H_pred = (H_pred_norm.cpu().numpy()[0] * std_H + mean_H).tolist()
    H_pred = np.array(H_pred, dtype=np.float32)
    
    # In known regions, use original given H values (non-NaN parts) instead of predictions
    # Only NaN parts use predicted values
    known_mask = ~np.isnan(H)
    H_pred[known_mask] = H[known_mask]
    
    H_pred = H_pred.tolist()
    
    # Calculate permeability: mu = dB / dH
    # Since time steps are uniform, we can simplify: mu = dB / dH
    dB_denorm = dB * std_dB + mean_dB
    H_pred_array = np.array(H_pred)
    dH = np.diff(H_pred_array)
    dH[dH == 0] = 1e-6  # Avoid division by zero
    mu = (dB_denorm[1:] / dH).tolist()
    mu = [mu[0]] + mu  # Pad first value to maintain length
    
    # Calculate core loss: core_loss = ∫ B dH / real_time
    # Use trapezoidal integration for better accuracy
    B_denorm = B  # B is already denormalized
    if len(B_denorm) == len(H_pred_array) and len(B_denorm) > 1:
        # Trapezoidal integration: sum((B[i] + B[i+1])/2 * dH[i])
        B_mid = (B_denorm[:-1] + B_denorm[1:]) / 2.0
        integral_B_dH = np.sum(B_mid * dH)
    else:
        # Fallback: simple integration if lengths don't match
        integral_B_dH = np.sum(B_denorm[:len(dH)] * dH)
    
    # Use default time if not provided
    if real_time is None or real_time <= 0:
        real_time = 1.0
    
    core_loss = integral_B_dH / real_time
    
    return {
        'H_pred': H_pred,
        'mu': mu,
        'core_loss': float(core_loss)
    }

