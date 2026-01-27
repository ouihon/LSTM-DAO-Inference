"""
Model A for material 3C90.

Contains model architecture, loading functions, and prediction logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


class ExtendedPI(nn.Module):
    """Fully vectorized generalized PI operator for magnetic hysteresis modeling.
    
    Args:
        K: Number of PI operators to combine.
    """
    def __init__(self, K=64):
        super().__init__()
        self.K = K

    def forward(self, B, params):
        """Forward pass of ExtendedPI operator.
        
        Args:
            B: Input magnetic flux density, shape (batch, T).
            params: PI parameters, shape (batch, T, 6*K).
        
        Returns:
            H: Predicted magnetic field strength, shape (batch, T).
        """
        batch, T = B.shape
        K = self.K

        params = params.reshape(batch, T, 6, K)

        r_up = params[:, :, 0, :]
        r_down = params[:, :, 1, :]
        b_up = params[:, :, 2, :]
        b_down = params[:, :, 3, :]
        w_up = params[:, :, 4, :]
        w_down = params[:, :, 5, :]

        B3 = B.unsqueeze(2).expand(-1, -1, K)

        dB = torch.zeros_like(B)
        dB[:, 1:] = B[:, 1:] - B[:, :-1]
        dB[:, 0] = dB[:, 1]

        dB3 = dB.unsqueeze(2).expand(-1, -1, K)
        rising = (dB3 >= 0)

        r = torch.where(rising, r_up, r_down)
        b = torch.where(rising, b_up, b_down)
        w = torch.where(rising, w_up, w_down)

        Bw = B3 - b
        x_minus = Bw - r
        x_plus = Bw + r

        y_lower, _ = torch.cummax(x_minus, dim=1)
        y_upper, _ = torch.cummin(x_plus, dim=1)

        Y = torch.minimum(torch.maximum(Bw, y_lower), y_upper)
        H = torch.sum(w * Y, dim=2)
        return H


class Hybrid_PI_Model(nn.Module):
    """Hybrid PI model combining LSTM backbone with ExtendedPI operator.
    
    Args:
        lstm_size: LSTM hidden size.
        mlp_size: MLP hidden size.
        num_layers: Number of LSTM layers.
        K: Number of PI operators.
    """
    def __init__(self, lstm_size=40, mlp_size=16, num_layers=1, K=4):
        super().__init__()

        self.K = K
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=lstm_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.pi_head = nn.Sequential(
            nn.Linear(lstm_size + 1, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, 6 * K)
        )
        self.pi = ExtendedPI(K=K)

    def forward(self, B, dB, ddB, H_value_full, H_mask, Temp, split_point):
        """Forward pass of Hybrid_PI_Model.
        
        Args:
            B: Magnetic flux density, shape [batch, 1000].
            dB: First derivative of B, shape [batch, 1000].
            ddB: Second derivative of B, shape [batch, 1000].
            H_value_full: Known H values (unknown = 0), shape [batch, 1000].
            H_mask: Mask for known H regions (1 = known), shape [batch, 1000].
            Temp: Temperature, shape [batch, 1].
            split_point: Split point indices, shape [batch].
        
        Returns:
            H_PI: Predicted magnetic field strength, shape [batch, 1000].
        """
        batch, T = B.shape
        Temp = Temp.view(batch, 1)

        x_full = torch.stack([B, dB, ddB, H_value_full, H_mask], dim=2)
        lstm_seq, _ = self.lstm(x_full)
        hidden = lstm_seq.shape[-1]

        lstm_flat = lstm_seq.reshape(batch * T, hidden)
        Temp_rep = Temp.repeat_interleave(T, dim=0)
        cond = torch.cat([lstm_flat, Temp_rep], dim=1)

        pi_raw = self.pi_head(cond).reshape(batch, T, 6 * self.K)

        r_down = F.softplus(pi_raw[:, :, 0*self.K:1*self.K]) + 1e-4
        dr = F.softplus(pi_raw[:, :, 1*self.K:2*self.K])
        r_up = r_down + dr

        b_up = pi_raw[:, :, 2*self.K:3*self.K]
        b_down = pi_raw[:, :, 3*self.K:4*self.K]

        w_up_raw = pi_raw[:, :, 4*self.K:5*self.K]
        w_down_raw = pi_raw[:, :, 5*self.K:6*self.K]

        w_up = F.softplus(w_up_raw)
        w_up = w_up / (w_up.sum(dim=2, keepdim=True) + 1e-6)

        w_down = F.softplus(w_down_raw)
        w_down = w_down / (w_down.sum(dim=2, keepdim=True) + 1e-6)

        pi_params_full = torch.cat(
            [
                r_up, r_down,
                b_up, b_down,
                w_up, w_down
            ],
            dim=2
        )

        H_PI_full = self.pi(B, pi_params_full)
        time_idx = torch.arange(T, device=B.device).unsqueeze(0)
        pred_mask = (time_idx >= split_point.unsqueeze(1))
        H_PI = H_PI_full * pred_mask.float()

        return H_PI


_model_A = None
_norm_params_A = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_A(weights_path=None, norm_params_path=None):
    """Load Model A weights and normalization parameters.
    
    Args:
        weights_path: Path to model weights file (.sd format).
        norm_params_path: Path to normalization parameters JSON file.
    
    Raises:
        FileNotFoundError: If weights file not found.
    """
    global _model_A, _norm_params_A
    
    if weights_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base_dir, 'weights', 'A.sd')
    
    if norm_params_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        norm_params_path = os.path.join(base_dir, 'weights', 'Normalization_Params_A.json')
    
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r') as f:
            _norm_params_A = json.load(f)
    else:
        print(f"Warning: Normalization params file not found at {norm_params_path}")
        _norm_params_A = {
            "mean_B": 0.0, "std_B": 1.0,
            "mean_dB": 0.0, "std_dB": 1.0,
            "mean_ddB": 0.0, "std_ddB": 1.0,
            "mean_H": 0.0, "std_H": 1.0,
            "mean_T": 0.0, "std_T": 1.0
        }
    
    _model_A = Hybrid_PI_Model(lstm_size=40, mlp_size=16, num_layers=1, K=4).to(_device)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=_device)
        if 'model_state' in checkpoint:
            _model_A.load_state_dict(checkpoint['model_state'])
        else:
            _model_A.load_state_dict(checkpoint)
        _model_A.eval()
        print(f"Model A loaded from {weights_path}")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Model initialized with random weights")


def predict_A(B, H, T, split_point=500, real_time=None):
    """Predict magnetic field strength H using Model A.
    
    Args:
        B: Magnetic flux density sequence, shape (1000,).
        H: Magnetic field strength sequence, shape (1000,).
        T: Temperature value.
        split_point: Split point indicating end of known H region.
        real_time: Time period for core loss calculation.
    
    Returns:
        dict: Contains H_pred, mu, and core_loss.
    
    Raises:
        ValueError: If model not loaded or input dimensions invalid.
    """
    global _model_A, _norm_params_A, _device
    
    if _model_A is None:
        load_model_A()
    
    if _model_A is None:
        raise ValueError("Model A failed to load. Please check the weights file.")
    
    if _norm_params_A is None:
        raise ValueError("Normalization parameters not loaded")
    
    B = np.array(B, dtype=np.float32)
    H = np.array(H, dtype=np.float32)
    
    if len(B) != 1000:
        raise ValueError(f"B must have length 1000, got {len(B)}")
    if len(H) != 1000:
        raise ValueError(f"H must have length 1000, got {len(H)}")
    
    H_known = H.copy()
    H_mask = np.ones(1000, dtype=np.float32)
    
    known_indices = ~np.isnan(H)
    if known_indices.sum() > 0:
        split_point = np.where(known_indices)[0][-1] + 1
    else:
        split_point = 0
    
    H_known[~known_indices] = 0.0
    H_mask[~known_indices] = 0.0
    
    B_prep = 2 * B[0] - B[1] if len(B) > 1 else B[0]
    dB = np.diff(np.concatenate([[B_prep], B])).astype(np.float32)
    dB[dB == 0] = 1e-4
    
    dB_prep = 2 * dB[0] - dB[1] if len(dB) > 1 else dB[0]
    ddB = np.diff(np.concatenate([[dB_prep], dB])).astype(np.float32)
    ddB[ddB == 0] = 1e-4
    
    mean_B = _norm_params_A["mean_B"]
    std_B = _norm_params_A["std_B"]
    mean_dB = _norm_params_A["mean_dB"]
    std_dB = _norm_params_A["std_dB"]
    mean_ddB = _norm_params_A["mean_ddB"]
    std_ddB = _norm_params_A["std_ddB"]
    mean_H = _norm_params_A["mean_H"]
    std_H = _norm_params_A["std_H"]
    mean_T = _norm_params_A["mean_T"]
    std_T = _norm_params_A["std_T"]
    
    B_norm = (B - mean_B) / (std_B + 1e-12)
    dB_norm = (dB - mean_dB) / (std_dB + 1e-12)
    ddB_norm = (ddB - mean_ddB) / (std_ddB + 1e-12)
    H_known_norm = (H_known - mean_H) / (std_H + 1e-12)
    T_norm = (np.array([T], dtype=np.float32) - mean_T) / (std_T + 1e-12)
    
    B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    dB_tensor = torch.tensor(dB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    ddB_tensor = torch.tensor(ddB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_known_tensor = torch.tensor(H_known_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_mask_tensor = torch.tensor(H_mask, dtype=torch.float32).unsqueeze(0).to(_device)
    T_tensor = torch.tensor(T_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    split_point_tensor = torch.tensor([split_point], dtype=torch.long).to(_device)
    
    with torch.no_grad():
        H_pred_norm = _model_A(
            B_tensor, dB_tensor, ddB_tensor,
            H_known_tensor, H_mask_tensor,
            T_tensor, split_point_tensor
        )
    
    H_pred = (H_pred_norm.cpu().numpy()[0] * std_H + mean_H).tolist()
    H_pred = np.array(H_pred, dtype=np.float32)
    
    known_mask = ~np.isnan(H)
    H_pred[known_mask] = H[known_mask]
    
    H_pred = H_pred.tolist()
    
    dB_denorm = dB * std_dB + mean_dB
    H_pred_array = np.array(H_pred)
    dH = np.diff(H_pred_array)
    dH[dH == 0] = 1e-6
    mu = (dB_denorm[1:] / dH).tolist()
    mu = [mu[0]] + mu
    
    B_denorm = B
    if len(B_denorm) == len(H_pred_array) and len(B_denorm) > 1:
        B_mid = (B_denorm[:-1] + B_denorm[1:]) / 2.0
        integral_B_dH = np.sum(B_mid * dH)
    else:
        integral_B_dH = np.sum(B_denorm[:len(dH)] * dH)
    
    if real_time is None or real_time <= 0:
        real_time = 1.0
    
    core_loss = integral_B_dH / real_time
    
    return {
        'H_pred': H_pred,
        'mu': mu,
        'core_loss': float(core_loss)
    }
