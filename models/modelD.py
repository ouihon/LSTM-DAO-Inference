"""
Model D definition and prediction functions for material 3F4.

Contains model architecture and inference logic (no training code).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


# ========== Model Architecture ==========

class ExtendedPI(nn.Module):
    """
    Fully vectorized generalized PI operator
    
    This is a hysteresis operator used for modeling magnetic materials.
    It operates without time loops, making it efficient for batch processing.
    
    Args:
        K: int - Number of PI operators to combine
    
    Input params shape: (batch, T, 6*K)
        Parameters are organized as:
        - r_up, r_down: radius parameters for rising/falling
        - b_up, b_down: bias parameters for rising/falling  
        - w_up, w_down: weight parameters for rising/falling
    """
    def __init__(self, K=64):
        super().__init__()
        self.K = K

    def forward(self, B, params):
        batch, T = B.shape
        K = self.K

        # ----------------------------------------------------------
        # 解包参数
        # ----------------------------------------------------------
        params = params.reshape(batch, T, 6, K)

        r_up   = params[:, :, 0, :]
        r_down = params[:, :, 1, :]

        b_up   = params[:, :, 2, :]
        b_down = params[:, :, 3, :]

        w_up   = params[:, :, 4, :]
        w_down = params[:, :, 5, :]

        # ----------------------------------------------------------
        # 扩展输入
        # ----------------------------------------------------------
        B3 = B.unsqueeze(2).expand(-1, -1, K)

        # 时间导数
        dB = torch.zeros_like(B)
        dB[:, 1:] = B[:, 1:] - B[:, :-1]
        dB[:, 0] = dB[:, 1]

        dB3 = dB.unsqueeze(2).expand(-1, -1, K)
        rising = (dB3 >= 0)

        # ----------------------------------------------------------
        # 方向相关选择
        # ----------------------------------------------------------
        r = torch.where(rising, r_up, r_down)
        b = torch.where(rising, b_up, b_down)
        w = torch.where(rising, w_up, w_down)

        # ----------------------------------------------------------
        # 线性PI偏置偏移
        # ----------------------------------------------------------
        Bw = B3 - b

        # ----------------------------------------------------------
        # Play算子
        # ----------------------------------------------------------
        x_minus = Bw - r
        x_plus  = Bw + r

        y_lower, _ = torch.cummax(x_minus, dim=1)
        y_upper, _ = torch.cummin(x_plus,  dim=1)

        Y = torch.minimum(torch.maximum(Bw, y_lower), y_upper)

        # ----------------------------------------------------------
        # 加权求和
        # ----------------------------------------------------------
        H = torch.sum(w * Y, dim=2)
        return H


class Hybrid_PI_Model(nn.Module):
    def __init__(self, lstm_size=40, mlp_size=16, num_layers=1, K=4):
        super().__init__()

        self.K = K

        # -------------------------------
        # LSTM backbone
        # -------------------------------
        self.lstm = nn.LSTM(
            input_size=5,        # B, dB, ddB, H_value, H_mask
            hidden_size=lstm_size,
            num_layers=num_layers,
            batch_first=True
        )

        # -------------------------------
        # PI parameter head (6*K)
        # -------------------------------
        self.pi_head = nn.Sequential(
            nn.Linear(lstm_size + 1, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, 6 * K)
        )

        self.pi = ExtendedPI(K=K)

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

        # PI head → reshape back
        pi_raw = self.pi_head(cond).reshape(batch, T, 6 * self.K)

        # ----------------------------------------------------------
        # 参数约束
        # ----------------------------------------------------------
        r_down = F.softplus(pi_raw[:, :, 0*self.K:1*self.K]) + 1e-4
        dr     = F.softplus(pi_raw[:, :, 1*self.K:2*self.K])
        r_up   = r_down + dr

        b_up   = pi_raw[:, :, 2*self.K:3*self.K]
        b_down = pi_raw[:, :, 3*self.K:4*self.K]

        w_up_raw   = pi_raw[:, :, 4*self.K:5*self.K]
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
_model_D = None

# Global normalization parameters
_norm_params_D = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_D(weights_path=None, norm_params_path=None):
    """
    Load Model D weights and normalization parameters
    
    This function loads the pre-trained model weights and normalization parameters
    needed for prediction. It uses default paths if not specified.
    
    Args:
        weights_path: str, optional - Path to model weights file (.sd format)
                     Defaults to weights/D.sd relative to project root
        norm_params_path: str, optional - Path to normalization parameters JSON file
                         Defaults to weights/Normalization_Params_D.json
    """
    global _model_D, _norm_params_D
    
    # 默认路径
    if weights_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_path = os.path.join(base_dir, 'weights', 'D.sd')
    
    if norm_params_path is None:
        # 尝试查找归一化参数文件
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        norm_params_path = os.path.join(base_dir, 'weights', 'Normalization_Params_D.json')
        if not os.path.exists(norm_params_path):
            # 如果不存在，使用默认值（需要根据实际情况调整）
            print(f"Warning: Normalization params file not found at {norm_params_path}")
            print("Using default normalization parameters")
            _norm_params_D = {
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
            _norm_params_D = json.load(f)
    else:
        print(f"Warning: Normalization params file not found at {norm_params_path}")
        _norm_params_D = {
            "mean_B": 0.0, "std_B": 1.0,
            "mean_dB": 0.0, "std_dB": 1.0,
            "mean_ddB": 0.0, "std_ddB": 1.0,
            "mean_H": 0.0, "std_H": 1.0,
            "mean_T": 0.0, "std_T": 1.0
        }
    
    # 创建模型
    _model_D = Hybrid_PI_Model(lstm_size=40, mlp_size=16, num_layers=1, K=4).to(_device)
    
    # 加载权重
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=_device)
        if 'model_state' in checkpoint:
            _model_D.load_state_dict(checkpoint['model_state'])
        else:
            _model_D.load_state_dict(checkpoint)
        _model_D.eval()
        print(f"Model D loaded from {weights_path}")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Model initialized with random weights")


def predict_D(B, H, T, split_point=500, real_time=None):
    """
    Predict magnetic field strength H using Model D
    
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
    global _model_D, _norm_params_D, _device
    
    if _model_D is None:
        load_model_D()
    
    if _model_D is None:
        raise ValueError("Model D failed to load. Please check the weights file.")
    
    if _norm_params_D is None:
        raise ValueError("Normalization parameters not loaded")
    
    # 转换为numpy数组
    B = np.array(B, dtype=np.float32)
    H = np.array(H, dtype=np.float32)
    
    # 确保长度为1000
    if len(B) != 1000:
        raise ValueError(f"B must have length 1000, got {len(B)}")
    if len(H) != 1000:
        raise ValueError(f"H must have length 1000, got {len(H)}")
    
    # 处理H中的NaN值
    H_known = H.copy()
    H_mask = np.ones(1000, dtype=np.float32)
    
    # 找到已知的H值（非NaN的部分）
    known_indices = ~np.isnan(H)
    if known_indices.sum() > 0:
        # 使用已知的最后一个位置作为split_point
        split_point = np.where(known_indices)[0][-1] + 1
    else:
        split_point = 0
    
    # 未知部分设为0
    H_known[~known_indices] = 0.0
    H_mask[~known_indices] = 0.0
    
    # 计算dB和ddB
    B_prep = 2 * B[0] - B[1] if len(B) > 1 else B[0]
    dB = np.diff(np.concatenate([[B_prep], B])).astype(np.float32)
    dB[dB == 0] = 1e-4
    
    dB_prep = 2 * dB[0] - dB[1] if len(dB) > 1 else dB[0]
    ddB = np.diff(np.concatenate([[dB_prep], dB])).astype(np.float32)
    ddB[ddB == 0] = 1e-4
    
    # 归一化
    mean_B = _norm_params_D["mean_B"]
    std_B = _norm_params_D["std_B"]
    mean_dB = _norm_params_D["mean_dB"]
    std_dB = _norm_params_D["std_dB"]
    mean_ddB = _norm_params_D["mean_ddB"]
    std_ddB = _norm_params_D["std_ddB"]
    mean_H = _norm_params_D["mean_H"]
    std_H = _norm_params_D["std_H"]
    mean_T = _norm_params_D["mean_T"]
    std_T = _norm_params_D["std_T"]
    
    B_norm = (B - mean_B) / (std_B + 1e-12)
    dB_norm = (dB - mean_dB) / (std_dB + 1e-12)
    ddB_norm = (ddB - mean_ddB) / (std_ddB + 1e-12)
    H_known_norm = (H_known - mean_H) / (std_H + 1e-12)
    T_norm = (np.array([T], dtype=np.float32) - mean_T) / (std_T + 1e-12)
    
    # 转换为tensor
    B_tensor = torch.tensor(B_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    dB_tensor = torch.tensor(dB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    ddB_tensor = torch.tensor(ddB_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_known_tensor = torch.tensor(H_known_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    H_mask_tensor = torch.tensor(H_mask, dtype=torch.float32).unsqueeze(0).to(_device)
    T_tensor = torch.tensor(T_norm, dtype=torch.float32).unsqueeze(0).to(_device)
    split_point_tensor = torch.tensor([split_point], dtype=torch.long).to(_device)
    
    # 预测
    with torch.no_grad():
        H_pred_norm = _model_D(
            B_tensor, dB_tensor, ddB_tensor,
            H_known_tensor, H_mask_tensor,
            T_tensor, split_point_tensor
        )
    
    # 反归一化
    H_pred = (H_pred_norm.cpu().numpy()[0] * std_H + mean_H).tolist()
    H_pred = np.array(H_pred, dtype=np.float32)
    
    # 在已知区域，使用原始给定的H值（非NaN的部分），而不是预测值
    # 只有NaN的部分才使用预测值
    known_mask = ~np.isnan(H)
    H_pred[known_mask] = H[known_mask]
    
    H_pred = H_pred.tolist()
    
    # 计算磁导率 mu = dB / (dH * dt)，这里简化为 mu = dB / dH
    # 由于时间步长相同，可以简化为 mu = dB / dH
    dB_denorm = dB * std_dB + mean_dB
    H_pred_array = np.array(H_pred)
    dH = np.diff(H_pred_array)
    dH[dH == 0] = 1e-6  # 避免除零
    mu = (dB_denorm[1:] / dH).tolist()
    mu = [mu[0]] + mu  # 补齐第一个值
    
    # 计算核心损耗：core_loss = ∫ B dH / real_time
    # ∫ B dH = sum(B * dH)，使用梯形积分提高精度
    B_denorm = B  # B已经是反归一化的
    if len(B_denorm) == len(H_pred_array) and len(B_denorm) > 1:
        # 使用梯形积分：sum((B[i] + B[i+1])/2 * dH[i])
        B_mid = (B_denorm[:-1] + B_denorm[1:]) / 2.0
        integral_B_dH = np.sum(B_mid * dH)
    else:
        # 如果长度不匹配，使用简单方法
        integral_B_dH = np.sum(B_denorm[:len(dH)] * dH)
    
    # 如果没有提供真实时间，使用默认值1.0
    if real_time is None or real_time <= 0:
        real_time = 1.0
    
    core_loss = integral_B_dH / real_time
    
    return {
        'H_pred': H_pred,
        'mu': mu,
        'core_loss': float(core_loss)
    }

