"""
Flask web application for AI Magnetic Component Design

This application provides a web interface for designing and optimizing magnetic components,
particularly for DC-AC converters. It includes:
- Real-time prediction using pre-trained neural network models
- Model fine-tuning capabilities
- Interactive design center with visualization tools
- Comprehensive documentation

Author: Magnetic Design Team
"""

from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
import numpy as np
import uuid
import zipfile
import shutil
import json
from datetime import datetime, timedelta
from threading import Thread
from werkzeug.utils import secure_filename
import time

# Add models directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

# Import prediction and loading functions for all models
from models.modelA import predict_A, load_model_A
from models.modelB import predict_B, load_model_B
from models.modelC import predict_C, load_model_C
from models.modelD import predict_D, load_model_D
from models.modelE import predict_E, load_model_E

# Initialize Flask application
app = Flask(__name__)

# ========== Configuration Constants ==========

# Directory for temporary file uploads
UPLOAD_FOLDER = 'temp'

# Directory for extracted training data
EXTRACT_FOLDER = 'uploads'

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'zip'}

# Number of days to retain cached files before cleanup
CACHE_RETENTION_DAYS = 15

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# Task ID cache: stores creation timestamps for cleanup purposes
task_cache = {}

# Import training status and training function from fine-tuning module
from train_finetune import training_status, train_model

# Material ID to model ID mapping (1-5 corresponds to models A-E)
MATERIAL_MODEL_MAP = {
    1: 'A',  # Material: 3C90
    2: 'B',  # Material: 3C94
    3: 'C',  # Material: 3E6
    4: 'D',  # Material: 3F4
    5: 'E'   # Other Material
}

# ========== Model Initialization ==========

# Dictionary mapping model IDs to their loading functions
models_to_load = {
    'A': load_model_A,
    'B': load_model_B,
    'C': load_model_C,
    'D': load_model_D,
    'E': load_model_E
}

# Load all models at startup
print("Initializing models...")
for model_id, load_func in models_to_load.items():
    print(f"Loading Model {model_id}...")
    try:
        load_func()
        print(f"Model {model_id} loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load Model {model_id}: {e}")

# ========== Route Handlers ==========

@app.route('/')
def index():
    """Render the main index page with navigation sidebar"""
    return render_template('index.html')


@app.route('/design')
def design():
    """Render the design center page for interactive component design"""
    return render_template('design.html')


@app.route('/document')
def document():
    """Render the documentation page with user manual"""
    return render_template('document.html')


@app.route('/fine_tune')
def fine_tune():
    """Render the fine-tuning page for model training"""
    return render_template('fine_tune.html')


# ========== Prediction Functions ==========

# Mapping of model IDs to their prediction functions
PREDICT_FUNCTIONS = {
    'A': predict_A,
    'B': predict_B,
    'C': predict_C,
    'D': predict_D,
    'E': predict_E
}


def predict_with_sliding_window(B, T, model_id, window_size=1000, real_time=None):
    """
    Perform prediction using sliding window approach for sequences longer than window_size
    
    This function handles B sequences of any length by:
    - Direct prediction if length <= window_size
    - Sliding window prediction if length > window_size
    - Proper handling of window boundaries and padding
    
    Args:
        B: numpy array, B sequence (magnetic flux density), can be longer than 1000
        T: float, temperature value
        model_id: str, model identifier ('A', 'B', 'C', 'D', 'E')
        window_size: int, size of sliding window (default: 1000)
        real_time: float, real time period in seconds (for core loss calculation)
                   If None, defaults to 1.0 second
    
    Returns:
        dict: {
            'H_pred': list, predicted H sequence (same length as input B)
            'mu': list, permeability sequence
            'core_loss': float, calculated core loss
        }
    """
    B = np.array(B, dtype=np.float32)
    B_len = len(B)
    
    # Use default time if not provided (assumes uniform data distribution)
    if real_time is None or real_time <= 0:
        real_time = 1.0  # Default: 1 second per cycle
    
    # Case 1: B length is within window size - direct prediction
    if B_len <= window_size:
        # Initialize H: first value is 0, rest are NaN (masked)
        H = np.full(window_size, np.nan, dtype=np.float32)
        H[0] = 0.0
        
        # Pad B to window_size if needed
        if B_len < window_size:
            B_padded = np.pad(B, (0, window_size - B_len), mode='edge')
        else:
            B_padded = B
        
        # Perform prediction
        predict_func = PREDICT_FUNCTIONS[model_id]
        result = predict_func(B_padded, H, T, split_point=1, real_time=real_time)
        
        # Trim results to original length if padding was applied
        if B_len < window_size:
            result['H_pred'] = result['H_pred'][:B_len]
            result['mu'] = result['mu'][:B_len]
        
        return result
    
    # Case 2: B length exceeds window size - use sliding window
    H_pred_full = []
    mu_full = []
    core_loss_total = 0.0
    
    # Sliding window with step size equal to window_size
    # Last window may need special handling if it's shorter
    step = window_size
    
    for start_idx in range(0, B_len, step):
        end_idx = min(start_idx + window_size, B_len)
        window_B = B[start_idx:end_idx]
        actual_window_len = len(window_B)
        
        # Handle last window if it's shorter than window_size
        if actual_window_len < window_size and end_idx == B_len:
            padding_needed = window_size - actual_window_len
            
            # Use preceding real data for padding
            if start_idx >= padding_needed:
                # Sufficient preceding data available
                padding_data = B[start_idx - padding_needed:start_idx]
                window_B = np.concatenate([padding_data, window_B])
            else:
                # Not enough preceding data, use available data with repetition
                padding_data = B[:start_idx] if start_idx > 0 else np.array([B[0]])
                # Repeat padding to required length
                while len(padding_data) < padding_needed:
                    padding_data = np.concatenate([[B[0]], padding_data])
                window_B = np.concatenate([padding_data[-padding_needed:], window_B])
        
        # Edge case: ensure window is exactly window_size (shouldn't happen, but safety check)
        if len(window_B) < window_size:
            window_B = np.pad(window_B, (0, window_size - len(window_B)), mode='edge')
        
        # Initialize H: first value is 0 (or last prediction from previous window), rest are NaN
        H = np.full(window_size, np.nan, dtype=np.float32)
        if start_idx == 0:
            H[0] = 0.0
        else:
            # Use last predicted value from previous window for continuity
            H[0] = H_pred_full[-1] if len(H_pred_full) > 0 else 0.0
        
        # Perform prediction on this window
        predict_func = PREDICT_FUNCTIONS[model_id]
        result = predict_func(window_B, H, T, split_point=1)
        
        # Extract predictions
        window_H_pred = result['H_pred']
        window_mu = result['mu']
        
        # Use only actual window length (trim padding if applied)
        if actual_window_len < window_size:
            window_H_pred = window_H_pred[:actual_window_len]
            window_mu = window_mu[:actual_window_len]
        
        # Append to full results
        H_pred_full.extend(window_H_pred)
        mu_full.extend(window_mu)
    
    # Ensure output length matches input B length
    H_pred_full = H_pred_full[:B_len]
    mu_full = mu_full[:B_len]
    
    # Calculate core loss: ∫ B dH / real_time
    # Use complete B and H_pred sequences for accurate calculation
    if len(B) > 1 and len(H_pred_full) > 1:
        H_pred_array = np.array(H_pred_full)
        dH = np.diff(H_pred_array)
        
        B_array = np.array(B)
        if len(B_array) == len(H_pred_array):
            # Use trapezoidal integration for better accuracy
            # ∫ B dH ≈ sum((B[i] + B[i+1])/2 * dH[i])
            B_mid = (B_array[:-1] + B_array[1:]) / 2.0
            integral_B_dH = np.sum(B_mid * dH)
        else:
            # Fallback: simple integration if lengths don't match
            integral_B_dH = np.sum(B_array[:len(dH)] * dH)
        
        # Core loss = ∫ B dH / real_time
        core_loss_total = integral_B_dH / real_time if real_time > 0 else integral_B_dH
    else:
        core_loss_total = 0.0
    
    return {
        'H_pred': H_pred_full,
        'mu': mu_full,
        'core_loss': float(core_loss_total)
    }


# ========== API Endpoints ==========

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for magnetic property prediction
    
    Accepts JSON with:
    - materialId (1-5) or modelId ('A'-'E')
    - sample: {B: list, T: float, real_time: float (optional)}
    
    Returns:
    - H_pred: predicted magnetic field strength
    - mu: permeability sequence
    - core_loss: calculated core loss
    """
    try:
        data = request.json
        
        # Validate input: must have either materialId or modelId
        if 'materialId' not in data and 'modelId' not in data:
            return jsonify({'error': 'materialId or modelId is required'}), 400
        
        # Prefer materialId, fall back to modelId if not provided
        if 'materialId' in data:
            material_id = data['materialId']
            if material_id not in MATERIAL_MODEL_MAP:
                return jsonify({'error': f'Invalid materialId: {material_id}'}), 400
            model_id = MATERIAL_MODEL_MAP[material_id]
        else:
            model_id = data['modelId']
        
        # Validate model ID
        if model_id not in PREDICT_FUNCTIONS:
            return jsonify({'error': f'Model {model_id} not supported'}), 400
        
        # Validate sample data
        if 'sample' not in data:
            return jsonify({'error': 'sample is required'}), 400
        
        sample = data['sample']
        
        # Validate sample contains required fields
        if 'B' not in sample or 'T' not in sample:
            return jsonify({'error': 'sample must contain B and T'}), 400
        
        B = sample['B']
        T = sample['T']
        # Get real_time if provided (for accurate core loss calculation)
        real_time = sample.get('real_time', None)
        
        # B can be any length; H is not required (defaults to first value = 0, rest masked)
        
        # Perform prediction using sliding window approach
        result = predict_with_sliding_window(B, T, model_id, real_time=real_time)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ========== Fine-tuning Related Functions ==========

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_hyperparameters(data):
    """
    Validate hyperparameters for model fine-tuning
    
    Checks all hyperparameters are within acceptable ranges:
    - epochs: 10-1000
    - learning_rate: 0.0001-0.1
    - batch_size: 8-128
    - validation_split: 0.1-0.4
    - window_size (stride): 1-1000
    - loss_type: 'MSE', 'RMSE', or 'energy'
    - frozen_layers: 0-3
    - base_model: 'A', 'B', 'C', 'D', or 'E'
    
    Returns:
        list: List of error messages (empty if validation passes)
    """
    errors = []
    
    # Validate epochs
    epochs = data.get('epochs', 100)
    if not isinstance(epochs, int) or epochs < 10 or epochs > 1000:
        errors.append('epochs must be between 10 and 1000')
    
    # Validate learning rate
    learning_rate = data.get('learning_rate', 0.001)
    if not isinstance(learning_rate, (int, float)) or learning_rate < 0.0001 or learning_rate > 0.1:
        errors.append('learning_rate must be between 0.0001 and 0.1')
    
    # Validate batch size
    batch_size = data.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size < 8 or batch_size > 128:
        errors.append('batch_size must be between 8 and 128')
    
    # Validate validation split
    validation_split = data.get('validation_split', 0.2)
    if not isinstance(validation_split, (int, float)) or validation_split < 0.1 or validation_split > 0.4:
        errors.append('validation_split must be between 0.1 and 0.4')
    
    # Validate window stride (note: frontend sends 'window_size' but it's actually stride)
    # Window size is fixed at 1000, this parameter controls the stride
    window_stride = data.get('window_size', 500)
    if not isinstance(window_stride, int) or window_stride < 1 or window_stride > 1000:
        errors.append('window_size (stride) must be between 1 and 1000')
    
    # Validate loss type
    loss_type = data.get('loss_type', 'MSE')
    if loss_type not in ['MSE', 'RMSE', 'energy']:
        errors.append('loss_type must be one of: MSE, RMSE, or energy')
    
    # Validate frozen layers
    frozen_layers = data.get('frozen_layers', 1)
    if not isinstance(frozen_layers, int) or frozen_layers < 0 or frozen_layers > 3:
        errors.append('frozen_layers must be between 0 and 3')
    
    # Validate base model
    base_model = data.get('base_model', 'A')
    if base_model not in ['A', 'B', 'C', 'D', 'E']:
        errors.append('base_model must be one of: A, B, C, D, or E')
    
    return errors


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    API endpoint for uploading training data
    
    Accepts a ZIP file containing:
    - H.csv: Magnetic field strength data (N samples, M time steps)
    - B.csv: Magnetic flux density data (N samples, M time steps)
    - T.csv: Temperature data (N samples, 1 value per sample)
    
    Returns:
    - task_id: Unique identifier for this training task
    - samples: Number of samples
    - sequence_length: Length of time sequences
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Filename is empty'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only ZIP files are accepted'}), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file to temp directory
        temp_dir = os.path.join(UPLOAD_FOLDER, task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        # Extract ZIP to uploads directory
        extract_dir = os.path.join(EXTRACT_FOLDER, task_id)
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Verify required files exist
        required_files = ['H.csv', 'T.csv', 'B.csv']
        missing_files = []
        for req_file in required_files:
            if not os.path.exists(os.path.join(extract_dir, req_file)):
                missing_files.append(req_file)
        
        if missing_files:
            # Cleanup on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return jsonify({'error': f'Missing required files: {", ".join(missing_files)}'}), 400
        
        # Validate data dimensions
        try:
            H_data = np.loadtxt(os.path.join(extract_dir, 'H.csv'), delimiter=',')
            T_data = np.loadtxt(os.path.join(extract_dir, 'T.csv'), delimiter=',')
            B_data = np.loadtxt(os.path.join(extract_dir, 'B.csv'), delimiter=',')
            
            # Handle dimension reshaping (support 1D arrays)
            if len(H_data.shape) == 1:
                H_data = H_data.reshape(-1, 1)
            if len(B_data.shape) == 1:
                B_data = B_data.reshape(-1, 1)
            if len(T_data.shape) == 1:
                T_data = T_data.reshape(-1, 1)
            
            # Extract dimensions: (N samples, M time steps)
            N_H, M_H = H_data.shape
            N_B, M_B = B_data.shape
            N_T, M_T = T_data.shape
            
            # Validate sample count consistency
            if N_H != N_T or N_H != N_B:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Sample count mismatch: H={N_H}, T={N_T}, B={N_B}'}), 400
            
            # Validate sequence length consistency
            if M_H != M_B:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Sequence length mismatch: H={M_H}, B={M_B}'}), 400
            
            # Validate temperature dimension (should be scalar per sample)
            if M_T != 1:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Temperature should be (N, 1), got (N, {M_T})'}), 400
            
        except Exception as e:
            # Cleanup on validation error
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return jsonify({'error': f'Data validation failed: {str(e)}'}), 400
        
        # Cleanup temp directory (data is now in extract_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Record task ID and creation time for cleanup tracking
        task_cache[task_id] = datetime.now()
        
        return jsonify({
            'task_id': task_id,
            'message': 'File uploaded successfully',
            'samples': int(N_H),
            'sequence_length': int(M_H)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/finetune', methods=['POST'])
def finetune():
    """
    API endpoint to start model fine-tuning
    
    Accepts JSON with:
    - task_id: ID from previous upload
    - base_model: 'A'-'E'
    - epochs, learning_rate, batch_size, validation_split
    - window_size (stride), loss_type, frozen_layers
    
    Returns:
    - status: 'ok' if training started successfully
    - task_id: Training task identifier
    """
    try:
        data = request.json
        
        # Validate task_id is provided
        if 'task_id' not in data:
            return jsonify({'error': 'task_id is required'}), 400
        
        task_id = data['task_id']
        
        # Check if task data exists
        data_dir = os.path.join(EXTRACT_FOLDER, task_id)
        if not os.path.exists(data_dir):
            return jsonify({'error': 'Task not found, please upload file first'}), 400
        
        # Validate hyperparameters
        errors = validate_hyperparameters(data)
        if errors:
            return jsonify({'error': '; '.join(errors)}), 400
        
        # Check if training is already in progress
        if task_id in training_status:
            status = training_status[task_id].get('status', 'unknown')
            if status == 'training':
                return jsonify({'error': 'Task is already training'}), 400
        
        # Update task cache timestamp
        task_cache[task_id] = datetime.now()
        
        # Extract hyperparameters
        base_model = data.get('base_model', 'A')
        epochs = data.get('epochs', 100)
        learning_rate = data.get('learning_rate', 0.001)
        batch_size = data.get('batch_size', 32)
        validation_split = data.get('validation_split', 0.2)
        window_stride = data.get('window_size', 500)  # Note: frontend sends 'window_size' but it's stride
        loss_type = data.get('loss_type', 'MSE')
        frozen_layers = data.get('frozen_layers', 1)
        
        # Start training in background thread
        thread = Thread(target=train_model, args=(
            task_id, base_model, data_dir, epochs, learning_rate,
            batch_size, validation_split, window_stride, loss_type, frozen_layers
        ))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'ok',
            'message': 'Training started',
            'task_id': task_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500


@app.route('/api/poll/<task_id>', methods=['GET'])
def poll_status(task_id):
    """
    API endpoint to poll training status
    
    Returns current training status including:
    - status: 'training', 'completed', 'failed', or 'not_found'
    - progress: percentage complete (0-100)
    - current_epoch: current epoch number
    - train_loss, val_loss: loss arrays
    - train_rmse, val_rmse: RMSE arrays
    - train_energy_loss, val_energy_loss: energy loss arrays
    - initial_rmse, initial_energy_loss: baseline metrics
    - epoch_times: time per epoch
    - total_windows, train_windows, val_windows: window counts
    """
    try:
        if task_id not in training_status:
            return jsonify({
                'status': 'not_found',
                'message': 'Task not found'
            })
        
        status = training_status[task_id]
        
        return jsonify({
            'status': status.get('status', 'unknown'),
            'progress': status.get('progress', 0),
            'current_epoch': status.get('current_epoch', 0),
            'train_loss': status.get('train_loss', []),
            'val_loss': status.get('val_loss', []),
            'train_rmse': status.get('train_rmse', []),
            'val_rmse': status.get('val_rmse', []),
            'train_energy_loss': status.get('train_energy_loss', []),
            'val_energy_loss': status.get('val_energy_loss', []),
            'initial_rmse': status.get('initial_rmse'),
            'initial_energy_loss': status.get('initial_energy_loss'),
            'epoch_times': status.get('epoch_times', []),
            'message': status.get('message', ''),
            'total_windows': status.get('total_windows', 0),
            'train_windows': status.get('train_windows', 0),
            'val_windows': status.get('val_windows', 0)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to query status: {str(e)}'}), 500


@app.route('/api/download/<task_id>/<file_type>', methods=['GET'])
def download_file(task_id, file_type):
    """
    API endpoint to download training results
    
    Available file types:
    - 'model': Base model definition file (.sd format)
    - 'weights': Trained model weights (.sd format)
    - 'report': Training report PDF
    
    Args:
        task_id: Training task identifier
        file_type: Type of file to download
    """
    try:
        # Validate file type
        if file_type not in ['model', 'weights', 'report']:
            return jsonify({'error': 'Invalid file type'}), 400
        
        output_dir = os.path.join(EXTRACT_FOLDER, task_id)
        
        if file_type == 'model':
            # Base model definition file (.sd format)
            base_model = 'A'  # Default, should ideally be retrieved from task info
            weights_path = os.path.join(os.path.dirname(__file__), 'weights', f'{base_model}.sd')
            
            if not os.path.exists(weights_path):
                return jsonify({'error': 'Model file not found'}), 404
            
            return send_file(weights_path, as_attachment=True, download_name=f'{base_model}.sd')
        
        elif file_type == 'weights':
            # Trained model weights
            weights_file = os.path.join(output_dir, 'model_weights.sd')
            
            if not os.path.exists(weights_file):
                return jsonify({'error': 'Weights file not found, training may not be complete'}), 404
            
            return send_file(weights_file, as_attachment=True, download_name='model_weights.sd')
        
        elif file_type == 'report':
            # Training report PDF
            report_file = os.path.join(output_dir, 'training_report.pdf')
            
            if not os.path.exists(report_file):
                return jsonify({'error': 'Report file not found, training may not be complete'}), 404
            
            return send_file(report_file, as_attachment=True, download_name='training_report.pdf')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


# ========== Cleanup and Maintenance ==========

def cleanup_old_files():
    """
    Clean up files older than CACHE_RETENTION_DAYS
    
    This function removes old training data and temporary files
    to prevent disk space issues. Runs daily at midnight.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=CACHE_RETENTION_DAYS)
        
        # Clean up uploads directory
        if os.path.exists(EXTRACT_FOLDER):
            for item in os.listdir(EXTRACT_FOLDER):
                item_path = os.path.join(EXTRACT_FOLDER, item)
                if os.path.isdir(item_path):
                    # Check if task ID is in cache
                    if item in task_cache:
                        if task_cache[item] < cutoff_date:
                            shutil.rmtree(item_path, ignore_errors=True)
                            del task_cache[item]
                            print(f"Cleaned up expired task: {item}")
                    else:
                        # If not in cache, check directory modification time
                        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                        if mtime < cutoff_date:
                            shutil.rmtree(item_path, ignore_errors=True)
                            print(f"Cleaned up expired directory: {item}")
        
        print(f"Cleanup completed: {datetime.now()}")
    except Exception as e:
        print(f"Cleanup task failed: {e}")


def run_scheduler():
    """
    Background scheduler that runs cleanup task daily at midnight
    
    Checks every minute and executes cleanup when time matches 00:00
    """
    while True:
        now = datetime.now()
        # Execute cleanup at midnight (00:00)
        if now.hour == 0 and now.minute == 0:
            cleanup_old_files()
            # Wait 1 minute to avoid repeated execution
            time.sleep(60)
        else:
            time.sleep(60)


# Start cleanup scheduler thread
scheduler_thread = Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()


# ========== Application Entry Point ==========

if __name__ == '__main__':
    # Run Flask development server
    # Note: For production, use a proper WSGI server like Gunicorn
    app.run(debug=True, port=6008)
