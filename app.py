"""
Flask web application for AI magnetic component design and optimization.

Provides web interface for magnetic component design with real-time prediction,
model fine-tuning, and interactive visualization tools.
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.modelA import predict_A, load_model_A
from models.modelB import predict_B, load_model_B
from models.modelC import predict_C, load_model_C
from models.modelD import predict_D, load_model_D
from models.modelE import predict_E, load_model_E

app = Flask(__name__)

UPLOAD_FOLDER = 'temp'
EXTRACT_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}
CACHE_RETENTION_DAYS = 15

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

task_cache = {}

from train_finetune import training_status, train_model

MATERIAL_MODEL_MAP = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E'
}

models_to_load = {
    'A': load_model_A,
    'B': load_model_B,
    'C': load_model_C,
    'D': load_model_D,
    'E': load_model_E
}

print("Initializing models...")
for model_id, load_func in models_to_load.items():
    print(f"Loading Model {model_id}...")
    try:
        load_func()
        print(f"Model {model_id} loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load Model {model_id}: {e}")

@app.route('/')
def index():
    """Render main navigation page.
    
    Returns:
        Rendered index.html template.
    """
    return render_template('index.html')


@app.route('/design')
def design():
    """Render interactive design center.
    
    Returns:
        Rendered design.html template.
    """
    return render_template('design.html')


@app.route('/document')
def document():
    """Render user documentation.
    
    Returns:
        Rendered document.html template.
    """
    return render_template('document.html')


@app.route('/fine_tune')
def fine_tune():
    """Render model fine-tuning interface.
    
    Returns:
        Rendered fine_tune.html template.
    """
    return render_template('fine_tune.html')


@app.route('/test')
def test():
    """Render model fine-tuning interface.
    
    Returns:
        Rendered test.html template.
    """
    return render_template('test.html')


PREDICT_FUNCTIONS = {
    'A': predict_A,
    'B': predict_B,
    'C': predict_C,
    'D': predict_D,
    'E': predict_E
}


def predict_with_sliding_window(B, T, model_id, window_size=1000, real_time=None):
    """Predict magnetic properties using sliding window approach.
    
    Handles B sequences of any length by:
    - Direct prediction for sequences <= window_size
    - Sliding window prediction for longer sequences
    - Proper boundary handling and padding
    
    Args:
        B: Magnetic flux density sequence as numpy array.
        T: Temperature value as float.
        model_id: Model identifier ('A', 'B', 'C', 'D', 'E').
        window_size: Size of sliding window (default 1000).
        real_time: Time period for core loss calculation (default 1.0).
    
    Returns:
        dict: Contains H_pred (predicted H sequence), mu (permeability), 
              and core_loss (calculated core loss).
    
    Raises:
        KeyError: If model_id not in PREDICT_FUNCTIONS.
    """
    B = np.array(B, dtype=np.float32)
    B_len = len(B)
    
    if real_time is None or real_time <= 0:
        real_time = 1.0
    
    if B_len <= window_size:
        H = np.full(window_size, np.nan, dtype=np.float32)
        H[0] = 0.0
        
        if B_len < window_size:
            B_padded = np.pad(B, (0, window_size - B_len), mode='edge')
        else:
            B_padded = B
        
        predict_func = PREDICT_FUNCTIONS[model_id]
        result = predict_func(B_padded, H, T, split_point=1, real_time=real_time)
        
        if B_len < window_size:
            result['H_pred'] = result['H_pred'][:B_len]
            result['mu'] = result['mu'][:B_len]
        
        return result
    
    H_pred_full = []
    mu_full = []
    core_loss_total = 0.0
    step = window_size
    
    for start_idx in range(0, B_len, step):
        end_idx = min(start_idx + window_size, B_len)
        window_B = B[start_idx:end_idx]
        actual_window_len = len(window_B)
        
        if actual_window_len < window_size and end_idx == B_len:
            padding_needed = window_size - actual_window_len
            
            if start_idx >= padding_needed:
                padding_data = B[start_idx - padding_needed:start_idx]
                window_B = np.concatenate([padding_data, window_B])
            else:
                padding_data = B[:start_idx] if start_idx > 0 else np.array([B[0]])
                while len(padding_data) < padding_needed:
                    padding_data = np.concatenate([[B[0]], padding_data])
                window_B = np.concatenate([padding_data[-padding_needed:], window_B])
        
        if len(window_B) < window_size:
            window_B = np.pad(window_B, (0, window_size - len(window_B)), mode='edge')
        
        H = np.full(window_size, np.nan, dtype=np.float32)
        if start_idx == 0:
            H[0] = 0.0
        else:
            H[0] = H_pred_full[-1] if len(H_pred_full) > 0 else 0.0
        
        predict_func = PREDICT_FUNCTIONS[model_id]
        result = predict_func(window_B, H, T, split_point=1)
        
        window_H_pred = result['H_pred']
        window_mu = result['mu']
        
        if actual_window_len < window_size:
            window_H_pred = window_H_pred[:actual_window_len]
            window_mu = window_mu[:actual_window_len]
        
        H_pred_full.extend(window_H_pred)
        mu_full.extend(window_mu)
    
    H_pred_full = H_pred_full[:B_len]
    mu_full = mu_full[:B_len]
    
    if len(B) > 1 and len(H_pred_full) > 1:
        H_pred_array = np.array(H_pred_full)
        dH = np.diff(H_pred_array)
        
        B_array = np.array(B)
        if len(B_array) == len(H_pred_array):
            B_mid = (B_array[:-1] + B_array[1:]) / 2.0
            integral_B_dH = np.sum(B_mid * dH)
        else:
            integral_B_dH = np.sum(B_array[:len(dH)] * dH)
        
        core_loss_total = integral_B_dH / real_time if real_time > 0 else integral_B_dH
    else:
        core_loss_total = 0.0
    
    return {
        'H_pred': H_pred_full,
        'mu': mu_full,
        'core_loss': float(core_loss_total)
    }


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for magnetic property prediction.
    
    Accepts JSON with materialId (1-5) or modelId ('A'-'E') and sample data.
    
    Returns:
        JSON response with H_pred, mu, and core_loss.
    
    Raises:
        400: Invalid input parameters.
        500: Internal server error.
    """
    try:
        data = request.json
        
        if 'materialId' not in data and 'modelId' not in data:
            return jsonify({'error': 'materialId or modelId is required'}), 400
        
        if 'materialId' in data:
            material_id = data['materialId']
            if material_id not in MATERIAL_MODEL_MAP:
                return jsonify({'error': f'Invalid materialId: {material_id}'}), 400
            model_id = MATERIAL_MODEL_MAP[material_id]
        else:
            model_id = data['modelId']
        
        if model_id not in PREDICT_FUNCTIONS:
            return jsonify({'error': f'Model {model_id} not supported'}), 400
        
        if 'sample' not in data:
            return jsonify({'error': 'sample is required'}), 400
        
        sample = data['sample']
        
        if 'B' not in sample or 'T' not in sample:
            return jsonify({'error': 'sample must contain B and T'}), 400
        
        B = sample['B']
        T = sample['T']
        real_time = sample.get('real_time', None)
        
        result = predict_with_sliding_window(B, T, model_id, real_time=real_time)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    """Check if file extension is allowed for upload.
    
    Args:
        filename: Name of file to check.
    
    Returns:
        bool: True if file extension is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_hyperparameters(data):
    """Validate hyperparameters for model fine-tuning.
    
    Args:
        data: Dictionary containing hyperparameters.
    
    Returns:
        list: Error messages if validation fails, empty list otherwise.
    """
    errors = []
    
    epochs = data.get('epochs', 100)
    if not isinstance(epochs, int) or epochs < 10 or epochs > 1000:
        errors.append('epochs must be between 10 and 1000')
    
    learning_rate = data.get('learning_rate', 0.001)
    if not isinstance(learning_rate, (int, float)) or learning_rate < 0.0001 or learning_rate > 0.1:
        errors.append('learning_rate must be between 0.0001 and 0.1')
    
    batch_size = data.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size < 8 or batch_size > 128:
        errors.append('batch_size must be between 8 and 128')
    
    validation_split = data.get('validation_split', 0.2)
    if not isinstance(validation_split, (int, float)) or validation_split < 0.1 or validation_split > 0.4:
        errors.append('validation_split must be between 0.1 and 0.4')
    
    window_stride = data.get('window_size', 500)
    if not isinstance(window_stride, int) or window_stride < 1 or window_stride > 1000:
        errors.append('window_size (stride) must be between 1 and 1000')
    
    loss_type = data.get('loss_type', 'MSE')
    if loss_type not in ['MSE', 'RMSE', 'energy']:
        errors.append('loss_type must be one of: MSE, RMSE, or energy')
    
    frozen_layers = data.get('frozen_layers', 1)
    if not isinstance(frozen_layers, int) or frozen_layers < 0 or frozen_layers > 3:
        errors.append('frozen_layers must be between 0 and 3')
    
    base_model = data.get('base_model', 'A')
    if base_model not in ['A', 'B', 'C', 'D', 'E']:
        errors.append('base_model must be one of: A, B, C, D, or E')
    
    return errors


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint for uploading training data.
    
    Accepts ZIP file containing H.csv, B.csv, and T.csv.
    
    Returns:
        JSON response with task_id, samples, and sequence_length.
    
    Raises:
        400: Invalid file or missing required files.
        500: Internal server error.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Filename is empty'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only ZIP files are accepted'}), 400
        
        task_id = str(uuid.uuid4())
        temp_dir = os.path.join(UPLOAD_FOLDER, task_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        extract_dir = os.path.join(EXTRACT_FOLDER, task_id)
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        required_files = ['H.csv', 'T.csv', 'B.csv']
        missing_files = []
        for req_file in required_files:
            if not os.path.exists(os.path.join(extract_dir, req_file)):
                missing_files.append(req_file)
        
        if missing_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return jsonify({'error': f'Missing required files: {", ".join(missing_files)}'}), 400
        
        try:
            H_data = np.loadtxt(os.path.join(extract_dir, 'H.csv'), delimiter=',')
            T_data = np.loadtxt(os.path.join(extract_dir, 'T.csv'), delimiter=',')
            B_data = np.loadtxt(os.path.join(extract_dir, 'B.csv'), delimiter=',')
            
            if len(H_data.shape) == 1:
                H_data = H_data.reshape(-1, 1)
            if len(B_data.shape) == 1:
                B_data = B_data.reshape(-1, 1)
            if len(T_data.shape) == 1:
                T_data = T_data.reshape(-1, 1)
            
            N_H, M_H = H_data.shape
            N_B, M_B = B_data.shape
            N_T, M_T = T_data.shape
            
            if N_H != N_T or N_H != N_B:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Sample count mismatch: H={N_H}, T={N_T}, B={N_B}'}), 400
            
            if M_H != M_B:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Sequence length mismatch: H={M_H}, B={M_B}'}), 400
            
            if M_T != 1:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                return jsonify({'error': f'Temperature should be (N, 1), got (N, {M_T})'}), 400
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return jsonify({'error': f'Data validation failed: {str(e)}'}), 400
        
        shutil.rmtree(temp_dir, ignore_errors=True)
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
    """API endpoint to start model fine-tuning.
    
    Args:
        JSON with task_id, base_model, and hyperparameters.
    
    Returns:
        JSON response with status and task_id.
    
    Raises:
        400: Invalid parameters or task not found.
        500: Internal server error.
    """
    try:
        data = request.json
        
        if 'task_id' not in data:
            return jsonify({'error': 'task_id is required'}), 400
        
        task_id = data['task_id']
        data_dir = os.path.join(EXTRACT_FOLDER, task_id)
        
        if not os.path.exists(data_dir):
            return jsonify({'error': 'Task not found, please upload file first'}), 400
        
        errors = validate_hyperparameters(data)
        if errors:
            return jsonify({'error': '; '.join(errors)}), 400
        
        if task_id in training_status:
            status = training_status[task_id].get('status', 'unknown')
            if status == 'training':
                return jsonify({'error': 'Task is already training'}), 400
        
        task_cache[task_id] = datetime.now()
        
        base_model = data.get('base_model', 'A')
        epochs = data.get('epochs', 100)
        learning_rate = data.get('learning_rate', 0.001)
        batch_size = data.get('batch_size', 32)
        validation_split = data.get('validation_split', 0.2)
        window_stride = data.get('window_size', 500)
        loss_type = data.get('loss_type', 'MSE')
        frozen_layers = data.get('frozen_layers', 1)
        
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
    """API endpoint to poll training status.
    
    Args:
        task_id: Training task identifier.
    
    Returns:
        JSON response with training status and metrics.
    
    Raises:
        500: Internal server error.
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
    """API endpoint to download training results.
    
    Args:
        task_id: Training task identifier.
        file_type: Type of file to download ('model', 'weights', 'report').
    
    Returns:
        File download response.
    
    Raises:
        400: Invalid file type.
        404: File not found.
        500: Internal server error.
    """
    try:
        if file_type not in ['model', 'weights', 'report']:
            return jsonify({'error': 'Invalid file type'}), 400
        
        output_dir = os.path.join(EXTRACT_FOLDER, task_id)
        
        if file_type == 'model':
            base_model = 'A'
            weights_path = os.path.join(os.path.dirname(__file__), 'weights', f'{base_model}.sd')
            
            if not os.path.exists(weights_path):
                return jsonify({'error': 'Model file not found'}), 404
            
            return send_file(weights_path, as_attachment=True, download_name=f'{base_model}.sd')
        
        elif file_type == 'weights':
            weights_file = os.path.join(output_dir, 'model_weights.sd')
            
            if not os.path.exists(weights_file):
                return jsonify({'error': 'Weights file not found, training may not be complete'}), 404
            
            return send_file(weights_file, as_attachment=True, download_name='model_weights.sd')
        
        elif file_type == 'report':
            report_file = os.path.join(output_dir, 'training_report.pdf')
            
            if not os.path.exists(report_file):
                return jsonify({'error': 'Report file not found, training may not be complete'}), 404
            
            return send_file(report_file, as_attachment=True, download_name='training_report.pdf')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/resources/<path:filename>')
def serve_resource(filename):
    """Serve files from static/resources directory for direct download.

    Args:
        filename: Path to file within static/resources directory.

    Returns:
        File response for direct download.

    Raises:
        404: If file not found.
    """
    return send_from_directory('static/resources', filename, as_attachment=True)

def cleanup_old_files():
    """Clean up files older than CACHE_RETENTION_DAYS.
    
    Removes old training data and temporary files to prevent disk space issues.
    Runs daily at midnight.
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=CACHE_RETENTION_DAYS)
        
        if os.path.exists(EXTRACT_FOLDER):
            for item in os.listdir(EXTRACT_FOLDER):
                item_path = os.path.join(EXTRACT_FOLDER, item)
                if os.path.isdir(item_path):
                    if item in task_cache:
                        if task_cache[item] < cutoff_date:
                            shutil.rmtree(item_path, ignore_errors=True)
                            del task_cache[item]
                            print(f"Cleaned up expired task: {item}")
                    else:
                        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                        if mtime < cutoff_date:
                            shutil.rmtree(item_path, ignore_errors=True)
                            print(f"Cleaned up expired directory: {item}")
        
        print(f"Cleanup completed: {datetime.now()}")
    except Exception as e:
        print(f"Cleanup task failed: {e}")


def run_scheduler():
    """Background scheduler that runs cleanup task daily at midnight."""
    while True:
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            cleanup_old_files()
            time.sleep(60)
        else:
            time.sleep(60)


scheduler_thread = Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()


if __name__ == '__main__':
    app.run(debug=True, port=6008)
