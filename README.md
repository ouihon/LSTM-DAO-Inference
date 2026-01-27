# AI Magnetic Component Design Platform

A comprehensive web-based platform for designing and optimizing magnetic components, particularly for DC-AC converters. This system combines advanced neural network models with intuitive design tools to help engineers create efficient magnetic components.

## Features

- **Interactive Design Center**: Real-time parameter adjustment with comprehensive visualization tools
- **Pre-trained Neural Network Models**: Five pre-trained models for different magnetic materials (3C90, 3C94, 3E6, 3F4, and Other)
- **Model Fine-tuning**: Custom model training and fine-tuning capabilities with transfer learning support
- **Real-time Visualization**: Multiple charts showing magnetic properties, B-H curves, core loss, and more
- **Grid-Connected DC-AC Simulation**: Support for various converter topologies (Single-Phase Full Bridge, 3-Phase Full Bridge, 3-Phase T-Type Three-Level)
- **Export Capabilities**: Export trained models in multiple formats (.sd, .pdf reports)

## System Requirements

- Python 3.7 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Internet connection for initial load
- Recommended: 1920x1080 or higher resolution
- GPU recommended for faster model inference (CUDA-compatible)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ouihon/ai-magnetic-design.git
cd ai-magnetic-design
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- Flask
- PyTorch
- NumPy
- Matplotlib
- Werkzeug

### 3. Prepare Model Weights

Ensure the following files are present in the `weights/` directory:
- `A.sd`, `B.sd`, `C.sd`, `D.sd`, `E.sd` - Pre-trained model weights
- `Normalization_Params_A.json`, `Normalization_Params_B.json`, etc. - Normalization parameters for each model

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:6008` by default.

## Project Structure

```
ai-magnetic-design/
├── app.py                 # Main Flask application
├── train_finetune.py      # Model fine-tuning training script
├── models/                # Model definitions
│   ├── modelA.py         # Model A (3C90 material)
│   ├── modelB.py         # Model B (3C94 material)
│   ├── modelC.py         # Model C (3E6 material)
│   ├── modelD.py         # Model D (3F4 material)
│   └── modelE.py         # Model E (Other materials)
├── templates/            # HTML templates
│   ├── index.html        # Main navigation page
│   ├── design.html       # Design center interface
│   ├── fine_tune.html    # Model fine-tuning interface
│   └── document.html     # User documentation
├── static/               # Static assets (CSS, JS, images)
├── weights/              # Pre-trained model weights
├── uploads/             # Uploaded training data (auto-created)
├── temp/                 # Temporary files (auto-created)
└── README.md            # This file
```

## Usage

### Design Center

1. Navigate to the Design Center from the main menu
2. Adjust core parameters:
   - **R (Outer Radius)**: Outer radius of toroidal core (mm)
   - **r (Inner Radius)**: Inner radius of toroidal core (mm)
   - **d (Height/Thickness)**: Height/thickness of core (mm)
   - **N (Number of Turns)**: Number of wire turns
3. Configure operating conditions:
   - Select topology (Single-Phase, 3-Phase Full Bridge, or 3-Phase T-Type)
   - Set DC voltage, grid voltage, frequency, etc.
4. Select flux region on charts to trigger AI predictions
5. View real-time visualizations of magnetic properties

### Model Fine-tuning

1. Navigate to Model Libraries from the main menu
2. Upload training data:
   - Prepare a ZIP file containing `H.csv`, `B.csv`, and `T.csv`
   - Data format: `H.csv` and `B.csv` should be (N samples, M time steps)
   - `T.csv` should be (N samples, 1) - one temperature value per sample
3. Configure hyperparameters:
   - **Base Model**: Select pre-trained model (A-E)
   - **Epochs**: Number of training epochs (10-1000)
   - **Learning Rate**: Training learning rate (0.0001-0.1)
   - **Batch Size**: Batch size for training (8-128)
   - **Validation Split**: Fraction for validation (0.1-0.4)
   - **Window Stride**: Stride for sliding windows (1-1000)
   - **Loss Function**: MSE, RMSE, or Energy loss
   - **Frozen Layers**: Number of layers to freeze (0-3)
4. Start training and monitor progress
5. Download trained weights and training report

### API Endpoints

#### Prediction API

```http
POST /api/predict
Content-Type: application/json

{
  "materialId": 1,  // 1-5 (A-E) or use "modelId": "A"
  "sample": {
    "B": [0.1, 0.2, ...],  // Magnetic flux density sequence
    "T": 25.0,             // Temperature
    "real_time": 1.0       // Optional: real time period in seconds
  }
}
```

Response:
```json
{
  "H_pred": [0.0, 0.1, ...],  // Predicted magnetic field strength
  "mu": [1000, 1100, ...],   // Permeability sequence
  "core_loss": 0.05          // Core loss value
}
```

#### Upload Training Data

```http
POST /api/upload
Content-Type: multipart/form-data

file: <ZIP file containing H.csv, B.csv, T.csv>
```

Response:
```json
{
  "task_id": "uuid-string",
  "message": "File uploaded successfully",
  "samples": 100,
  "sequence_length": 1000
}
```

#### Start Fine-tuning

```http
POST /api/finetune
Content-Type: application/json

{
  "task_id": "uuid-from-upload",
  "base_model": "A",
  "epochs": 100,
  "learning_rate": 0.001,
  "batch_size": 32,
  "validation_split": 0.2,
  "window_size": 500,
  "loss_type": "MSE",
  "frozen_layers": 1
}
```

#### Poll Training Status

```http
GET /api/poll/<task_id>
```

#### Download Results

```http
GET /api/download/<task_id>/<file_type>
```

Where `file_type` can be:
- `model`: Base model definition
- `weights`: Trained model weights
- `report`: Training report PDF

## Model Architecture

The models use a hybrid architecture combining:
- **LSTM Backbone**: Processes time series of magnetic properties (B, dB, ddB, H, H_mask)
- **PI Parameter Head**: MLP that generates parameters for the ExtendedPI operator
- **ExtendedPI Operator**: Vectorized hysteresis operator for modeling magnetic materials

Key features:
- Fully vectorized (no time loops)
- Temperature-conditioned predictions
- Support for partial H sequences (masking unknown regions)

## Data Format

### Training Data Format

Training data should be provided as a ZIP file containing three CSV files:

1. **H.csv**: Magnetic field strength
   - Shape: (N samples, M time steps) or (M,) for single sample
   - Each row is a time series

2. **B.csv**: Magnetic flux density
   - Shape: (N samples, M time steps) or (M,) for single sample
   - Must match H.csv dimensions

3. **T.csv**: Temperature
   - Shape: (N samples, 1) or (N,) for single column
   - One temperature value per sample

Example:
```
H.csv: 100 rows × 1000 columns (100 samples, 1000 time steps each)
B.csv: 100 rows × 1000 columns (must match H.csv)
T.csv: 100 rows × 1 column (one temperature per sample)
```

## Configuration

### Model Configuration

Models are configured in `app.py`:
- Material-to-model mapping: `MATERIAL_MODEL_MAP`
- Model loading functions: `models_to_load`
- Prediction functions: `PREDICT_FUNCTIONS`

### Training Configuration

Training parameters can be adjusted in `train_finetune.py`:
- Window size: Fixed at 1000 time steps
- Model architectures: Different parameters for different base models
- Loss functions: MSE, RMSE, or Energy loss

## Maintenance

### Automatic Cleanup

The application automatically cleans up old training data:
- Files older than 15 days are removed
- Cleanup runs daily at midnight
- Task cache tracks creation times

### Manual Cleanup

To manually clean up:
```bash
# Remove old uploads
rm -rf uploads/<old-task-id>

# Clear temp files
rm -rf temp/*
```

## Troubleshooting

### Model Loading Issues

- Ensure weight files exist in `weights/` directory
- Check normalization parameter files are present
- Verify file permissions

### Training Issues

- Verify data format matches requirements
- Check data dimensions are consistent
- Ensure sufficient disk space for outputs
- Monitor GPU memory if using CUDA

### Prediction Issues

- Verify input B sequence length (can be any length, will use sliding window)
- Check temperature value is within valid range
- Ensure model is loaded before prediction

## Development

### Code Structure

- **app.py**: Flask routes and API endpoints
- **train_finetune.py**: Training logic and metrics calculation
- **models/model*.py**: Model definitions and prediction functions
- **templates/**: HTML frontend templates

### Adding New Models

1. Create new model file in `models/` directory
2. Implement `load_model_X()` and `predict_X()` functions
3. Add to `MODEL_CLASSES` in `train_finetune.py`
4. Register in `app.py` (models_to_load, PREDICT_FUNCTIONS)

### Extending Functionality

- Add new API endpoints in `app.py`
- Extend training metrics in `train_finetune.py`
- Add visualization features in HTML templates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

Magnetic Design Team

## Acknowledgments

- PyTorch team for deep learning framework
- Flask team for web framework
- Chart.js for visualization components

## Support

For issues and questions, please open an issue in the repository or contact the development team.

