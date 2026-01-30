# AI Magnetic Component Design Platform

A comprehensive web-based platform for designing and optimizing magnetic components, particularly for DC-AC converters. This system combines advanced neural network models with intuitive design tools to help engineers create efficient magnetic components. We have developed a [online website](https://u2802747-b031-2c6ccc63.westb.seetacloud.com:8443) to simulate and analyse, or you can try to deploy this in your own machine. 

## Features

- **Interactive Design Center**: Real-time parameter adjustment with comprehensive visualization tools
- **Core Loss Analysis**: Upload B data, select materials, and compare core loss predictions across different magnetic materials
- **Pre-trained Neural Network Models**: Five pre-trained models for different magnetic materials (3C90, 3C94, 3E6, 3F4, and Other)
- **Model Fine-tuning**: Custom model training and fine-tuning capabilities with transfer learning support
- **Real-time Visualization**: Multiple charts showing magnetic properties, B-H curves, core loss, and more
- **Grid-Connected DC-AC Simulation**: Support for various converter topologies (Single-Phase Full Bridge, 3-Phase Full Bridge, 3-Phase T-Type Three-Level)
- **Export Capabilities**: Export trained models in multiple formats (.sd, .pdf reports)

## System Requirements

- Python 3.10 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Internet connection for initial load
- Recommended: 1920x1080 or higher resolution
- GPU recommended for faster model inference (CUDA-compatible)

## Installation

### 0. (Optional) Create a virtual python environment using Conda, Virtualenv or something

```bash
conda create -n magica python=3.10
conda activate magica
```

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
│   ├── test.html         # Core loss analysis interface
│   └── document.html     # User documentation
├── static/               # Static assets (CSS, JS, images)
├── weights/              # Pre-trained model weights
├── uploads/             # Uploaded training data (auto-created)
├── temp/                 # Temporary files (auto-created)
└── README.md            # This file
```

## Usage

For comprehensive documentation with detailed instructions, examples, and troubleshooting, visit the [User Documentation](http://localhost:6008/document) page after starting the application.

### Quick Start Guide

#### Design Center
1. Navigate to the Design Center from the main menu
2. Adjust core parameters (R, r, d, N) and observe real-time calculations for Ae, Le, and L
3. Configure operating conditions: topology, voltages, frequencies, filter parameters
4. Select materials and view magnetic properties in synchronized charts
5. Select flux regions to trigger AI predictions for B-H curves

#### Core Loss Analysis
1. Navigate to the Test interface from the main menu
2. Upload B.csv file containing magnetic flux density data (single line, comma-separated values in Tesla)
3. Adjust temperature slider (0-100°C) to set operating temperature
4. Select magnetic materials for comparison (3C90, 3C94, 3E6, 3F4, Other)
5. View real-time AI predictions for:
   - Core loss comparison (W/m³) across selected materials
   - Magnetic flux density B (T) waveform
   - Field strength H (A/m) predictions

#### Grid-Connected DC-AC Simulation
The platform includes a comprehensive time-domain simulation model for grid-connected inverters with:
- Three supported topologies: Single-Phase Full Bridge, 3Phase Full Bridge, 3Phase T-Type Three-Level
- Power-based PI controller with adjustable Kp and Ki gains
- Optional harmonic injection (3rd, 5th, 7th harmonics)
- Synchronized chart visualization with zoom/pan capabilities
- Flux selection for AI prediction triggering

#### Model Libraries
Access pre-trained neural network models for five magnetic materials:
- **3C90**: High-frequency ferrite material
- **3C94**: Advanced soft magnetic material with low core loss  
- **3E6**: Non-crystalline magnetic material
- **3F4**: Composite material with distributed air gaps
- **Custom Model**: User-trained models for specific applications

#### Model Training & Fine-tuning
1. Upload training data as ZIP containing H.csv, B.csv, T.csv
2. Select base model (A-E) and configure hyperparameters
3. Monitor real-time training metrics (loss curves, RMSE, Energy Loss)
4. Download trained weights (.sd) and comprehensive PDF report

#### Core Loss Analysis Interface (test.html)
The test.html interface provides a specialized tool for analyzing core loss across different magnetic materials:

**Key Features:**
- **B Data Upload**: Upload single-line CSV files containing magnetic flux density (B) data in Tesla
- **Temperature Control**: Adjust operating temperature from 0-100°C with real-time updates
- **Material Selection**: Compare up to 5 magnetic materials (3C90, 3C94, 3E6, 3F4, Other)
- **Real-time AI Predictions**: Automatic H field prediction and core loss calculation
- **Visualization**: Three synchronized charts showing:
  - Core loss comparison (W/m³) across selected materials
  - Magnetic flux density B (T) waveform
  - Predicted field strength H (A/m) for each material

**Core Loss Calculation:**
The interface uses numerical integration of the hysteresis loop (∮ H dB) to calculate core loss density:
- Sampling frequency: 16 MHz (default)
- Trapezoidal rule integration for energy calculation
- Power density conversion: W/m³ = Energy (J/m³) / Time (s)

**Data Format:**
- B.csv: Single line of comma-separated values representing B in Tesla
- Example: `0.1,0.2,0.15,0.25,...`

### Detailed Documentation Sections
The User Documentation page provides comprehensive coverage of:
- **Getting Started**: System requirements and key features
- **Design Center**: Core parameters, visualization tools, chart synchronization
- **Core Loss Analysis**: B data upload, material selection, AI prediction workflow
- **Grid-Connected DC-AC Simulation**: System architecture, PWM modeling, power control
- **Model Libraries**: Pre-trained model specifications and selection
- **Parameters & Settings**: Valid parameter ranges and model settings
- **Model Training**: Dataset format, hyperparameters, training metrics
- **Export & Integration**: Model weights and training report export
- **Troubleshooting**: Common issues and solutions

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
  "core_loss": 0.05          // Core loss value (W/m³)
}
```

**Note**: The test.html interface uses the same `/api/predict` endpoint but includes additional client-side core loss calculation using the `calculateCoreLoss()` function which computes core loss density (W/m³) from B and H data using numerical integration of the hysteresis loop.

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

