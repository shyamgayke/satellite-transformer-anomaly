# Transformer-Based Satellite Telemetry Anomaly Detection

**Self-contained PyTorch implementation** with attention-based explainability for satellite sensor data.

## Usage

```bash
python anomaly_detection.py
```

The script automatically installs dependencies, generates synthetic telemetry data, trains the model, detects anomalies, and produces publication-ready visualizations.

## Implementation Details

**Dataset**: 500 timesteps across 4 sensors (Temperature, Battery, Gyroscope, Power)  
**Anomalies**: Temperature spike (t=200), battery drift (t=300-350), gyroscope failure (t=400), power spike (t=150)  
**Model**: Transformer autoencoder with positional encoding and 2-head attention  
**Detection**: Reconstruction error threshold (mean + 2σ)  
**Explainability**: Attention weights × reconstruction error for sensor attribution

## Expected Output
Input shape: (480, 20, 4)
Epoch 0, Loss: 1.6794
Epoch 2, Loss: 1.0618
Epoch 4, Loss: 0.9161
Epoch 6, Loss: 0.8936

Anomaly detected at timestep 320
Most contributing sensor: Battery

Explanation for anomaly at index 320:
Anomaly score: 2.5975
Battery contribution: 0.4908
Gyroscope contribution: 0.1846
Temperature contribution: 0.0043
Power contribution: 0.0000

**Generated visualizations**:
1. Telemetry signals with injected anomalies
2. Anomaly detection scores vs. dynamic threshold
3. Attention heatmap for primary anomaly

## Paper Integration

**Section 5.3 Implementation:**
Complete implementation available at:
https://github.com/shyamgayke/satellite-transformer-anomaly

## Dependencies

Auto-installed by script:
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21
- Matplotlib ≥ 3.5

## Files
anomaly_detection.py # Complete self-contained implementation
README.md # This file
