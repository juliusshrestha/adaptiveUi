# Adaptive UI System for Cognitive Load Mitigation

Research project implementing an adaptive User Interface (UI) designed to mitigate user cognitive load through real-time eye-gaze tracking and emotion detection.

## Overview

This system implements a **Sense-Analyze-Adapt** loop to manage cognitive load effectively:

1. **Sense**: Capture eye gaze coordinates and detect facial expressions
2. **Analyze**: Monitor for cognitive overload triggers (gaze wander, fixation duration, negative affect)
3. **Adapt**: Trigger UI modifications (simplification, guidance, layout reorganization)

## System Architecture

### Data Acquisition (Sensing)
- **Eye Gaze Input**: Mobile front-facing camera captures gaze coordinates (x, y) with Kalman Filter smoothing
- **Emotion Input**: Video frames processed via lightweight CNN to detect stress, frustration, or confusion

### Adaptation Logic
Monitors for cognitive overload triggers:
- **Gaze Wander**: Excessive scanning without interaction
- **Fixation Duration**: Unusually long gazes on specific UI elements
- **Negative Affect**: Detection of frustration through facial analysis

Upon threshold detection, triggers UI changes:
- Simplification: Hiding non-essential elements
- Guidance: Highlighting next steps or providing tooltips
- Layout Reorganization: Increasing whitespace or font size

### Metrics Collection
- **NASA-TLX**: Subjective workload assessment
- **Task Completion Time (TCT)**: Efficiency measurement
- **Error Frequency**: Incorrect interactions per session
- **Pupillary Dilation**: Physiological marker (optional, requires hardware)

## Project Structure

```
adaptiveUi/
├── src/
│   ├── data_acquisition/      # Gaze tracking and emotion detection
│   │   ├── gaze_tracker.py
│   │   └── emotion_detector.py
│   ├── signal_processing/     # Kalman Filter and EKF
│   ├── adaptation_engine/     # Cognitive load monitoring and UI adaptation
│   │   ├── cognitive_load_monitor.py
│   │   └── ui_adapter.py
│   ├── metrics/               # Metrics collection
│   │   └── cognitive_load_metrics.py
│   └── main.py                # Main system entry point
├── data/
│   ├── raw/                   # Raw data (if stored)
│   └── processed/             # Processed data
├── results/                   # Experimental results
├── logs/                      # System logs
├── notebooks/                 # Jupyter notebooks for analysis
├── config/                    # Configuration files
│   └── config.yaml
├── requirements.txt           # Python dependencies
├── setup_env.sh              # Environment setup script
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Make setup script executable
chmod +x setup_env.sh

# Run setup script
./setup_env.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

Run the verification script to check all dependencies:

```bash
python verify_installation.py
```

**Note**: If MediaPipe installation fails with a syntax error during compilation, the package may still be functional. The error occurs in test files and doesn't affect the main functionality. You can verify MediaPipe works by running `python -c "import mediapipe; print(mediapipe.__version__)"`.

### 3. Configure System

Edit `config/config.yaml` to set:
- Camera index
- Emotion detection model path (if using TensorFlow Lite model)
- Cognitive load thresholds
- Adaptation parameters

### 4. Run System

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run main system (from project root directory)
python main.py

# OR run directly from src (will auto-fix imports)
python -m src.main
```

## Technology Stack

- **Frontend**: Flutter or React Native (for mobile application layer)
- **Gaze Tracking**: MediaPipe Face Mesh
- **Emotion Recognition**: TensorFlow Lite for on-device real-time classification
- **Signal Processing**: Python (NumPy, SciPy) for Extended Kalman Filter
- **Data Analysis**: Pandas, scikit-learn, statsmodels

## Research Design

- **Design**: Within-subject experimental design
- **Conditions**: 
  - Static UI (Control)
  - Adaptive UI (Experimental)
- **Sample Size**: N=30 university students
- **Tasks**: High-load tasks (multi-step financial forms, complex navigation)

## Ethical Considerations

- All processing occurs on-device in real-time
- No raw video footage stored
- Informed consent required
- Biometric data privacy maintained

## Usage Examples

### Basic Usage

```python
from src.main import AdaptiveUISystem

# Initialize system
system = AdaptiveUISystem(
    camera_index=0,
    use_kalman=True,
    emotion_model_path='path/to/model.tflite'
)

# Run system
system.run(display=True)
```

### Metrics Collection

```python
from src.metrics.cognitive_load_metrics import MetricsCollector, NASATLX

# Start task
collector = MetricsCollector()
task = collector.start_task('task_001')

# Record interactions and errors
collector.record_interaction('click', {'element': 'button'})
collector.record_error('wrong_input', {'field': 'email'})

# Complete task
completed_task = collector.end_task()

# Add NASA-TLX
tlx = NASATLX()
tlx.set_rating('mental_demand', 60)
tlx.set_rating('physical_demand', 20)
# ... set other ratings
tlx.calculate_weighted_score()
collector.add_nasatlx(tlx)

# Export results
collector.export_to_json('results/experiment_001.json')
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

## License

Research project - See LICENSE file for details.

## Authors

Research Team

## Acknowledgments

This project implements the research methodology outlined in Chapter 3 of the Adaptive UI research study.

