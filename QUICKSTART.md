# Quick Start Guide

## Environment Setup (Already Done!)

Your Python virtual environment has been created and dependencies installed. To activate it:

```bash
source venv/bin/activate
```

## Verify Installation

```bash
python verify_installation.py
```

## Running the System

### Basic Usage

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the main system (from project root directory)
python main.py

# OR run as a module
python -m src.main
```

**Important**: Always run from the project root directory (`/Users/frankenstein/Documents/adaptiveUi`), not from within the `src` directory.

The system will:
1. Initialize camera (default: camera index 0)
2. Start gaze tracking and emotion detection
3. Monitor cognitive load
4. Display real-time results in a window
5. Press 'q' to quit

### Using the Modules Individually

#### Gaze Tracking

```python
from src.data_acquisition.gaze_tracker import GazeTracker
import cv2

tracker = GazeTracker(use_kalman=True)
tracker.setup_mediapipe()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
gaze_coords = tracker.get_gaze(frame)
print(f"Gaze coordinates: {gaze_coords}")
```

#### Emotion Detection

```python
from src.data_acquisition.emotion_detector import EmotionDetector
import cv2

detector = EmotionDetector(model_path='path/to/model.tflite')
detector.setup_tflite()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
emotion = detector.detect_emotion(frame)
print(f"Emotion: {emotion['dominant_emotion']}")
print(f"Negative affect: {emotion['negative_affect_score']}")
```

#### Cognitive Load Monitoring

```python
from src.adaptation_engine.cognitive_load_monitor import CognitiveLoadMonitor

monitor = CognitiveLoadMonitor()
monitor.update_gaze(0.5, 0.5)  # Normalized coordinates
overload_status = monitor.check_cognitive_overload()
print(f"Overload detected: {overload_status['overload_detected']}")
```

#### Metrics Collection

```python
from src.metrics.cognitive_load_metrics import MetricsCollector, NASATLX

collector = MetricsCollector()
task = collector.start_task('task_001')

# Record interactions
collector.record_interaction('click', {'element': 'submit_button'})

# Record errors
collector.record_error('validation_error', {'field': 'email'})

# Complete task
completed = collector.end_task()
print(f"Completion time: {completed.completion_time}s")
print(f"Errors: {completed.error_count}")

# Add NASA-TLX
tlx = NASATLX()
tlx.set_rating('mental_demand', 60)
tlx.set_rating('physical_demand', 20)
tlx.set_rating('temporal_demand', 50)
tlx.set_rating('performance', 70)
tlx.set_rating('effort', 55)
tlx.set_rating('frustration', 40)
tlx.calculate_weighted_score()
collector.add_nasatlx(tlx)

# Export results
collector.export_to_json('results/experiment_001.json')
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## Project Structure Overview

- `src/data_acquisition/` - Gaze tracking and emotion detection
- `src/adaptation_engine/` - Cognitive load monitoring and UI adaptation logic
- `src/metrics/` - NASA-TLX, task completion time, error tracking
- `src/utils/` - Configuration loading and utilities
- `config/` - Configuration files
- `tests/` - Unit tests
- `data/` - Data storage (raw and processed)
- `results/` - Experimental results
- `logs/` - System logs

## Next Steps

1. **Configure the system**: Edit `config/config.yaml` with your settings
2. **Obtain emotion model**: Download or train a TensorFlow Lite model for emotion detection
3. **Run experiments**: Use the metrics collection system for your research
4. **Integrate with mobile app**: Connect this backend to your Flutter/React Native frontend

## Troubleshooting

### Camera not opening
- Check camera permissions
- Try different camera index in `config/config.yaml`
- Verify camera is not being used by another application

### MediaPipe issues
- If MediaPipe fails to import, try: `pip install --upgrade mediapipe`
- The system will use mock data if MediaPipe is unavailable

### TensorFlow Lite model
- The system works without a model (uses mock emotion detection)
- To use real emotion detection, provide path to `.tflite` model file
- Set `emotion_model_path` in `config/config.yaml` or when initializing `EmotionDetector`

