# PyGaze Eye Tracking - Quick Start Guide

## What is PyGaze?

PyGaze is a professional Python library for eye tracking that supports multiple eye tracker hardware systems including EyeLink, EyeTribe, and others. This project provides a high-level wrapper that makes PyGaze easier to use.

## Installation

```bash
# Install PyGaze
pip install python-pygaze

# Install project dependencies
pip install -r requirements.txt
```

## Quick Example

```python
from src.pygaze_tracker import PyGazeTracker

# Initialize (dummy mode for testing)
tracker = PyGazeTracker(tracker='dummy')

# Connect and calibrate
tracker.connect()
tracker.calibrate()

# Start recording
tracker.start_recording()

# Collect gaze samples
for i in range(100):
    sample = tracker.sample()
    if sample and sample.valid:
        print(f"Gaze: ({sample.x}, {sample.y})")

# Cleanup
tracker.stop_recording()
tracker.close()
```

## Running Examples

```bash
# Basic example
python examples/pygaze_basic_example.py

# Event detection
python examples/pygaze_event_detection.py

# Calibration
python examples/pygaze_calibration_example.py

# Gaze-contingent display
python examples/pygaze_gaze_contingent.py

# Cognitive load integration
python examples/pygaze_cognitive_load_integration.py
```

## Project Structure

```
src/pygaze_tracker/
├── pygaze_wrapper.py          # Main wrapper class
├── event_detector.py          # Event detection
└── calibration_manager.py    # Calibration utilities

examples/
├── pygaze_basic_example.py
├── pygaze_event_detection.py
├── pygaze_calibration_example.py
├── pygaze_gaze_contingent.py
└── pygaze_cognitive_load_integration.py
```

## Key Features

✅ **Easy Connection**: Simple connection to various eye trackers  
✅ **Calibration**: Full calibration with validation  
✅ **Event Detection**: Automatic detection of fixations, saccades, blinks  
✅ **Data Analysis**: Built-in analysis tools  
✅ **Integration**: Works with cognitive load calculator  

## Supported Trackers

- **EyeLink** (SR Research)
- **EyeTribe**
- **Dummy Mode** (for testing without hardware)
- **Experimental**: EyeLogic, GazePoint, SMI, Tobii

## Documentation

- **Full Documentation**: See `docs/PYGAZE_PROJECT.md`
- **Examples**: See `examples/README.md`
- **PyGaze Official**: http://www.pygaze.org/

## Next Steps

1. Run the basic example to get familiar
2. Try event detection example
3. Integrate with your project
4. Connect real hardware (if available)

## Citation

If using PyGaze in research:

```
Dalmaijer, E., Mathôt, S., & Van der Stigchel, S. (2014). 
PyGaze: An open-source, cross-platform toolbox for minimal-effort 
programming of eyetracking experiments. 
Behavior Research Methods. 
doi:10.3758/s13428-013-0422-2
```
