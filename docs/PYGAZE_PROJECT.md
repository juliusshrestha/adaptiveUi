# PyGaze Eye Tracking Project

A comprehensive Python project for eye tracking using PyGaze, a professional eye tracking library that supports multiple eye tracker hardware systems.

## Overview

This project provides a high-level wrapper around PyGaze that simplifies common eye tracking operations including:

- **Connection Management**: Easy connection to various eye tracker types
- **Calibration**: Full calibration procedures with validation
- **Recording**: Start/stop recording with automatic data logging
- **Event Detection**: Automatic detection of fixations, saccades, and blinks
- **Data Analysis**: Built-in analysis tools for eye tracking metrics

## Supported Eye Trackers

PyGaze supports the following eye trackers:

### Fully Supported
- **EyeLink** - SR Research EyeLink systems
- **EyeTribe** - The EyeTribe tracker

### Experimental Support
- **EyeLogic** - EyeLogic eye trackers
- **GazePoint / OpenGaze** - GazePoint eye trackers
- **SMI** - SensoMotoric Instruments eye trackers
- **Tobii** - Tobii eye trackers

### Testing/Dummy Modes
- **Simple Dummy** - Does nothing (for testing without hardware)
- **Advanced Dummy** - Mouse simulation of eye movements (for development)

## Installation

### 1. Install PyGaze

```bash
# Using pip
pip install python-pygaze

# Using conda
conda install python-pygaze -c cogsci
```

### 2. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Eye Tracker Drivers (if using hardware)

For EyeLink:
- Install EyeLink Developer Kit from SR Research
- Ensure `edfapi` is accessible

For EyeTribe:
- Install EyeTribe SDK
- Ensure tracker is connected via USB

## Project Structure

```
src/pygaze_tracker/
├── __init__.py                 # Module exports
├── pygaze_wrapper.py          # Main PyGaze wrapper class
├── event_detector.py           # Event detection and analysis
└── calibration_manager.py     # Calibration utilities

examples/
├── pygaze_basic_example.py           # Basic usage example
├── pygaze_event_detection.py         # Event detection example
├── pygaze_calibration_example.py     # Calibration example
└── pygaze_gaze_contingent.py         # Gaze-contingent display
```

## Quick Start

### Basic Usage

```python
from src.pygaze_tracker import PyGazeTracker

# Initialize tracker (dummy mode for testing)
tracker = PyGazeTracker(tracker='dummy')

# Connect
tracker.connect()

# Calibrate
tracker.calibrate()

# Start recording
tracker.start_recording()

# Collect samples
for i in range(100):
    sample = tracker.sample()
    if sample and sample.valid:
        print(f"Gaze: ({sample.x}, {sample.y})")

# Stop recording
tracker.stop_recording()

# Close
tracker.close()
```

### Event Detection

```python
from src.pygaze_tracker import PyGazeTracker, EventDetector

tracker = PyGazeTracker(tracker='dummy')
tracker.connect()
tracker.calibrate()
tracker.start_recording()

# Initialize event detector
detector = EventDetector(tracker)

# Collect a fixation
fixation = detector.collect_fixation(timeout=5.0, min_duration=0.2)
if fixation:
    print(f"Fixation: {fixation.duration:.3f}s at ({fixation.x}, {fixation.y})")

# Collect a saccade
saccade = detector.collect_saccade(timeout=5.0)
if saccade:
    print(f"Saccade: amplitude={saccade.amplitude:.1f}, velocity={saccade.velocity:.1f}")

# Analyze all events
fixation_stats = detector.analyze_fixations()
print(f"Total fixations: {fixation_stats['count']}")
print(f"Mean duration: {fixation_stats['mean_duration']:.3f}s")

tracker.stop_recording()
tracker.close()
```

### Calibration Management

```python
from src.pygaze_tracker import PyGazeTracker, CalibrationManager

tracker = PyGazeTracker(tracker='dummy')
tracker.connect()

# Initialize calibration manager
manager = CalibrationManager(tracker, calibration_file='calibration.json')

# Perform calibration with validation
if manager.perform_calibration(validate=True, max_attempts=3):
    print("Calibration successful!")
    
    # Save calibration
    manager.save_calibration()
    
    # Later, load calibration
    manager.load_calibration()

tracker.close()
```

## API Reference

### PyGazeTracker

Main wrapper class for PyGaze functionality.

#### Methods

- `connect() -> bool`: Connect to eye tracker
- `calibrate() -> bool`: Perform calibration
- `drift_correction(pos=None, fix_triggered=False) -> bool`: Perform drift correction
- `start_recording() -> bool`: Start recording eye tracking data
- `stop_recording() -> bool`: Stop recording
- `sample() -> GazeSample`: Get most recent gaze sample
- `wait_for_fixation_start() -> Tuple[float, Tuple[float, float]]`: Wait for fixation start
- `wait_for_fixation_end() -> Tuple[float, Tuple[float, float]]`: Wait for fixation end
- `wait_for_saccade_start() -> Tuple[float, Tuple[float, float]]`: Wait for saccade start
- `wait_for_saccade_end() -> Tuple[float, Tuple[float, float], Tuple[float, float]]`: Wait for saccade end
- `wait_for_blink_start() -> float`: Wait for blink start
- `wait_for_blink_end() -> float`: Wait for blink end
- `log(msg: str)`: Log message to eye tracker log file
- `log_var(var: str, val: Any)`: Log variable to log file
- `close()`: Close connection to eye tracker

### EventDetector

High-level event detection and analysis.

#### Methods

- `collect_fixation(timeout=None, min_duration=None) -> Fixation`: Collect a fixation event
- `collect_saccade(timeout=None) -> Saccade`: Collect a saccade event
- `collect_blink(timeout=None) -> Blink`: Collect a blink event
- `sample_continuous(duration, sample_rate=60.0, callback=None) -> List[GazeSample]`: Continuously sample gaze
- `analyze_fixations(window_start=None, window_end=None) -> dict`: Analyze fixation statistics
- `analyze_saccades(window_start=None, window_end=None) -> dict`: Analyze saccade statistics
- `analyze_blinks(window_start=None, window_end=None) -> dict`: Analyze blink statistics
- `clear_buffer()`: Clear all event buffers

### CalibrationManager

Manages calibration procedures and data persistence.

#### Methods

- `perform_calibration(validate=True, max_attempts=3) -> bool`: Perform full calibration
- `perform_drift_correction(pos=None, fix_triggered=False, max_attempts=3) -> bool`: Perform drift correction
- `validate_calibration(validation_points=None, tolerance=2.0) -> bool`: Validate calibration accuracy
- `save_calibration(filepath=None) -> bool`: Save calibration data to file
- `load_calibration(filepath=None) -> bool`: Load calibration data from file
- `get_calibration_info() -> dict`: Get information about current calibration

## Data Structures

### GazeSample

```python
@dataclass
class GazeSample:
    x: float              # X coordinate
    y: float              # Y coordinate
    timestamp: float      # Timestamp (ms from start)
    pupil_size: Optional[float]  # Pupil size (if available)
    valid: bool           # Whether sample is valid
```

### Fixation

```python
@dataclass
class Fixation:
    start_time: float                    # Start time (ms)
    end_time: float                      # End time (ms)
    duration: float                      # Duration (seconds)
    x: float                             # Fixation center X
    y: float                             # Fixation center Y
    start_pos: Tuple[float, float]       # Start position
    end_pos: Tuple[float, float]         # End position
```

### Saccade

```python
@dataclass
class Saccade:
    start_time: float                    # Start time (ms)
    end_time: float                      # End time (ms)
    duration: float                      # Duration (seconds)
    start_pos: Tuple[float, float]      # Start position
    end_pos: Tuple[float, float]         # End position
    amplitude: float                     # Saccade amplitude (pixels)
    velocity: float                      # Saccade velocity (pixels/s)
```

### Blink

```python
@dataclass
class Blink:
    start_time: float     # Start time (ms)
    end_time: float       # End time (ms)
    duration: float       # Duration (seconds)
```

## Examples

### Example 1: Basic Eye Tracking

See `examples/pygaze_basic_example.py` for a complete example of:
- Connecting to an eye tracker
- Performing calibration
- Recording and collecting samples
- Proper cleanup

### Example 2: Event Detection

See `examples/pygaze_event_detection.py` for:
- Detecting fixations, saccades, and blinks
- Analyzing event statistics
- Working with event buffers

### Example 3: Calibration Management

See `examples/pygaze_calibration_example.py` for:
- Full calibration procedures
- Drift correction
- Calibration validation
- Saving/loading calibration data

### Example 4: Gaze-Contingent Display

See `examples/pygaze_gaze_contingent.py` for:
- Real-time gaze-contingent display
- Visual feedback of gaze position
- Interactive applications

## Integration with Cognitive Load Calculator

The PyGaze tracker can be integrated with the existing cognitive load calculator:

```python
from src.pygaze_tracker import PyGazeTracker
from src.metrics.cognitive_load_calculator import CognitiveLoadCalculator

# Initialize components
tracker = PyGazeTracker(tracker='dummy')
tracker.connect()
tracker.calibrate()
tracker.start_recording()

calculator = CognitiveLoadCalculator()

# Collect gaze data and update calculator
while True:
    sample = tracker.sample()
    if sample and sample.valid:
        calculator.update_gaze(
            x=sample.x,
            y=sample.y,
            pupil_size=sample.pupil_size
        )
    
    # Calculate CLI periodically
    if calculator.should_calculate_cli():
        result = calculator.calculate_cli()
        if result:
            print(f"CLI: {result['cli']:.2f} ({result['load_level']})")
```

## Configuration

### Tracker Selection

```python
# Dummy mode (for testing)
tracker = PyGazeTracker(tracker='dummy')

# EyeLink
tracker = PyGazeTracker(tracker='eyelink')

# EyeTribe
tracker = PyGazeTracker(tracker='eyetribe')
```

### Event Detection Settings

```python
tracker = PyGazeTracker(
    tracker='dummy',
    eventdetection='pygaze',  # or 'native'
    saccvelthresh=35,         # Saccade velocity threshold
    saccaccthresh=9500,       # Saccade acceleration threshold
    fixtresh=1.5,             # Fixation threshold (degrees)
    blinksize=150             # Blink size threshold
)
```

## Troubleshooting

### Connection Issues

- **EyeLink**: Ensure EyeLink Developer Kit is installed and tracker is connected
- **EyeTribe**: Check USB connection and install EyeTribe SDK
- **Dummy Mode**: Always works, use for testing without hardware

### Calibration Issues

- Ensure good lighting conditions
- Keep head still during calibration
- Use fixation-triggered drift correction if available
- Increase `max_attempts` if calibration fails

### Event Detection Issues

- Adjust thresholds based on your setup
- Use 'native' detection if available for your tracker
- Check that recording is started before detecting events

## Citation

If you use PyGaze in your research, please cite:

```
Dalmaijer, E., Mathôt, S., & Van der Stigchel, S. (2014). 
PyGaze: An open-source, cross-platform toolbox for minimal-effort 
programming of eyetracking experiments. 
Behavior Research Methods. 
doi:10.3758/s13428-013-0422-2
```

## Resources

- **PyGaze Documentation**: http://www.pygaze.org/
- **PyGaze GitHub**: https://github.com/esdalmaijer/PyGaze
- **EyeLink Documentation**: https://www.sr-research.com/support/thread-13.html
- **EyeTribe Documentation**: https://theeyetribe.com/

## License

This project is part of the Adaptive UI research project. See main project LICENSE for details.
