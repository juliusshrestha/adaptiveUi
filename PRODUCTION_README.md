# Production Readiness

This document outlines the production-ready state of the Adaptive UI System.

## Code Organization

### Production Code (`src/`)
- `src/main.py` - Main entry point
- `src/data_acquisition/direct_gaze_tracker.py` - Production gaze tracker (iris-based)
- `src/data_acquisition/emotion_detector.py` - Emotion detection
- `src/adaptation_engine/` - Cognitive load monitoring and UI adaptation
- `src/metrics/` - Metrics collection
- `src/utils/` - Utilities (logging, config, calibration)

### Development Tools (`tools/`)
- Calibration tools for Y-axis tuning
- Testing and debugging scripts
- Analysis tools for log files
- Archived unused implementations

## Configuration

### Default Settings (Production)
- **Logging**: Disabled by default (`log_enabled = False`)
- **Debug Output**: Disabled by default (`debug_output = False`)
- **Smoothing**: Enabled for stable tracking

### Enabling Debug Features
To enable logging or debug output for troubleshooting:

```python
tracker = DirectGazeTracker()
tracker.log_enabled = True  # Enable CSV logging
tracker.debug_output = True  # Enable verbose console output
```

## Calibration

The system uses calibrated parameters for gaze mapping:
- `scale_x = 17.05` - Horizontal sensitivity
- `scale_y = 124.7` - Vertical sensitivity  
- `normalized_y_offset = -0.1034` - Y-axis calibration offset

These values were calibrated using the tools in `tools/calibrate_y_axis.py`.

## Running in Production

```bash
# Activate virtual environment
source venv/bin/activate

# Run the system
python main.py
```

## File Structure

```
adaptiveUi/
├── src/                    # Production code
│   ├── main.py            # Entry point
│   ├── data_acquisition/  # Gaze & emotion detection
│   ├── adaptation_engine/ # Cognitive load & UI adaptation
│   ├── metrics/           # Metrics collection
│   └── utils/             # Utilities
├── tools/                  # Development tools (not used in production)
├── config/                 # Configuration files
├── logs/                   # Log files (gitignored)
└── main.py                 # Convenience entry point
```

## Cleanup Summary

### Removed/Archived
- Unused gaze tracker implementations (moved to `tools/archive/`)
- Test and debug scripts (moved to `tools/`)
- Excessive debug logging (made optional)
- Redundant documentation (consolidated)

### Production Features
- Clean, organized code structure
- Optional logging (disabled by default)
- Optional debug output (disabled by default)
- Proper error handling
- Configuration via config.yaml
