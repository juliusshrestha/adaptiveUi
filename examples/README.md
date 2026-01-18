# PyGaze Examples

This directory contains example scripts demonstrating various PyGaze eye tracking capabilities.

## Examples

### 1. `pygaze_basic_example.py`

**Basic eye tracking operations**

Demonstrates:
- Connecting to an eye tracker
- Performing calibration
- Starting/stopping recording
- Collecting gaze samples
- Proper cleanup

**Usage:**
```bash
python examples/pygaze_basic_example.py
```

**Requirements:**
- PyGaze installed
- Dummy mode (no hardware needed)

---

### 2. `pygaze_event_detection.py`

**Event detection and analysis**

Demonstrates:
- Detecting fixations (gaze stability)
- Detecting saccades (rapid eye movements)
- Detecting blinks
- Analyzing event statistics

**Usage:**
```bash
python examples/pygaze_event_detection.py
```

**Requirements:**
- PyGaze installed
- Dummy mode (no hardware needed)

---

### 3. `pygaze_calibration_example.py`

**Calibration management**

Demonstrates:
- Full calibration procedures
- Drift correction
- Calibration validation
- Saving/loading calibration data

**Usage:**
```bash
python examples/pygaze_calibration_example.py
```

**Requirements:**
- PyGaze installed
- Dummy mode (no hardware needed)

---

### 4. `pygaze_gaze_contingent.py`

**Gaze-contingent display**

Demonstrates:
- Real-time gaze tracking
- Visual feedback (red dot follows gaze)
- Interactive display using pygame

**Usage:**
```bash
python examples/pygaze_gaze_contingent.py
```

**Requirements:**
- PyGaze installed
- pygame (`pip install pygame`)
- Dummy mode (no hardware needed)

**Controls:**
- Move mouse to see dot follow
- Press SPACE to exit

---

### 5. `pygaze_cognitive_load_integration.py`

**Integration with cognitive load calculator**

Demonstrates:
- Real-time cognitive load monitoring
- Combining gaze data with cognitive load metrics
- Event-based load detection
- Visual CLI feedback

**Usage:**
```bash
python examples/pygaze_cognitive_load_integration.py
```

**Requirements:**
- PyGaze installed
- Cognitive load calculator module
- Dummy mode (no hardware needed)

---

## Running Examples

### From Project Root

```bash
# Activate virtual environment (if using one)
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate      # On Windows

# Run an example
python examples/pygaze_basic_example.py
```

### From Examples Directory

```bash
cd examples
python pygaze_basic_example.py
```

## Notes

- All examples use **dummy mode** by default, which simulates eye tracking using mouse movements
- To use real eye trackers, change `tracker='dummy'` to `tracker='eyelink'` or `tracker='eyetribe'` (requires hardware)
- Examples are designed to be educational and demonstrate key concepts
- Modify examples as needed for your specific use case

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root or have the project in your Python path:

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/pygaze_basic_example.py
```

### PyGaze Not Found

Install PyGaze:
```bash
pip install python-pygaze
```

### Pygame Not Found (for gaze-contingent example)

Install pygame:
```bash
pip install pygame
```

## Next Steps

After running the examples:

1. **Modify for your use case**: Adapt examples to your specific needs
2. **Integrate with your project**: Use the wrapper classes in your own code
3. **Connect real hardware**: Replace dummy mode with actual eye tracker
4. **Extend functionality**: Build on top of the provided classes

See `docs/PYGAZE_PROJECT.md` for full API documentation.
