## Getting Started

This project has two “modes” of running:

- **Desktop app**: `python main.py` (opens an OpenCV window)
- **Backend server for the extension**: `python -m src.server.run_server` (WebSocket server)

### Supported Python

For reliable installs on macOS:

- **Recommended**: Python **3.10 / 3.11**
- **Not recommended** (often missing wheels): Python **3.13** (OpenCV/MediaPipe may not install)

### Install (recommended: conda)

```bash
conda env create -f environment.yml
conda activate adaptiveui
```

### Install (venv, if you already have Python 3.10/3.11)

```bash
./setup_env.sh
source venv/bin/activate
```

### Verify installation

```bash
python verify_installation.py
```

### Run the desktop app

```bash
python main.py
```

- Press **`q`** to quit
- Press **`c`** to calibrate:
  - **Direct gaze mode**: runs 9-point calibration (writes `config/gaze_calibration.json`)
  - **Monitor-plane mode**: runs center calibration (yaw/pitch offsets)

### Run the WebSocket server (for the browser extension)

```bash
python -m src.server.run_server
```

Then load the extension (see `docs/EXTENSION.md`).

### Configure gaze tracking mode

Edit `config/config.yaml`:

- **Direct** (default): `gaze_tracking.mode: direct`
- **Monitor-plane**: `gaze_tracking.mode: monitor_plane`

Monitor-plane settings live under `gaze_tracking.monitor_plane`.

### Using the modules directly (quick snippets)

Gaze:

```python
import cv2
from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker

tracker = DirectGazeTracker()
tracker.setup_mediapipe()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
print(tracker.get_gaze(frame))
```

Cognitive load:

```python
from src.adaptation_engine.cognitive_load_monitor import CognitiveLoadMonitor

monitor = CognitiveLoadMonitor()
monitor.update_gaze(0.5, 0.5)
print(monitor.check_cognitive_overload())
```


