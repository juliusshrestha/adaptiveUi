# Adaptive UI (Cognitive Load) – Backend + Browser Extension

Adaptive UI system that estimates **cognitive load** from gaze/emotion/mouse signals and applies **real-time UI adaptations** in the browser via a Chrome extension.

## What you get

- **Gaze tracking**
  - `direct`: iris-based normalized gaze (default)
  - `monitor_plane`: ported “monitor tracking” style mapping + center calibration
- **Cognitive Load Index (CLI)**: windowed metrics + weighted score
- **WebSocket server**: streams gaze + CLI + adaptation commands
- **Chrome extension**: applies simplification/guidance/layout changes to web pages

## Repository layout

```
adaptiveUi/
├── src/                       # Python backend (Sense → Analyze → Adapt)
├── config/                    # Runtime config (config.yaml, gaze_calibration.json)
├── extension/                 # Chrome extension (MV3)
├── docs/                      # Consolidated docs
├── requirements/              # base/full/dev dependency sets
├── tools/                     # Dev/debug utilities (not required for production)
├── research/                  # Notebooks / research artifacts
├── main.py                    # Desktop runner (OpenCV display)
└── setup_env.sh               # venv bootstrap (minimal install)
```

## Supported Python

On macOS, use **Python 3.10 / 3.11**.

Python **3.13** frequently fails due to missing wheels for **OpenCV** / **MediaPipe**.

## Install (recommended: conda)

```bash
conda env create -f environment.yml
conda activate adaptiveui
```

## Install (venv)

Requires Python 3.10/3.11 on your PATH:

```bash
./setup_env.sh
source venv/bin/activate
```

## Run

### Desktop app (visual)

```bash
python main.py
```

- Press **`q`** to quit
- Press **`c`** to calibrate (direct mode: 9-point; monitor-plane: center)

### WebSocket server (for the extension)

```bash
python -m src.server.run_server
```

Then load the Chrome extension from `extension/`. Full steps: `docs/EXTENSION.md`.

## Configure gaze mode

In `config/config.yaml`:

- `gaze_tracking.mode: direct`
- `gaze_tracking.mode: monitor_plane`

When using the extension popup:

- Switch **Gaze Mode** to `Monitor Plane`
- Use **Center Calibrate** (look at center → click calibrate)

## Docs

- `docs/GETTING_STARTED.md`
- `docs/EXTENSION.md`
- `docs/CALIBRATION.md`
- `docs/WEBSOCKET_API.md`


