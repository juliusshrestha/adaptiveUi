# Browser Extension (Chrome) – Production Setup

## What it does

The extension connects to the local backend WebSocket server and:

- Applies UI adaptations on web pages (simplify / guidance / layout)
- Shows real-time cognitive load in the popup
- Shows a gaze indicator overlay (based on backend gaze coordinates)

## Install (unpacked)

1. Open Chrome → `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the folder `extension/`

## Run the backend server

From the project root:

```bash
python -m src.server.run_server
```

Default server address: `ws://127.0.0.1:8765`

## Use the new Monitor Plane feature from the popup

In the extension popup:

- **Gaze Mode**:
  - `Direct`: default tracker (supports 9-point calibration in the desktop app)
  - `Monitor Plane`: monitor-plane mapping (ported monitor tracking)
- **Center Calibrate**:
  - Enabled only when `Monitor Plane` is selected
  - Look at the center of your monitor and click **Calibrate**

## Notes

- The backend must be running for the popup to show “Connected”.
- If the extension says “Disconnected”, start the server and click **Reconnect**.

