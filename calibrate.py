"""
Standalone gaze calibration utility.

Usage:
  python calibrate.py

This runs the 9-point calibration and writes results to `config/gaze_calibration.json`.
"""

import cv2

from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker
from src.utils.gaze_calibration import GazeCalibrator


def main():
    tracker = DirectGazeTracker()
    ok = tracker.setup_mediapipe()
    if not ok:
        raise RuntimeError("Failed to initialize MediaPipe for DirectGazeTracker")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Failed to open camera (index 0)")

    try:
        calibrator = GazeCalibrator(calibration_path="config/gaze_calibration.json")
        success = calibrator.run_calibration(tracker, cam, fullscreen=True)
        if success:
            print("Calibration saved to config/gaze_calibration.json")
            print(f"scale_x={calibrator.scale_x:.3f} scale_y={calibrator.scale_y:.3f}")
            print(f"offset_x={calibrator.offset_x:.3f} offset_y={calibrator.offset_y:.3f}")
        else:
            print("Calibration cancelled or failed.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

