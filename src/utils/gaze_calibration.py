"""
Gaze calibration utilities (9-point calibration).

This module was referenced by `src/main.py` and documented in `docs/CALIBRATION.md`,
but was missing from the repo. It provides:

- `GazeCalibrator.run_calibration(...)`: interactive 9-point calibration
- `GazeCalibrator.apply_calibration(x, y)`: apply saved calibration to normalized gaze coords
- `GazeCalibrator.load_calibration(...)` / `save_calibration(...)`: persistence

Calibration model:
We fit a linear transform per axis: target = a * measured + b.
For compatibility with the docs, we store as (scale, offset) around 0.5:
  calibrated = (measured - 0.5) * scale + 0.5 + offset
which expands to: calibrated = scale * measured + (0.5 + offset - 0.5 * scale)
So: scale = a, offset = b - 0.5 * (1 - a)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np
import cv2


@dataclass
class CalibrationPoint:
    target: Tuple[float, float]
    measured: Tuple[float, float]
    std: Tuple[float, float]


class GazeCalibrator:
    def __init__(self, calibration_path: str = "config/gaze_calibration.json"):
        self.calibration_path = calibration_path

        # Default (identity)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # Optional extra (kept for backward compatibility with older JSON)
        self.normalized_y_offset = 0.0

        self._loaded = False
        self.load_calibration(silent=True)

    @property
    def is_loaded(self) -> bool:
        """True if a calibration file was successfully loaded."""
        return bool(self._loaded)

    def load_calibration(self, path: Optional[str] = None, silent: bool = False) -> bool:
        """Load calibration parameters from JSON file."""
        p = Path(path or self.calibration_path)
        if not p.exists():
            return False

        try:
            data = json.loads(p.read_text())
            self.scale_x = float(data.get("scale_x", self.scale_x))
            self.scale_y = float(data.get("scale_y", self.scale_y))
            self.offset_x = float(data.get("offset_x", self.offset_x))
            self.offset_y = float(data.get("offset_y", self.offset_y))
            self.normalized_y_offset = float(data.get("normalized_y_offset", self.normalized_y_offset))
            self._loaded = True
            return True
        except Exception:
            if not silent:
                raise
            return False

    def save_calibration(self, calibration: Dict[str, Any], path: Optional[str] = None) -> None:
        """Save full calibration JSON to disk."""
        p = Path(path or self.calibration_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(calibration, indent=2, default=str))

    def apply_calibration(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply calibration to normalized gaze coordinates (0..1).
        """
        # Keep inputs sane even if tracker returns slightly out of bounds
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))

        x_cal = (x - 0.5) * self.scale_x + 0.5 + self.offset_x
        y_cal = (y - 0.5) * self.scale_y + 0.5 + self.offset_y

        x_cal = float(max(0.0, min(1.0, x_cal)))
        y_cal = float(max(0.0, min(1.0, y_cal)))
        return x_cal, y_cal

    def _fit_axis(self, measured: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """
        Fit target = a * measured + b. Returns (a, b).
        """
        measured = measured.astype(float)
        target = target.astype(float)

        if measured.size < 2 or float(np.std(measured)) < 1e-6:
            return 1.0, 0.0

        a = float(np.cov(target, measured)[0, 1] / (np.var(measured) + 1e-9))
        b = float(np.mean(target) - a * np.mean(measured))
        return a, b

    def run_calibration(
        self,
        gaze_tracker,
        camera: cv2.VideoCapture,
        *,
        points: Optional[List[Tuple[float, float]]] = None,
        point_duration_sec: float = 2.0,
        warmup_sec: float = 0.5,
        fullscreen: bool = True,
    ) -> bool:
        """
        Interactive 9-point calibration. Displays targets and samples gaze.

        Returns:
            True on success, False if cancelled or insufficient samples.
        """
        if points is None:
            # 3x3 grid with 10% margin
            xs = [0.1, 0.5, 0.9]
            ys = [0.1, 0.5, 0.9]
            points = [(x, y) for y in ys for x in xs]

        # Determine screen size (best effort)
        screen_w, screen_h = 1920, 1080
        try:
            import pyautogui  # type: ignore

            screen_w, screen_h = pyautogui.size()
        except Exception:
            pass

        win = "Calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if fullscreen:
            try:
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass
        cv2.resizeWindow(win, screen_w, screen_h)

        # Wait for SPACE to start
        start_t = time.time()
        while True:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "Press SPACE to start calibration (q to cancel)",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                cv2.destroyWindow(win)
                return False
            if k == ord(" "):
                break
            # safety: if window focus is odd, still keep alive
            if time.time() - start_t > 300:
                cv2.destroyWindow(win)
                return False

        samples: List[CalibrationPoint] = []

        for (tx, ty) in points:
            px = int(tx * (screen_w - 1))
            py = int(ty * (screen_h - 1))

            # Warmup / settle
            warmup_end = time.time() + warmup_sec
            while time.time() < warmup_end:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(canvas, (px, py), 18, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    "Look at the dot",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(win, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    cv2.destroyWindow(win)
                    return False

                # Drain camera frames but don't record yet
                ret, frame = camera.read()
                if not ret:
                    continue
                _ = gaze_tracker.get_gaze(cv2.flip(frame, 1))

            # Collect for this point
            end_t = time.time() + point_duration_sec
            measured: List[Tuple[float, float]] = []
            while time.time() < end_t:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.circle(canvas, (px, py), 18, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.rectangle(canvas, (40, screen_h - 80), (screen_w - 40, screen_h - 50), (60, 60, 60), 1)
                prog = 1.0 - max(0.0, (end_t - time.time()) / max(0.001, point_duration_sec))
                cv2.rectangle(
                    canvas,
                    (40, screen_h - 80),
                    (40 + int((screen_w - 80) * prog), screen_h - 50),
                    (0, 255, 0),
                    -1,
                )
                cv2.imshow(win, canvas)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    cv2.destroyWindow(win)
                    return False

                ret, frame = camera.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                gaze = gaze_tracker.get_gaze(frame)
                if gaze is not None:
                    gx, gy = gaze
                    measured.append((float(gx), float(gy)))

            if len(measured) < 5:
                # Too few samples at this point; abort
                cv2.destroyWindow(win)
                return False

            arr = np.array(measured, dtype=float)
            mx, my = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
            sx, sy = float(np.std(arr[:, 0])), float(np.std(arr[:, 1]))
            samples.append(
                CalibrationPoint(
                    target=(float(tx), float(ty)),
                    measured=(mx, my),
                    std=(sx, sy),
                )
            )

        cv2.destroyWindow(win)

        # Fit linear model for each axis
        target_x = np.array([p.target[0] for p in samples], dtype=float)
        target_y = np.array([p.target[1] for p in samples], dtype=float)
        meas_x = np.array([p.measured[0] for p in samples], dtype=float)
        meas_y = np.array([p.measured[1] for p in samples], dtype=float)

        a_x, b_x = self._fit_axis(meas_x, target_x)
        a_y, b_y = self._fit_axis(meas_y, target_y)

        # Convert to scale/offset around 0.5 (doc-compatible)
        scale_x = float(a_x)
        scale_y = float(a_y)
        offset_x = float(b_x - 0.5 * (1.0 - a_x))
        offset_y = float(b_y - 0.5 * (1.0 - a_y))

        pred_x = a_x * meas_x + b_x
        pred_y = a_y * meas_y + b_y
        errors = np.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)

        calibration_json: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "normalized_y_offset": float(self.normalized_y_offset),
            "avg_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "num_points": int(len(samples)),
            "points": [
                {
                    "target": [p.target[0], p.target[1]],
                    "measured": [p.measured[0], p.measured[1]],
                    "std": [p.std[0], p.std[1]],
                }
                for p in samples
            ],
        }

        # Update live parameters + persist
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._loaded = True

        self.save_calibration(calibration_json)
        return True

