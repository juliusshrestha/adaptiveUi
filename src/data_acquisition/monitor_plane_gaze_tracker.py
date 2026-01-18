"""
Monitor-plane style gaze tracker (ported from Webcam3DTracker/MonitorTracking.py).

Goal:
- Compute a stable 3D gaze direction from MediaPipe FaceMesh landmarks
- Map that direction to monitor (x, y) coordinates using yaw/pitch mapping
- Return normalized gaze coordinates (0..1) that can be consumed by CognitiveLoadMonitor

This is intentionally a *minimal* port:
- No debug orbit window, no marker stamping, no mouse control
- Designed to be used inside AdaptiveUISystem.process_frame()
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple
import math

import numpy as np
import cv2
import mediapipe as mp


@dataclass
class MonitorGazeConfig:
    """Configuration for monitor-plane gaze mapping."""

    # Degrees at which left/right edge is reached (tune per setup)
    yaw_degrees: float = 15.0
    pitch_degrees: float = 5.0

    # Smoothing
    direction_filter_length: int = 15
    screen_smoothing_length: int = 8

    # Monitor size fallback (used if auto-detect fails)
    monitor_width: int = 1920
    monitor_height: int = 1080

    # Auto calibration for eye spheres (locks after N valid frames)
    auto_lock_eye_spheres: bool = True
    auto_lock_frames_required: int = 20


class MonitorPlaneGazeTracker:
    """
    MediaPipe FaceMesh gaze tracker that outputs *monitor* gaze coordinates.

    Public API mirrors DirectGazeTracker:
    - setup_mediapipe()
    - get_gaze(frame) -> Optional[(x_norm, y_norm)]
    - calibrate_center() to set yaw/pitch offsets so current gaze maps to center
    - reset_calibration()
    """

    # Nose-only landmark indices used for a stable PCA head frame
    _NOSE_INDICES = [
        4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
        461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
        3, 248,
    ]

    def __init__(self, config: Optional[MonitorGazeConfig] = None):
        self.config = config or MonitorGazeConfig()

        # MediaPipe FaceMesh
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self.initialized = False

        # Monitor size (best-effort autodetect)
        self.monitor_width, self.monitor_height = self._detect_monitor_size()
        self.center_x = self.monitor_width // 2
        self.center_y = self.monitor_height // 2

        # Calibration offsets (yaw/pitch) + manual pixel offsets
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.manual_offset_x_px = 0
        self.manual_offset_y_px = 0

        # Stabilize PCA eigenvector sign flips
        self._R_ref_nose: Optional[np.ndarray] = None

        # Eye-sphere calibration state (ported concept)
        self._left_sphere_locked = False
        self._right_sphere_locked = False
        self._left_sphere_local_offset: Optional[np.ndarray] = None
        self._right_sphere_local_offset: Optional[np.ndarray] = None
        self._left_calibration_nose_scale: Optional[float] = None
        self._right_calibration_nose_scale: Optional[float] = None
        self._base_radius = 20.0  # world units (in the same units as landmark scaling)

        # Auto-lock support
        self._valid_frame_count = 0

        # Smoothing buffers
        self._combined_gaze_directions: Deque[np.ndarray] = deque(maxlen=self.config.direction_filter_length)
        self._screen_position_history: Deque[Tuple[int, int]] = deque(maxlen=self.config.screen_smoothing_length)

    def _detect_monitor_size(self) -> Tuple[int, int]:
        """Best-effort monitor size detection (falls back to config)."""
        try:
            import pyautogui  # type: ignore

            w, h = pyautogui.size()
            if isinstance(w, int) and isinstance(h, int) and w > 100 and h > 100:
                return w, h
        except Exception:
            pass

        return self.config.monitor_width, self.config.monitor_height

    def setup_mediapipe(self) -> bool:
        """Initialize MediaPipe FaceMesh."""
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # enables iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.initialized = True
            return True
        except Exception:
            self._face_mesh = None
            self.initialized = False
            return False

    def reset_calibration(self):
        """Reset yaw/pitch offsets and eye-sphere locking."""
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.manual_offset_x_px = 0
        self.manual_offset_y_px = 0

        self._left_sphere_locked = False
        self._right_sphere_locked = False
        self._left_sphere_local_offset = None
        self._right_sphere_local_offset = None
        self._left_calibration_nose_scale = None
        self._right_calibration_nose_scale = None

        self._valid_frame_count = 0
        self._combined_gaze_directions.clear()
        self._screen_position_history.clear()

    def _compute_scale(self, points_3d: np.ndarray) -> float:
        """Robust-ish scale proxy from nose landmark cloud."""
        # Average pairwise distance for small point sets is fine (24 points)
        n = int(points_3d.shape[0])
        if n < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
                count += 1
        return total / count if count > 0 else 1.0

    def _compute_head_frame(self, face_landmarks, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute head_center (3,), R_final (3x3), and nose_points_3d (Nx3).

        Ported PCA orientation from MonitorTracking.py.
        """
        points_3d = np.array(
            [[face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w] for i in self._NOSE_INDICES],
            dtype=float,
        )
        center = np.mean(points_3d, axis=0)
        centered = points_3d - center

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]

        # Ensure right-handed
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        R_final = eigvecs

        # Stabilize sign flips vs reference (axis-by-axis)
        if self._R_ref_nose is None:
            self._R_ref_nose = R_final.copy()
        else:
            for k in range(3):
                if float(np.dot(R_final[:, k], self._R_ref_nose[:, k])) < 0.0:
                    R_final[:, k] *= -1.0

        return center, R_final, points_3d

    def _lock_eye_spheres(
        self,
        head_center: np.ndarray,
        R_final: np.ndarray,
        nose_points_3d: np.ndarray,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
    ) -> None:
        """
        Lock per-eye sphere centers relative to the head frame.

        This follows MonitorTracking.py's approach: store local offsets and add
        a base radius along camera direction (in head-local coordinates).
        """
        current_nose_scale = self._compute_scale(nose_points_3d)

        # Camera direction in this coordinate system (approx).
        # In the original script they use [0,0,1] in world and transform to local.
        camera_dir_world = np.array([0.0, 0.0, 1.0], dtype=float)
        camera_dir_local = R_final.T @ camera_dir_world

        left_offset = R_final.T @ (iris_3d_left - head_center)
        right_offset = R_final.T @ (iris_3d_right - head_center)

        left_offset = left_offset + self._base_radius * camera_dir_local
        right_offset = right_offset + self._base_radius * camera_dir_local

        self._left_sphere_local_offset = left_offset
        self._right_sphere_local_offset = right_offset
        self._left_calibration_nose_scale = current_nose_scale
        self._right_calibration_nose_scale = current_nose_scale

        self._left_sphere_locked = True
        self._right_sphere_locked = True

    def _convert_gaze_dir_to_screen_xy(
        self,
        gaze_dir: np.ndarray,
    ) -> Tuple[int, int, float, float]:
        """
        Convert 3D gaze direction to 2D monitor coordinates.

        Ported from MonitorTracking.py's convert_gaze_to_screen_coordinates().
        Returns (screen_x_px, screen_y_px, raw_yaw_deg, raw_pitch_deg).
        """
        reference_forward = np.array([0.0, 0.0, -1.0], dtype=float)
        avg_direction = gaze_dir / (np.linalg.norm(gaze_dir) + 1e-9)

        # Yaw (project onto XZ)
        xz_proj = np.array([avg_direction[0], 0.0, avg_direction[2]], dtype=float)
        xz_proj = xz_proj / (np.linalg.norm(xz_proj) + 1e-9)
        yaw_rad = math.acos(float(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0)))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad

        # Pitch (project onto YZ)
        yz_proj = np.array([0.0, avg_direction[1], avg_direction[2]], dtype=float)
        yz_proj = yz_proj / (np.linalg.norm(yz_proj) + 1e-9)
        pitch_rad = math.acos(float(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0)))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = float(np.degrees(yaw_rad))
        pitch_deg = float(np.degrees(pitch_rad))

        # Match the original script's sign convention tweaks
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg

        # Apply calibration offsets
        yaw_deg += float(self.calibration_offset_yaw)
        pitch_deg += float(self.calibration_offset_pitch)

        yaw_range = float(self.config.yaw_degrees)
        pitch_range = float(self.config.pitch_degrees)

        # Map to monitor pixels
        screen_x = int(((-yaw_deg + yaw_range) / (2.0 * yaw_range)) * self.monitor_width)
        screen_y = int(((pitch_range - pitch_deg) / (2.0 * pitch_range)) * self.monitor_height)

        # Manual correction offsets
        screen_x += int(self.manual_offset_x_px)
        screen_y += int(self.manual_offset_y_px)

        # Clamp
        screen_x = max(0, min(screen_x, self.monitor_width - 1))
        screen_y = max(0, min(screen_y, self.monitor_height - 1))

        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def calibrate_center(self, frame: np.ndarray) -> bool:
        """
        Calibrate yaw/pitch offsets so the current gaze maps to screen center.

        Returns True if calibration succeeded.
        """
        gaze = self.get_gaze(frame, _internal_no_center_calibration=True)
        if gaze is None:
            return False

        # We need raw yaw/pitch from the last conversion. Easiest: re-run conversion here.
        # get_gaze() stores smoothed direction; use the last buffered direction if present.
        if not self._combined_gaze_directions:
            return False
        dir_now = self._combined_gaze_directions[-1]

        _, _, raw_yaw, raw_pitch = self._convert_gaze_dir_to_screen_xy(dir_now)

        # Offsets required to center
        self.calibration_offset_yaw = 0.0 - raw_yaw
        self.calibration_offset_pitch = 0.0 - raw_pitch
        return True

    def get_gaze(self, frame: np.ndarray, _internal_no_center_calibration: bool = False) -> Optional[Tuple[float, float]]:
        """
        Compute normalized gaze coordinates (0..1) on the monitor.
        Returns None if face not detected or tracker not ready.
        """
        if not self.initialized or self._face_mesh is None:
            return None

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            self._valid_frame_count = 0
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Iris center indices (FaceMesh with refine_landmarks=True)
        left_iris_idx = 468
        right_iris_idx = 473
        if len(face_landmarks) <= right_iris_idx:
            return None

        head_center, R_final, nose_points_3d = self._compute_head_frame(face_landmarks, w, h)

        iris_l = face_landmarks[left_iris_idx]
        iris_r = face_landmarks[right_iris_idx]
        iris_3d_left = np.array([iris_l.x * w, iris_l.y * h, iris_l.z * w], dtype=float)
        iris_3d_right = np.array([iris_r.x * w, iris_r.y * h, iris_r.z * w], dtype=float)

        # Auto-lock eye spheres after enough stable frames
        if self.config.auto_lock_eye_spheres and not (self._left_sphere_locked and self._right_sphere_locked):
            self._valid_frame_count += 1
            if self._valid_frame_count >= self.config.auto_lock_frames_required:
                self._lock_eye_spheres(head_center, R_final, nose_points_3d, iris_3d_left, iris_3d_right)
        else:
            self._valid_frame_count = 0

        if not (self._left_sphere_locked and self._right_sphere_locked):
            return None

        # Recompute per-eye sphere positions (scale-aware)
        current_nose_scale = self._compute_scale(nose_points_3d)

        scale_ratio_l = (
            current_nose_scale / self._left_calibration_nose_scale
            if self._left_calibration_nose_scale
            else 1.0
        )
        scale_ratio_r = (
            current_nose_scale / self._right_calibration_nose_scale
            if self._right_calibration_nose_scale
            else 1.0
        )

        sphere_world_l = head_center + R_final @ (self._left_sphere_local_offset * scale_ratio_l)  # type: ignore[arg-type]
        sphere_world_r = head_center + R_final @ (self._right_sphere_local_offset * scale_ratio_r)  # type: ignore[arg-type]

        # Per-eye gaze directions
        left_gaze_dir = iris_3d_left - sphere_world_l
        right_gaze_dir = iris_3d_right - sphere_world_r
        left_norm = float(np.linalg.norm(left_gaze_dir))
        right_norm = float(np.linalg.norm(right_gaze_dir))
        if left_norm < 1e-9 or right_norm < 1e-9:
            return None
        left_gaze_dir /= left_norm
        right_gaze_dir /= right_norm

        raw_combined = (left_gaze_dir + right_gaze_dir) * 0.5
        raw_combined /= (np.linalg.norm(raw_combined) + 1e-9)

        self._combined_gaze_directions.append(raw_combined)
        avg_combined = np.mean(np.stack(list(self._combined_gaze_directions), axis=0), axis=0)
        avg_combined /= (np.linalg.norm(avg_combined) + 1e-9)

        # Convert to monitor pixel coordinates
        raw_x, raw_y, _, _ = self._convert_gaze_dir_to_screen_xy(avg_combined)

        # Smooth screen position (weighted toward recent)
        self._screen_position_history.append((raw_x, raw_y))
        if len(self._screen_position_history) > 1:
            weights = np.array([1.0 + i * 0.5 for i in range(len(self._screen_position_history))], dtype=float)
            weights /= float(weights.sum())
            positions = np.array(list(self._screen_position_history), dtype=float)
            sx = int(np.sum(positions[:, 0] * weights))
            sy = int(np.sum(positions[:, 1] * weights))
        else:
            sx, sy = raw_x, raw_y

        # Return normalized coords
        x_norm = float(sx) / float(max(1, self.monitor_width - 1))
        y_norm = float(sy) / float(max(1, self.monitor_height - 1))
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        return (x_norm, y_norm)

