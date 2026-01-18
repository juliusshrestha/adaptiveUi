"""
Data Acquisition Module
Handles eye gaze tracking, emotion detection, and mouse tracking for CLI calculation
"""

from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker
from src.data_acquisition.monitor_plane_gaze_tracker import (
    MonitorPlaneGazeTracker,
    MonitorGazeConfig,
)
from src.data_acquisition.emotion_detector import EmotionDetector
from src.data_acquisition.mouse_tracker import (
    MouseTracker,
    MouseMetricsSnapshot,
    PYNPUT_AVAILABLE
)

__all__ = [
    'DirectGazeTracker',
    'MonitorPlaneGazeTracker',
    'MonitorGazeConfig',
    'EmotionDetector',
    'MouseTracker',
    'MouseMetricsSnapshot',
    'PYNPUT_AVAILABLE'
]