"""
PyGaze Eye Tracking Module

A comprehensive wrapper around PyGaze for eye tracking experiments.
Supports multiple eye trackers including EyeLink, EyeTribe, and dummy modes.
"""

from .pygaze_wrapper import PyGazeTracker
from .event_detector import EventDetector
from .calibration_manager import CalibrationManager

__all__ = [
    'PyGazeTracker',
    'EventDetector',
    'CalibrationManager',
]

__version__ = '1.0.0'
