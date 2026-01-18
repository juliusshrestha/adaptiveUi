"""
Tests for gaze tracking module
"""

import numpy as np
import pytest
from src.data_acquisition.gaze_tracker import KalmanFilter, GazeTracker


def test_kalman_filter_initialization():
    """Test Kalman filter initialization"""
    kf = KalmanFilter()
    assert kf.state is not None
    assert kf.covariance is not None


def test_kalman_filter_prediction():
    """Test Kalman filter prediction"""
    kf = KalmanFilter()
    prediction = kf.predict()
    assert len(prediction) == 2
    assert isinstance(prediction, np.ndarray)


def test_kalman_filter_update():
    """Test Kalman filter update"""
    kf = KalmanFilter()
    measurement = np.array([0.5, 0.5])
    filtered = kf.update(measurement)
    assert len(filtered) == 2
    assert isinstance(filtered, np.ndarray)


def test_gaze_tracker_initialization():
    """Test gaze tracker initialization"""
    tracker = GazeTracker(use_kalman=True)
    assert tracker.use_kalman is True
    assert tracker.kalman_filter is not None


def test_gaze_tracker_without_kalman():
    """Test gaze tracker without Kalman filter"""
    tracker = GazeTracker(use_kalman=False)
    assert tracker.use_kalman is False
    assert tracker.kalman_filter is None

