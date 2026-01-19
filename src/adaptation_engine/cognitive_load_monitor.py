"""
Cognitive Load Monitor
Enhanced version using the Cognitive Load Calculator for accurate CLI estimation.

Combines:
- Gaze metrics (fixation, saccades, dispersion)
- Emotion metrics (valence, arousal, frozen face)
- Mouse metrics (click rate, path efficiency, dwell)

Into a weighted Cognitive Load Index (CLI).
"""

from typing import Dict, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import numpy as np

from src.metrics.cognitive_load_calculator import (
    CognitiveLoadCalculator,
    GazeMetrics,
    EmotionMetrics,
    MouseMetrics
)


class GazeEvent:
    """Represents a gaze event with timestamp and coordinates"""
    
    def __init__(self, x: float, y: float, timestamp: datetime):
        self.x = x
        self.y = y
        self.timestamp = timestamp


class CognitiveLoadMonitor:
    """
    Enhanced Cognitive Load Monitor using composite CLI calculation.
    
    Monitors for cognitive overload using:
    - Eye Gaze (Processing Metric): Fixations, saccades, gaze dispersion
    - Facial Emotion (Stress Metric): Negative affect, valence/arousal, frozen face
    - Mouse/Clicks (Efficiency Metric): Click rate, path efficiency, dwell time
    """
    
    def __init__(
        self,
        gaze_wander_threshold: float = 0.3,  # Legacy threshold for backwards compatibility
        fixation_duration_threshold: float = 3.0,  # Seconds
        negative_affect_threshold: float = 0.5,
        gaze_history_window: int = 30,  # Number of gaze points to track
        # CLI weights (should sum to 1.0)
        weight_gaze: float = 0.5,
        weight_emotion: float = 0.3,
        weight_mouse: float = 0.2,
        # Windowing / update behavior
        window_duration_sec: float = 5.0,
        update_interval_sec: float = 5.0,
        # Overload threshold
        cli_overload_threshold: float = 0.65
    ):
        """
        Initialize cognitive load monitor.
        
        Args:
            gaze_wander_threshold: Legacy threshold for gaze wander detection
            fixation_duration_threshold: Duration (seconds) for fixation detection
            negative_affect_threshold: Threshold for negative emotion detection
            gaze_history_window: Number of recent gaze points to maintain
            weight_gaze: Weight for gaze metrics in CLI (default 0.5)
            weight_emotion: Weight for emotion metrics in CLI (default 0.3)
            weight_mouse: Weight for mouse metrics in CLI (default 0.2)
            cli_overload_threshold: CLI value above which overload is detected
        """
        # Legacy thresholds for backwards compatibility
        self.gaze_wander_threshold = gaze_wander_threshold
        self.fixation_duration_threshold = fixation_duration_threshold
        self.negative_affect_threshold = negative_affect_threshold
        
        # Gaze history for analysis
        self.gaze_history: deque = deque(maxlen=gaze_history_window)
        
        # Current fixation tracking (legacy)
        self.current_fixation_start: Optional[datetime] = None
        self.current_fixation_location: Optional[Tuple[float, float]] = None
        self.fixation_radius = 0.05  # 5% of screen size
        
        # Initialize the new CLI calculator
        self.cli_calculator = CognitiveLoadCalculator(
            weight_gaze=weight_gaze,
            weight_emotion=weight_emotion,
            weight_mouse=weight_mouse,
            window_duration=window_duration_sec,
            update_interval=update_interval_sec,
            history_window=gaze_history_window
        )
        
        # CLI overload threshold
        self.cli_overload_threshold = cli_overload_threshold
        
        # State
        self.cognitive_load_score = 0.0
        self.overload_detected = False
        self.negative_affect_score = 0.0
        self._last_cli_result: Optional[Dict[str, Any]] = None
        
    def update_gaze(
        self,
        x: float,
        y: float,
        timestamp: Optional[datetime] = None,
        pupil_size: Optional[float] = None
    ):
        """
        Update gaze tracking with new gaze data.
        
        Args:
            x: Gaze x coordinate (normalized 0-1)
            y: Gaze y coordinate (normalized 0-1)
            timestamp: Timestamp of gaze event (defaults to now)
            pupil_size: Optional pupil diameter for dilation tracking
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Legacy gaze event tracking
        gaze_event = GazeEvent(x, y, timestamp)
        self.gaze_history.append(gaze_event)
        
        # Update legacy fixation tracking
        self._update_fixation(gaze_event)
        
        # Update CLI calculator with gaze data
        self.cli_calculator.update_gaze(x, y, pupil_size=pupil_size, timestamp=timestamp)
    
    def _update_fixation(self, gaze_event: GazeEvent):
        """Update current fixation state (legacy method)"""
        if self.current_fixation_location is None:
            # Start new fixation
            self.current_fixation_start = gaze_event.timestamp
            self.current_fixation_location = (gaze_event.x, gaze_event.y)
        else:
            # Check if still in same fixation region
            fx, fy = self.current_fixation_location
            distance = np.sqrt((gaze_event.x - fx)**2 + (gaze_event.y - fy)**2)
            
            if distance > self.fixation_radius:
                # Fixation broken, start new one
                self.current_fixation_start = gaze_event.timestamp
                self.current_fixation_location = (gaze_event.x, gaze_event.y)
    
    def detect_gaze_wander(self) -> bool:
        """
        Detect excessive scanning without interaction (gaze wander).
        
        Returns:
            True if gaze wander detected
        """
        if len(self.gaze_history) < 10:
            return False
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(self.gaze_history)):
            prev = self.gaze_history[i-1]
            curr = self.gaze_history[i]
            distance = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            total_distance += distance
        
        # Average distance per gaze point
        avg_distance = total_distance / len(self.gaze_history)
        
        return avg_distance > self.gaze_wander_threshold
    
    def detect_fixation_duration(self) -> Tuple[bool, Optional[float]]:
        """
        Detect unusually long fixations.
        
        Returns:
            Tuple of (is_long_fixation, duration_in_seconds)
        """
        if self.current_fixation_start is None:
            return False, None
        
        duration = (datetime.now() - self.current_fixation_start).total_seconds()
        is_long = duration > self.fixation_duration_threshold
        
        return is_long, duration
    
    def update_emotion(self, emotion_result: Dict[str, Any]):
        """
        Update emotion state.
        
        Args:
            emotion_result: Result from emotion detector
        """
        # Legacy negative affect tracking
        self.negative_affect_score = emotion_result.get('negative_affect_score', 0.0)
        
        # Update CLI calculator with emotion data
        self.cli_calculator.update_emotion(emotion_result)
    
    def update_mouse(
        self,
        x: int,
        y: int,
        clicked: bool = False,
        is_error_click: bool = False,
        timestamp: Optional[datetime] = None
    ):
        """
        Update mouse tracking with new mouse data.
        
        Args:
            x: Mouse X position (pixels)
            y: Mouse Y position (pixels)
            clicked: Whether a click occurred
            is_error_click: Whether this click is considered an error
            timestamp: Timestamp of mouse event
        """
        self.cli_calculator.update_mouse(
            x, y, clicked=clicked, is_error_click=is_error_click, timestamp=timestamp
        )
    
    def check_cognitive_overload(self) -> Dict[str, Any]:
        """
        Check for cognitive overload using the enhanced CLI calculator.
        
        Returns:
            Dictionary with overload status, CLI score, and detailed breakdown
        """
        # Calculate CLI using the enhanced calculator
        cli_result = self.cli_calculator.calculate_cli()
        self._last_cli_result = cli_result
        
        # Update state
        self.cognitive_load_score = cli_result['cli']
        self.overload_detected = cli_result['cli'] > self.cli_overload_threshold
        
        # Legacy trigger detection (for backwards compatibility)
        gaze_wander = self.detect_gaze_wander()
        long_fixation, fixation_duration = self.detect_fixation_duration()
        negative_affect = self.negative_affect_score > self.negative_affect_threshold
        
        return {
            'overload_detected': self.overload_detected,
            'cognitive_load_score': self.cognitive_load_score,
            'load_level': cli_result['load_level'],
            
            # CLI component scores
            'cli': cli_result['cli'],
            'gaze_score': cli_result['gaze_score'],
            'emotion_score': cli_result['emotion_score'],
            'mouse_score': cli_result['mouse_score'],
            
            # Legacy triggers (for backwards compatibility)
            'gaze_wander': gaze_wander,
            'long_fixation': long_fixation,
            'fixation_duration': fixation_duration,
            'negative_affect': negative_affect,
            
            # Detailed metrics
            'details': cli_result['details'],
            'weights': cli_result['weights'],
            
            # Legacy triggers dict
            'triggers': {
                'gaze_wander': gaze_wander,
                'fixation_duration': long_fixation,
                'negative_affect': negative_affect
            }
        }
    
    def get_cli_interpretation(self) -> str:
        """
        Get human-readable interpretation of current cognitive load.
        
        Returns:
            Interpretation string
        """
        if self._last_cli_result is None:
            return "No cognitive load data available yet."
        
        return self.cli_calculator.get_load_interpretation(self._last_cli_result)
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of all cognitive load metrics.
        
        Returns:
            Dictionary with all metric details
        """
        if self._last_cli_result is None:
            return {}
        
        return {
            'cli': self._last_cli_result['cli'],
            'load_level': self._last_cli_result['load_level'],
            'gaze': self._last_cli_result['details']['gaze'],
            'emotion': self._last_cli_result['details']['emotion'],
            'mouse': self._last_cli_result['details']['mouse'],
            'interpretation': self.get_cli_interpretation()
        }
    
    def reset(self):
        """Reset monitor state"""
        self.gaze_history.clear()
        self.current_fixation_start = None
        self.current_fixation_location = None
        self.cognitive_load_score = 0.0
        self.overload_detected = False
        self.negative_affect_score = 0.0
        self._last_cli_result = None
        
        # Reset CLI calculator
        self.cli_calculator.reset()
