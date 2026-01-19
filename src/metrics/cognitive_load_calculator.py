"""
Cognitive Load Index (CLI) Calculator with 10-second window

Key changes:
1. Time-based sliding window (10 seconds) instead of frame-based
2. CLI calculation triggers every 10 seconds
3. All metrics calculated from data within the 10-second window
4. Automatic cleanup of old data outside the window
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import math


@dataclass
class GazeMetrics:
    """Gaze-based cognitive load indicators"""
    fixation_duration: float = 0.0
    saccade_frequency: float = 0.0
    gaze_dispersion: float = 0.0
    pupil_dilation: float = 0.0
    search_pattern_score: float = 0.0
    
    def get_load_score(self) -> float:
        scores = []
        if self.fixation_duration > 0:
            scores.append(min(1.0, self.fixation_duration / 4.0))
        if self.saccade_frequency > 0:
            scores.append(min(1.0, self.saccade_frequency / 5.0))
        scores.append(min(1.0, self.gaze_dispersion))
        if self.pupil_dilation > 0:
            scores.append(min(1.0, self.pupil_dilation / 0.3))
        scores.append(self.search_pattern_score)
        return np.mean(scores) if scores else 0.0


@dataclass
class EmotionMetrics:
    """Emotion-based cognitive load indicators"""
    negative_affect_score: float = 0.0
    valence: float = 0.5
    arousal: float = 0.5
    brow_furrow_intensity: float = 0.0
    frozen_face_duration: float = 0.0
    dominant_emotion: str = "neutral"
    
    def get_load_score(self) -> float:
        scores = []
        scores.append(min(1.0, self.negative_affect_score))
        scores.append(1.0 - self.valence)
        if self.valence < 0.5:
            scores.append(self.arousal)
        else:
            scores.append(0.0)
        scores.append(self.brow_furrow_intensity)
        if self.frozen_face_duration > 1.0:
            scores.append(min(1.0, self.frozen_face_duration / 5.0))
        return np.mean(scores) if scores else 0.0


@dataclass
class MouseMetrics:
    """Mouse/Click-based cognitive load indicators"""
    click_rate: float = 0.0
    error_click_rate: float = 0.0
    path_efficiency: float = 1.0
    hover_dwell_time: float = 0.0
    movement_speed_variance: float = 0.0
    
    def get_load_score(self) -> float:
        scores = []
        if self.click_rate > 0:
            scores.append(min(1.0, self.click_rate / 3.0))
        if self.error_click_rate > 0:
            scores.append(min(1.0, self.error_click_rate / 1.0))
        scores.append(1.0 - self.path_efficiency)
        if self.hover_dwell_time > 0.5:
            scores.append(min(1.0, (self.hover_dwell_time - 0.5) / 3.0))
        scores.append(min(1.0, self.movement_speed_variance))
        return np.mean(scores) if scores else 0.0


class CognitiveLoadCalculator:
    """
    Cognitive Load Calculator with 10-second sliding window
    
    Collects data continuously and calculates CLI every 10 seconds
    based on data from the previous 10-second window.
    """
    
    def __init__(
        self,
        weight_gaze: float = 0.5,
        weight_emotion: float = 0.3,
        weight_mouse: float = 0.2,
        window_duration: float = 5.0,    # Window size in seconds (<=0 => use ALL history)
        update_interval: float = 5.0,    # Update CLI every N seconds (<=0 => every call)
        history_window: int = 30,         # DEPRECATED: kept for backward compatibility
        baseline_frames: int = 60         # DEPRECATED: kept for backward compatibility
    ):
        """
        Initialize the calculator with 10-second window.
        
        Args:
            weight_gaze: Weight for gaze metrics (default 0.5)
            weight_emotion: Weight for emotion metrics (default 0.3)
            weight_mouse: Weight for mouse metrics (default 0.2)
            window_duration: Time window for data (seconds, default 10)
            update_interval: CLI update frequency (seconds, default 10)
            history_window: DEPRECATED - kept for backward compatibility
            baseline_frames: DEPRECATED - kept for backward compatibility
        """
        # Normalize weights
        total_weight = weight_gaze + weight_emotion + weight_mouse
        self.weight_gaze = weight_gaze / total_weight
        self.weight_emotion = weight_emotion / total_weight
        self.weight_mouse = weight_mouse / total_weight
        
        # Window configuration
        # If window_duration <= 0: treat as "use all history" (no time cutoff)
        self.window_duration: Optional[timedelta]
        if window_duration is None or window_duration <= 0:
            self.window_duration = None
        else:
            self.window_duration = timedelta(seconds=window_duration)

        # If update_interval <= 0: calculate on every call (no throttling)
        if update_interval is None or update_interval <= 0:
            self.update_interval = timedelta(seconds=0)
        else:
            self.update_interval = timedelta(seconds=update_interval)
        
        # Time-stamped data storage (timestamp, data)
        self.gaze_data: deque = deque()      # (timestamp, x, y, pupil_size)
        self.emotion_data: deque = deque()   # (timestamp, emotion_dict)
        self.mouse_data: deque = deque()     # (timestamp, x, y)
        self.click_data: deque = deque()     # (timestamp, is_error)
        
        # Timing control
        self.last_cli_update: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        
        # Normalization (adaptive min/max)
        self.gaze_min = 0.0
        self.gaze_max = 1.0
        self.emotion_min = 0.0
        self.emotion_max = 1.0
        self.mouse_min = 0.0
        self.mouse_max = 1.0
        
        # Score history for normalization
        self.gaze_score_history: deque = deque()     # (timestamp, score)
        self.emotion_score_history: deque = deque()  # (timestamp, score)
        self.mouse_score_history: deque = deque()    # (timestamp, score)
        
        # Baseline pupil
        self._baseline_pupil: Optional[float] = None
        
        # Cache last result
        self._last_cli_result: Optional[Dict[str, Any]] = None
    
    def _cleanup_old_data(self, current_time: datetime):
        """Remove data older than the window duration (if windowing is enabled)."""
        if self.window_duration is None:
            return

        cutoff = current_time - self.window_duration
        
        while self.gaze_data and self.gaze_data[0][0] < cutoff:
            self.gaze_data.popleft()
        
        while self.emotion_data and self.emotion_data[0][0] < cutoff:
            self.emotion_data.popleft()
        
        while self.mouse_data and self.mouse_data[0][0] < cutoff:
            self.mouse_data.popleft()
        
        while self.click_data and self.click_data[0][0] < cutoff:
            self.click_data.popleft()
        
        while self.gaze_score_history and self.gaze_score_history[0][0] < cutoff:
            self.gaze_score_history.popleft()
        
        while self.emotion_score_history and self.emotion_score_history[0][0] < cutoff:
            self.emotion_score_history.popleft()
        
        while self.mouse_score_history and self.mouse_score_history[0][0] < cutoff:
            self.mouse_score_history.popleft()
    
    def update_gaze(
        self,
        x: float,
        y: float,
        pupil_size: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """Add gaze data to the window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        # Set baseline pupil
        if pupil_size is not None and self._baseline_pupil is None:
            self._baseline_pupil = pupil_size
        
        self.gaze_data.append((timestamp, x, y, pupil_size))
        self._cleanup_old_data(timestamp)
    
    def update_emotion(
        self,
        emotion_result: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Add emotion data to the window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        self.emotion_data.append((timestamp, emotion_result))
        self._cleanup_old_data(timestamp)
    
    def update_mouse(
        self,
        x: int,
        y: int,
        clicked: bool = False,
        is_error_click: bool = False,
        timestamp: Optional[datetime] = None
    ):
        """Add mouse data to the window."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        self.mouse_data.append((timestamp, x, y))
        
        if clicked:
            self.click_data.append((timestamp, is_error_click))
        
        self._cleanup_old_data(timestamp)
    
    def should_calculate_cli(self, current_time: Optional[datetime] = None) -> bool:
        """Check if it's time to calculate CLI."""
        if current_time is None:
            current_time = datetime.now()
        
        if self.last_cli_update is None:
            return True
        
        return (current_time - self.last_cli_update) >= self.update_interval

    def _get_windowed(self, data: deque, current_time: datetime):
        """Return data filtered to the active time window (or all data if window disabled)."""
        if self.window_duration is None:
            return list(data)
        cutoff = current_time - self.window_duration
        return [row for row in data if row[0] >= cutoff]

    def _effective_window_seconds(self, positions: List[Tuple]) -> float:
        """
        Effective time span represented by the data.
        - If time window is enabled: use configured window seconds
        - If disabled: use (last_ts - first_ts)
        """
        if self.window_duration is not None:
            return max(0.001, self.window_duration.total_seconds())
        if len(positions) >= 2:
            return max(0.001, (positions[-1][0] - positions[0][0]).total_seconds())
        return 0.001
    
    def _calculate_gaze_metrics(self, current_time: datetime) -> GazeMetrics:
        """Calculate gaze metrics from windowed data."""
        metrics = GazeMetrics()
        
        if not self.gaze_data:
            return metrics
        
        positions = self._get_windowed(self.gaze_data, current_time)
        if not positions:
            return metrics
        
        # Gaze dispersion
        if len(positions) >= 2:
            coords = np.array([(p[1], p[2]) for p in positions])
            metrics.gaze_dispersion = (np.std(coords[:, 0]) + np.std(coords[:, 1])) / 2.0
        
        # Saccades and fixations
        fixation_durations = []
        saccade_count = 0
        fixation_threshold = 0.03
        
        current_fix_start = None
        
        for i in range(1, len(positions)):
            prev_t, prev_x, prev_y, _ = positions[i-1]
            curr_t, curr_x, curr_y, _ = positions[i]
            
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            time_diff = (curr_t - prev_t).total_seconds()
            
            # Saccade detection
            if distance > 0.03 and time_diff < 0.1:
                saccade_count += 1
            
            # Fixation detection
            if distance < fixation_threshold:
                if current_fix_start is None:
                    current_fix_start = prev_t
            else:
                if current_fix_start is not None:
                    duration = (prev_t - current_fix_start).total_seconds()
                    fixation_durations.append(duration)
                    current_fix_start = None
        
        if fixation_durations:
            metrics.fixation_duration = np.mean(fixation_durations)
        
        # Saccade frequency
        window_sec = self._effective_window_seconds(positions)
        metrics.saccade_frequency = saccade_count / window_sec if window_sec > 0 else 0
        
        # Pupil dilation
        if self._baseline_pupil is not None:
            pupil_sizes = [p[3] for p in positions if p[3] is not None]
            if pupil_sizes:
                avg_pupil = np.mean(pupil_sizes)
                metrics.pupil_dilation = abs(avg_pupil - self._baseline_pupil) / self._baseline_pupil
        
        # Search pattern
        metrics.search_pattern_score = min(1.0, (
            metrics.gaze_dispersion * 2 + metrics.saccade_frequency / 3
        ) / 2)
        
        return metrics
    
    def _calculate_emotion_metrics(self, current_time: datetime) -> EmotionMetrics:
        """Calculate emotion metrics from windowed data."""
        metrics = EmotionMetrics()
        
        if not self.emotion_data:
            return metrics
        
        # Average recent emotion data
        emotions_list = self._get_windowed(self.emotion_data, current_time)
        if not emotions_list:
            return metrics
        
        negative_affects = []
        valences = []
        arousals = []
        brow_furrows = []
        
        for ts, emotion_result in emotions_list:
            emotions = emotion_result.get('emotions', {})
            
            negative_affects.append(emotion_result.get('negative_affect_score', 0.0))
            
            # Valence
            pos_sum = sum(emotions.get(e, 0.0) for e in ['happy', 'surprise'])
            neg_sum = sum(emotions.get(e, 0.0) for e in ['angry', 'sad', 'fear', 'disgust'])
            valences.append(pos_sum / (pos_sum + neg_sum) if (pos_sum + neg_sum) > 0 else 0.5)
            
            # Arousal
            max_prob = max(emotions.values()) if emotions else 0.0
            neutral_prob = emotions.get('neutral', 0.0)
            arousals.append((max_prob - neutral_prob + 1) / 2)
            
            # Brow furrow
            brow_furrows.append(min(1.0, (
                emotions.get('angry', 0.0) * 0.5 +
                emotions.get('sad', 0.0) * 0.3 +
                emotions.get('fear', 0.0) * 0.4
            ) * 2))
        
        metrics.negative_affect_score = np.mean(negative_affects) if negative_affects else 0.0
        metrics.valence = np.mean(valences) if valences else 0.5
        metrics.arousal = np.mean(arousals) if arousals else 0.5
        metrics.brow_furrow_intensity = np.mean(brow_furrows) if brow_furrows else 0.0
        
        # Frozen face: check if emotion unchanged for window
        if len(emotions_list) >= 2:
            dominant_emotions = [e[1].get('dominant_emotion', 'neutral') for e in emotions_list]
            if len(set(dominant_emotions)) == 1:
                metrics.frozen_face_duration = (
                    emotions_list[-1][0] - emotions_list[0][0]
                ).total_seconds()
                metrics.dominant_emotion = dominant_emotions[0]
            else:
                metrics.dominant_emotion = emotions_list[-1][1].get('dominant_emotion', 'neutral')
        
        return metrics
    
    def _calculate_mouse_metrics(self, current_time: datetime) -> MouseMetrics:
        """Calculate mouse metrics from windowed data."""
        metrics = MouseMetrics()
        
        if not self.mouse_data:
            return metrics
        
        positions = self._get_windowed(self.mouse_data, current_time)
        if not positions:
            return metrics

        # Clicks should use the same window definition
        click_positions = self._get_windowed(self.click_data, current_time)
        window_sec = self._effective_window_seconds(positions)
        
        # Click rates
        total_clicks = len(click_positions)
        error_clicks = sum(1 for _, is_err in click_positions if is_err)
        
        metrics.click_rate = total_clicks / window_sec if window_sec > 0 else 0
        metrics.error_click_rate = error_clicks / window_sec if window_sec > 0 else 0
        
        # Path efficiency
        if len(positions) >= 2:
            actual_dist = sum(
                math.sqrt(
                    (positions[i][1] - positions[i-1][1])**2 +
                    (positions[i][2] - positions[i-1][2])**2
                )
                for i in range(1, len(positions))
            )
            
            optimal_dist = math.sqrt(
                (positions[-1][1] - positions[0][1])**2 +
                (positions[-1][2] - positions[0][2])**2
            )
            
            if actual_dist > 0:
                metrics.path_efficiency = min(1.0, optimal_dist / actual_dist)
        
        # Movement speed variance
        if len(positions) >= 3:
            speeds = []
            for i in range(1, len(positions)):
                dx = positions[i][1] - positions[i-1][1]
                dy = positions[i][2] - positions[i-1][2]
                dt = (positions[i][0] - positions[i-1][0]).total_seconds()
                if dt > 0:
                    speeds.append(math.sqrt(dx*dx + dy*dy) / dt)
            
            if speeds:
                metrics.movement_speed_variance = min(1.0, 
                    np.std(speeds) / (np.mean(speeds) + 1)
                )
        
        # Hover dwell time
        max_dwell = 0.0
        dwell_start = None
        hover_threshold = 5
        
        for i in range(1, len(positions)):
            dist = math.sqrt(
                (positions[i][1] - positions[i-1][1])**2 +
                (positions[i][2] - positions[i-1][2])**2
            )
            
            if dist < hover_threshold:
                if dwell_start is None:
                    dwell_start = positions[i-1][0]
            else:
                if dwell_start is not None:
                    dwell = (positions[i-1][0] - dwell_start).total_seconds()
                    max_dwell = max(max_dwell, dwell)
                    dwell_start = None
        
        metrics.hover_dwell_time = max_dwell
        
        return metrics
    
    def calculate_cli(
        self,
        force_update: bool = False,
        current_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate CLI if 10 seconds have passed since last update.
        
        Args:
            force_update: Force calculation regardless of interval
            current_time: Optional current time (defaults to now)
        
        Returns:
            CLI result dict or None if not time to update
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check if we should calculate
        if not force_update and not self.should_calculate_cli(current_time):
            return self._last_cli_result
        
        # Calculate metrics from windowed data (or full history if window disabled)
        gaze_metrics = self._calculate_gaze_metrics(current_time)
        emotion_metrics = self._calculate_emotion_metrics(current_time)
        mouse_metrics = self._calculate_mouse_metrics(current_time)
        
        # Get raw scores
        gaze_raw = gaze_metrics.get_load_score()
        emotion_raw = emotion_metrics.get_load_score()
        mouse_raw = mouse_metrics.get_load_score()
        
        # Store scores with timestamp
        self.gaze_score_history.append((current_time, gaze_raw))
        self.emotion_score_history.append((current_time, emotion_raw))
        self.mouse_score_history.append((current_time, mouse_raw))
        
        # Update adaptive min/max
        alpha = 0.1
        
        if self.gaze_score_history:
            scores = [s[1] for s in self.gaze_score_history]
            self.gaze_min = min(self.gaze_min * (1-alpha) + min(scores) * alpha, gaze_raw)
            self.gaze_max = max(self.gaze_max * (1-alpha) + max(scores) * alpha, gaze_raw)
        
        if self.emotion_score_history:
            scores = [s[1] for s in self.emotion_score_history]
            self.emotion_min = min(self.emotion_min * (1-alpha) + min(scores) * alpha, emotion_raw)
            self.emotion_max = max(self.emotion_max * (1-alpha) + max(scores) * alpha, emotion_raw)
        
        if self.mouse_score_history:
            scores = [s[1] for s in self.mouse_score_history]
            self.mouse_min = min(self.mouse_min * (1-alpha) + min(scores) * alpha, mouse_raw)
            self.mouse_max = max(self.mouse_max * (1-alpha) + max(scores) * alpha, mouse_raw)
        
        # Normalize
        def normalize(score, min_val, max_val):
            if max_val - min_val < 0.001:
                return 0.5
            return (score - min_val) / (max_val - min_val)
        
        gaze_norm = normalize(gaze_raw, self.gaze_min, self.gaze_max)
        emotion_norm = normalize(emotion_raw, self.emotion_min, self.emotion_max)
        mouse_norm = normalize(mouse_raw, self.mouse_min, self.mouse_max)
        
        # Calculate weighted CLI
        cli = (
            self.weight_gaze * gaze_norm +
            self.weight_emotion * emotion_norm +
            self.weight_mouse * mouse_norm
        )
        cli = max(0.0, min(1.0, cli))
        
        # Load level
        if cli < 0.33:
            load_level = 'low'
        elif cli < 0.66:
            load_level = 'medium'
        else:
            load_level = 'high'
        
        # Build result
        result = {
            'cli': cli,
            'gaze_score': gaze_norm,
            'emotion_score': emotion_norm,
            'mouse_score': mouse_norm,
            'gaze_raw': gaze_raw,
            'emotion_raw': emotion_raw,
            'mouse_raw': mouse_raw,
            'load_level': load_level,
            'timestamp': current_time,
            'window_duration': self.window_duration.total_seconds() if self.window_duration is not None else None,
            'weights': {
                'gaze': self.weight_gaze,
                'emotion': self.weight_emotion,
                'mouse': self.weight_mouse
            },
            'details': {
                'gaze': {
                    'fixation_duration': gaze_metrics.fixation_duration,
                    'saccade_frequency': gaze_metrics.saccade_frequency,
                    'gaze_dispersion': gaze_metrics.gaze_dispersion,
                    'pupil_dilation': gaze_metrics.pupil_dilation,
                    'search_pattern': gaze_metrics.search_pattern_score
                },
                'emotion': {
                    'negative_affect': emotion_metrics.negative_affect_score,
                    'valence': emotion_metrics.valence,
                    'arousal': emotion_metrics.arousal,
                    'brow_furrow': emotion_metrics.brow_furrow_intensity,
                    'frozen_face_duration': emotion_metrics.frozen_face_duration,
                    'dominant_emotion': emotion_metrics.dominant_emotion
                },
                'mouse': {
                    'click_rate': mouse_metrics.click_rate,
                    'error_click_rate': mouse_metrics.error_click_rate,
                    'path_efficiency': mouse_metrics.path_efficiency,
                    'hover_dwell': mouse_metrics.hover_dwell_time,
                    'speed_variance': mouse_metrics.movement_speed_variance
                }
            }
        }
        
        # Update timing
        self.last_cli_update = current_time
        self._last_cli_result = result
        
        return result
    
    def get_load_interpretation(self, cli_result: Dict[str, Any]) -> str:
        """Get human-readable interpretation of cognitive load."""
        cli = cli_result['cli']
        level = cli_result['load_level']
        details = cli_result['details']
        
        interpretations = []
        
        if level == 'low':
            interpretations.append("User appears comfortable.")
        elif level == 'medium':
            interpretations.append("User experiencing moderate cognitive load.")
        else:
            interpretations.append("User may be experiencing cognitive overload!")
        
        gaze = details['gaze']
        if gaze['fixation_duration'] > 2.0:
            interpretations.append(f"Long fixations ({gaze['fixation_duration']:.1f}s) suggest processing difficulty.")
        if gaze['saccade_frequency'] > 2.0:
            interpretations.append(f"High saccade rate ({gaze['saccade_frequency']:.1f}/s) indicates searching.")
        
        emotion = details['emotion']
        if emotion['negative_affect'] > 0.4:
            interpretations.append(f"Negative affect ({emotion['negative_affect']:.2f}) - possible frustration.")
        if emotion['frozen_face_duration'] > 3.0:
            interpretations.append(f"Frozen expression ({emotion['frozen_face_duration']:.1f}s) may indicate tunnel vision.")
        
        mouse = details['mouse']
        if mouse['click_rate'] > 1.5:
            interpretations.append(f"High click rate ({mouse['click_rate']:.1f}/s) suggests trial-and-error.")
        if mouse['path_efficiency'] < 0.5:
            interpretations.append(f"Low path efficiency ({mouse['path_efficiency']:.2f}) indicates uncertainty.")
        
        return " ".join(interpretations)
    
    def reset(self):
        """Reset all state."""
        self.gaze_data.clear()
        self.emotion_data.clear()
        self.mouse_data.clear()
        self.click_data.clear()
        
        self.gaze_score_history.clear()
        self.emotion_score_history.clear()
        self.mouse_score_history.clear()
        
        self.last_cli_update = None
        self.start_time = None
        self._baseline_pupil = None
        self._last_cli_result = None