"""
Event Detection Module for PyGaze

Provides higher-level event detection and analysis capabilities
for fixations, saccades, and blinks.
"""

import time
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np
import logging

from .pygaze_wrapper import PyGazeTracker, GazeSample, Fixation, Saccade, Blink


@dataclass
class EventBuffer:
    """Buffer for storing gaze events"""
    fixations: List[Fixation]
    saccades: List[Saccade]
    blinks: List[Blink]
    samples: List[GazeSample]


class EventDetector:
    """
    High-level event detector for eye tracking data
    
    Provides methods to detect and analyze:
    - Fixations (gaze stability)
    - Saccades (rapid eye movements)
    - Blinks (eye closures)
    """
    
    def __init__(
        self,
        tracker: PyGazeTracker,
        fixation_threshold: float = 1.5,
        saccade_velocity_threshold: int = 35,
        min_fixation_duration: float = 0.1,
        max_fixation_duration: float = 3.0
    ):
        """
        Initialize event detector
        
        Args:
            tracker: PyGazeTracker instance
            fixation_threshold: Fixation threshold in degrees
            saccade_velocity_threshold: Saccade velocity threshold
            min_fixation_duration: Minimum fixation duration in seconds
            max_fixation_duration: Maximum fixation duration in seconds
        """
        self.tracker = tracker
        self.fixation_threshold = fixation_threshold
        self.saccade_velocity_threshold = saccade_velocity_threshold
        self.min_fixation_duration = min_fixation_duration
        self.max_fixation_duration = max_fixation_duration
        
        self.logger = logging.getLogger(__name__)
        
        # Event buffers
        self.buffer = EventBuffer(
            fixations=[],
            saccades=[],
            blinks=[],
            samples=[]
        )
        
        # Current event tracking
        self.current_fixation: Optional[Fixation] = None
        self.current_saccade: Optional[Saccade] = None
        self.current_blink: Optional[Blink] = None
    
    def collect_fixation(
        self,
        timeout: Optional[float] = None,
        min_duration: Optional[float] = None
    ) -> Optional[Fixation]:
        """
        Collect a single fixation event
        
        Args:
            timeout: Maximum time to wait (seconds)
            min_duration: Minimum fixation duration (seconds)
        
        Returns:
            Fixation object or None if timeout/no fixation
        """
        start_time = time.time()
        min_duration = min_duration or self.min_fixation_duration
        
        # Wait for fixation start
        result = self.tracker.wait_for_fixation_start()
        if result is None:
            return None
        
        fix_start_time, start_pos = result
        
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            return None
        
        # Wait for fixation end
        result = self.tracker.wait_for_fixation_end()
        if result is None:
            return None
        
        fix_end_time, end_pos = result
        duration = (fix_end_time - fix_start_time) / 1000.0  # Convert to seconds
        
        # Check minimum duration
        if duration < min_duration:
            return None
        
        # Calculate fixation center
        x = (start_pos[0] + end_pos[0]) / 2.0
        y = (start_pos[1] + end_pos[1]) / 2.0
        
        fixation = Fixation(
            start_time=fix_start_time,
            end_time=fix_end_time,
            duration=duration,
            x=x,
            y=y,
            start_pos=start_pos,
            end_pos=end_pos
        )
        
        self.buffer.fixations.append(fixation)
        return fixation
    
    def collect_saccade(
        self,
        timeout: Optional[float] = None
    ) -> Optional[Saccade]:
        """
        Collect a single saccade event
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            Saccade object or None if timeout/no saccade
        """
        start_time = time.time()
        
        # Wait for saccade start
        result = self.tracker.wait_for_saccade_start()
        if result is None:
            return None
        
        sacc_start_time, start_pos = result
        
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            return None
        
        # Wait for saccade end
        result = self.tracker.wait_for_saccade_end()
        if result is None:
            return None
        
        sacc_end_time, start_pos, end_pos = result
        duration = (sacc_end_time - sacc_start_time) / 1000.0  # Convert to seconds
        
        # Calculate amplitude and velocity
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        amplitude = np.sqrt(dx**2 + dy**2)
        velocity = amplitude / duration if duration > 0 else 0
        
        saccade = Saccade(
            start_time=sacc_start_time,
            end_time=sacc_end_time,
            duration=duration,
            start_pos=start_pos,
            end_pos=end_pos,
            amplitude=amplitude,
            velocity=velocity
        )
        
        self.buffer.saccades.append(saccade)
        return saccade
    
    def collect_blink(
        self,
        timeout: Optional[float] = None
    ) -> Optional[Blink]:
        """
        Collect a single blink event
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            Blink object or None if timeout/no blink
        """
        start_time = time.time()
        
        # Wait for blink start
        blink_start = self.tracker.wait_for_blink_start()
        if blink_start is None:
            return None
        
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            return None
        
        # Wait for blink end
        blink_end = self.tracker.wait_for_blink_end()
        if blink_end is None:
            return None
        
        duration = (blink_end - blink_start) / 1000.0  # Convert to seconds
        
        blink = Blink(
            start_time=blink_start,
            end_time=blink_end,
            duration=duration
        )
        
        self.buffer.blinks.append(blink)
        return blink
    
    def sample_continuous(
        self,
        duration: float,
        sample_rate: float = 60.0,
        callback: Optional[Callable[[GazeSample], None]] = None
    ) -> List[GazeSample]:
        """
        Continuously sample gaze data
        
        Args:
            duration: Sampling duration in seconds
            sample_rate: Target sample rate (Hz)
            callback: Optional callback function for each sample
        
        Returns:
            List of GazeSample objects
        """
        samples = []
        start_time = time.time()
        interval = 1.0 / sample_rate
        
        while (time.time() - start_time) < duration:
            sample = self.tracker.sample()
            if sample and sample.valid:
                samples.append(sample)
                self.buffer.samples.append(sample)
                
                if callback:
                    callback(sample)
            
            time.sleep(interval)
        
        return samples
    
    def analyze_fixations(
        self,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None
    ) -> dict:
        """
        Analyze fixation data in a time window
        
        Args:
            window_start: Start time (ms) or None for all
            window_end: End time (ms) or None for all
        
        Returns:
            Dictionary with fixation statistics
        """
        fixations = self.buffer.fixations
        
        # Filter by time window
        if window_start is not None or window_end is not None:
            fixations = [
                f for f in fixations
                if (window_start is None or f.start_time >= window_start) and
                   (window_end is None or f.end_time <= window_end)
            ]
        
        if not fixations:
            return {
                'count': 0,
                'total_duration': 0.0,
                'mean_duration': 0.0,
                'mean_x': 0.0,
                'mean_y': 0.0,
                'dispersion': 0.0
            }
        
        durations = [f.duration for f in fixations]
        x_coords = [f.x for f in fixations]
        y_coords = [f.y for f in fixations]
        
        # Calculate dispersion (standard deviation of positions)
        if len(x_coords) > 1:
            dispersion = np.sqrt(
                np.std(x_coords)**2 + np.std(y_coords)**2
            )
        else:
            dispersion = 0.0
        
        return {
            'count': len(fixations),
            'total_duration': sum(durations),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'mean_x': np.mean(x_coords),
            'mean_y': np.mean(y_coords),
            'dispersion': dispersion,
            'fixations': fixations
        }
    
    def analyze_saccades(
        self,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None
    ) -> dict:
        """
        Analyze saccade data in a time window
        
        Args:
            window_start: Start time (ms) or None for all
            window_end: End time (ms) or None for all
        
        Returns:
            Dictionary with saccade statistics
        """
        saccades = self.buffer.saccades
        
        # Filter by time window
        if window_start is not None or window_end is not None:
            saccades = [
                s for s in saccades
                if (window_start is None or s.start_time >= window_start) and
                   (window_end is None or s.end_time <= window_end)
            ]
        
        if not saccades:
            return {
                'count': 0,
                'mean_amplitude': 0.0,
                'mean_velocity': 0.0,
                'mean_duration': 0.0
            }
        
        amplitudes = [s.amplitude for s in saccades]
        velocities = [s.velocity for s in saccades]
        durations = [s.duration for s in saccades]
        
        return {
            'count': len(saccades),
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'mean_velocity': np.mean(velocities),
            'std_velocity': np.std(velocities),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'saccades': saccades
        }
    
    def analyze_blinks(
        self,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None
    ) -> dict:
        """
        Analyze blink data in a time window
        
        Args:
            window_start: Start time (ms) or None for all
            window_end: End time (ms) or None for all
        
        Returns:
            Dictionary with blink statistics
        """
        blinks = self.buffer.blinks
        
        # Filter by time window
        if window_start is not None or window_end is not None:
            blinks = [
                b for b in blinks
                if (window_start is None or b.start_time >= window_start) and
                   (window_end is None or b.end_time <= window_end)
            ]
        
        if not blinks:
            return {
                'count': 0,
                'mean_duration': 0.0,
                'frequency': 0.0
            }
        
        durations = [b.duration for b in blinks]
        
        # Calculate frequency (blinks per minute)
        if window_start and window_end:
            window_duration = (window_end - window_start) / 1000.0 / 60.0  # minutes
        else:
            # Use first and last blink
            if len(blinks) > 1:
                window_duration = (blinks[-1].end_time - blinks[0].start_time) / 1000.0 / 60.0
            else:
                window_duration = 1.0
        
        frequency = len(blinks) / window_duration if window_duration > 0 else 0
        
        return {
            'count': len(blinks),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'frequency': frequency,
            'blinks': blinks
        }
    
    def clear_buffer(self):
        """Clear all event buffers"""
        self.buffer.fixations.clear()
        self.buffer.saccades.clear()
        self.buffer.blinks.clear()
        self.buffer.samples.clear()
    
    def get_buffer(self) -> EventBuffer:
        """Get current event buffer"""
        return self.buffer
