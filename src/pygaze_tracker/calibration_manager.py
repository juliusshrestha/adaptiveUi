"""
Calibration Manager for PyGaze

Provides utilities for managing calibration procedures,
storing calibration data, and applying calibration corrections.
"""

import json
import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .pygaze_wrapper import PyGazeTracker


@dataclass
class CalibrationPoint:
    """Represents a calibration point"""
    x: float
    y: float
    timestamp: float
    success: bool = True


@dataclass
class CalibrationData:
    """Stores calibration session data"""
    timestamp: str
    tracker_type: str
    screen_resolution: Tuple[int, int]
    points: List[CalibrationPoint]
    success: bool
    accuracy: Optional[float] = None
    precision: Optional[float] = None


class CalibrationManager:
    """
    Manages calibration procedures and data
    
    Provides:
    - Calibration point management
    - Calibration data persistence
    - Accuracy/precision calculation
    - Calibration validation
    """
    
    def __init__(
        self,
        tracker: PyGazeTracker,
        calibration_file: Optional[str] = None
    ):
        """
        Initialize calibration manager
        
        Args:
            tracker: PyGazeTracker instance
            calibration_file: Path to save/load calibration data
        """
        self.tracker = tracker
        self.calibration_file = calibration_file
        self.logger = logging.getLogger(__name__)
        
        self.current_calibration: Optional[CalibrationData] = None
        self.calibration_points: List[CalibrationPoint] = []
    
    def perform_calibration(
        self,
        validate: bool = True,
        max_attempts: int = 3
    ) -> bool:
        """
        Perform full calibration procedure
        
        Args:
            validate: Whether to validate calibration after completion
            max_attempts: Maximum number of calibration attempts
        
        Returns:
            True if calibration successful
        """
        for attempt in range(max_attempts):
            self.logger.info(f"Calibration attempt {attempt + 1}/{max_attempts}")
            
            # Perform calibration
            success = self.tracker.calibrate()
            
            if success:
                if validate:
                    # Perform validation
                    if self.validate_calibration():
                        self.logger.info("Calibration and validation successful")
                        return True
                    else:
                        self.logger.warning("Calibration succeeded but validation failed")
                else:
                    return True
            else:
                self.logger.warning(f"Calibration attempt {attempt + 1} failed")
        
        self.logger.error("Calibration failed after all attempts")
        return False
    
    def perform_drift_correction(
        self,
        pos: Optional[Tuple[float, float]] = None,
        fix_triggered: bool = False,
        max_attempts: int = 3
    ) -> bool:
        """
        Perform drift correction
        
        Args:
            pos: (x, y) position for drift correction (None for center)
            fix_triggered: Use fixation-triggered drift correction
            max_attempts: Maximum number of attempts
        
        Returns:
            True if drift correction successful
        """
        for attempt in range(max_attempts):
            self.logger.info(f"Drift correction attempt {attempt + 1}/{max_attempts}")
            
            success = self.tracker.drift_correction(
                pos=pos,
                fix_triggered=fix_triggered
            )
            
            if success:
                self.logger.info("Drift correction successful")
                return True
            else:
                self.logger.warning(f"Drift correction attempt {attempt + 1} failed")
        
        self.logger.error("Drift correction failed after all attempts")
        return False
    
    def validate_calibration(
        self,
        validation_points: Optional[List[Tuple[float, float]]] = None,
        tolerance: float = 2.0
    ) -> bool:
        """
        Validate calibration accuracy
        
        Args:
            validation_points: List of (x, y) points to validate
            tolerance: Maximum acceptable error in degrees
        
        Returns:
            True if calibration is within tolerance
        """
        if validation_points is None:
            # Default validation points (center and corners)
            screen = self.tracker.display
            if hasattr(screen, 'dispsize'):
                width, height = screen.dispsize
            else:
                width, height = 1920, 1080  # Default
            
            validation_points = [
                (width / 2, height / 2),  # Center
                (width * 0.1, height * 0.1),  # Top-left
                (width * 0.9, height * 0.9),  # Bottom-right
            ]
        
        errors = []
        
        for target_x, target_y in validation_points:
            # Perform drift correction at this point
            if not self.tracker.drift_correction(pos=(target_x, target_y)):
                self.logger.warning(f"Drift correction failed at ({target_x}, {target_y})")
                return False
            
            # Sample gaze position
            sample = self.tracker.sample()
            if sample and sample.valid:
                # Calculate error (in pixels, convert to degrees if needed)
                error = ((sample.x - target_x)**2 + (sample.y - target_y)**2)**0.5
                errors.append(error)
            else:
                self.logger.warning(f"Could not get valid sample at ({target_x}, {target_y})")
                return False
        
        # Calculate average error
        avg_error = sum(errors) / len(errors) if errors else float('inf')
        
        # Convert pixels to degrees (rough approximation: 1 degree ≈ 30-40 pixels at typical viewing distance)
        # This is a rough estimate and depends on screen size and viewing distance
        error_degrees = avg_error / 35.0  # Approximate conversion
        
        self.logger.info(f"Calibration validation: average error = {avg_error:.2f} pixels ({error_degrees:.2f} degrees)")
        
        return error_degrees <= tolerance
    
    def save_calibration(self, filepath: Optional[str] = None) -> bool:
        """
        Save calibration data to file
        
        Args:
            filepath: Path to save file (uses default if None)
        
        Returns:
            True if save successful
        """
        if self.current_calibration is None:
            self.logger.warning("No calibration data to save")
            return False
        
        filepath = filepath or self.calibration_file
        if filepath is None:
            self.logger.warning("No calibration file path specified")
            return False
        
        try:
            # Convert to dict for JSON serialization
            data = asdict(self.current_calibration)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Calibration data saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: Optional[str] = None) -> bool:
        """
        Load calibration data from file
        
        Args:
            filepath: Path to load file (uses default if None)
        
        Returns:
            True if load successful
        """
        filepath = filepath or self.calibration_file
        if filepath is None or not os.path.exists(filepath):
            self.logger.warning(f"Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct calibration data
            points = [CalibrationPoint(**p) for p in data['points']]
            self.current_calibration = CalibrationData(
                timestamp=data['timestamp'],
                tracker_type=data['tracker_type'],
                screen_resolution=tuple(data['screen_resolution']),
                points=points,
                success=data['success'],
                accuracy=data.get('accuracy'),
                precision=data.get('precision')
            )
            
            self.logger.info(f"Calibration data loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading calibration: {e}")
            return False
    
    def get_calibration_info(self) -> Optional[Dict]:
        """
        Get information about current calibration
        
        Returns:
            Dictionary with calibration information or None
        """
        if self.current_calibration is None:
            return None
        
        return {
            'timestamp': self.current_calibration.timestamp,
            'tracker_type': self.current_calibration.tracker_type,
            'screen_resolution': self.current_calibration.screen_resolution,
            'success': self.current_calibration.success,
            'accuracy': self.current_calibration.accuracy,
            'precision': self.current_calibration.precision,
            'num_points': len(self.current_calibration.points)
        }
