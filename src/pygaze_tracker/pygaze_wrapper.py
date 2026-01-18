"""
PyGaze Wrapper - Simplified interface for PyGaze eye tracking

This module provides a high-level wrapper around PyGaze that simplifies
common eye tracking operations like calibration, recording, and event detection.
"""

import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from pygaze import libscreen
    from pygaze import libtime
    from pygaze import liblog
    from pygaze.libeyetracker import EyeTracker
    PYGAZE_AVAILABLE = True
    PYGAZE_ERROR = None
except ImportError as e:
    PYGAZE_AVAILABLE = False
    PYGAZE_ERROR = str(e)
    # Check if it's a PsychoPy dependency issue
    if 'psychopy' in str(e).lower() or 'psychopytime' in str(e).lower():
        logging.warning(
        "PyGaze requires PsychoPy. Install with: pip install psychopy\n"
        "Note: PsychoPy may require additional system dependencies (HDF5). "
        "See: https://www.psychopy.org/installation.html"
    )
    else:
        logging.warning(f"PyGaze not available: {e}\nInstall with: pip install python-pygaze")
except Exception as e:
    PYGAZE_AVAILABLE = False
    PYGAZE_ERROR = str(e)
    logging.warning(f"PyGaze import error: {e}")


@dataclass
class GazeSample:
    """Represents a single gaze sample"""
    x: float
    y: float
    timestamp: float
    pupil_size: Optional[float] = None
    valid: bool = True


@dataclass
class Fixation:
    """Represents a fixation event"""
    start_time: float
    end_time: float
    duration: float
    x: float
    y: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]


@dataclass
class Saccade:
    """Represents a saccade event"""
    start_time: float
    end_time: float
    duration: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    amplitude: float
    velocity: float


@dataclass
class Blink:
    """Represents a blink event"""
    start_time: float
    end_time: float
    duration: float


class PyGazeTracker:
    """
    High-level wrapper for PyGaze eye tracking
    
    This class simplifies common eye tracking operations:
    - Initialization and connection
    - Calibration
    - Recording
    - Event detection (fixations, saccades, blinks)
    - Data logging
    """
    
    def __init__(
        self,
        display: Optional[Any] = None,
        tracker: str = 'dummy',
        logfile: Optional[str] = None,
        eventdetection: str = 'pygaze',
        saccvelthresh: int = 35,
        saccaccthresh: int = 9500,
        fixtresh: float = 1.5,
        blinksize: int = 150,
        **kwargs
    ):
        """
        Initialize PyGaze tracker
        
        Args:
            display: Display object (if None, will create default)
            tracker: Eye tracker type ('dummy', 'eyelink', 'eyetribe', etc.)
            logfile: Path to log file (optional)
            eventdetection: 'pygaze' or 'native' for event detection
            saccvelthresh: Saccade velocity threshold
            saccaccthresh: Saccade acceleration threshold
            fixtresh: Fixation threshold (degrees)
            blinksize: Blink size threshold
            **kwargs: Additional arguments passed to EyeTracker
        """
        if not PYGAZE_AVAILABLE:
            error_msg = "PyGaze is not available."
            if PYGAZE_ERROR:
                if 'psychopy' in PYGAZE_ERROR.lower():
                    error_msg = (
                        "PyGaze requires PsychoPy to be installed.\n"
                        "Install with: pip install psychopy\n"
                        "Note: On macOS, you may need to install HDF5 first:\n"
                        "  brew install hdf5\n"
                        "Or use conda: conda install -c conda-forge psychopy\n"
                        f"Original error: {PYGAZE_ERROR}"
                    )
                else:
                    error_msg = f"PyGaze import failed: {PYGAZE_ERROR}\nInstall with: pip install python-pygaze"
            else:
                error_msg = "PyGaze is not installed. Install with: pip install python-pygaze"
            raise ImportError(error_msg)
        
        self.logger = logging.getLogger(__name__)
        self.tracker_type = tracker
        self.eventdetection = eventdetection
        
        # Initialize display if not provided
        if display is None:
            self.display = libscreen.Display()
        else:
            self.display = display
        
        # Initialize eye tracker
        try:
            self.eyetracker = EyeTracker(
                self.display,
                trackertype=tracker,
                logfile=logfile,
                eventdetection=eventdetection,
                saccvelthresh=saccvelthresh,
                saccaccthresh=saccaccthresh,
                fixtresh=fixtresh,
                blinksize=blinksize,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize eye tracker: {e}")
            raise
        
        # State tracking
        self.is_connected = False
        self.is_recording = False
        self.is_calibrated = False
        self.start_time = None
        
        # Event detection settings
        self.set_detection_type(eventdetection)
        
        self.logger.info(f"PyGaze tracker initialized (type: {tracker})")
    
    def connect(self) -> bool:
        """
        Connect to the eye tracker
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.eyetracker.connected():
                self.is_connected = True
                self.logger.info("Eye tracker connected")
                return True
            else:
                self.logger.warning("Eye tracker connection failed")
                self.is_connected = False
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to eye tracker: {e}")
            self.is_connected = False
            return False
    
    def calibrate(self) -> bool:
        """
        Perform eye tracker calibration
        
        Returns:
            True if calibration successful, False otherwise
        """
        if not self.is_connected:
            self.logger.warning("Not connected to eye tracker. Attempting connection...")
            if not self.connect():
                return False
        
        try:
            result = self.eyetracker.calibrate()
            self.is_calibrated = result
            if result:
                self.logger.info("Calibration successful")
            else:
                self.logger.warning("Calibration failed")
            return result
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            self.is_calibrated = False
            return False
    
    def drift_correction(
        self,
        pos: Optional[Tuple[float, float]] = None,
        fix_triggered: bool = False
    ) -> bool:
        """
        Perform drift correction
        
        Args:
            pos: (x, y) position for drift correction target (None for center)
            fix_triggered: Use fixation-triggered drift correction
        
        Returns:
            True if drift correction successful, False otherwise
        """
        if not self.is_connected:
            self.logger.warning("Not connected to eye tracker")
            return False
        
        try:
            if fix_triggered:
                result = self.eyetracker.fix_triggered_drift_correction(pos=pos)
            else:
                result = self.eyetracker.drift_correction(pos=pos, fix_triggered=False)
            
            if result:
                self.logger.info("Drift correction successful")
            else:
                self.logger.warning("Drift correction failed")
            return result
        except Exception as e:
            self.logger.error(f"Error during drift correction: {e}")
            return False
    
    def start_recording(self) -> bool:
        """
        Start recording eye tracking data
        
        Returns:
            True if recording started successfully
        """
        if not self.is_connected:
            self.logger.warning("Not connected to eye tracker")
            return False
        
        try:
            self.eyetracker.start_recording()
            self.is_recording = True
            self.start_time = libtime.get_time()
            self.logger.info("Recording started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording eye tracking data
        
        Returns:
            True if recording stopped successfully
        """
        if not self.is_recording:
            self.logger.warning("Not currently recording")
            return False
        
        try:
            self.eyetracker.stop_recording()
            self.is_recording = False
            self.logger.info("Recording stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            return False
    
    def sample(self) -> Optional[GazeSample]:
        """
        Get the most recent gaze sample
        
        Returns:
            GazeSample object or None if no valid sample
        """
        if not self.is_connected:
            return None
        
        try:
            x, y = self.eyetracker.sample()
            
            # Check if sample is valid (PyGaze returns (-1, -1) on error)
            if x == -1 and y == -1:
                return GazeSample(x=0, y=0, timestamp=0, valid=False)
            
            # Get pupil size if available
            pupil_size = None
            try:
                pupil_size = self.eyetracker.pupil_size()
            except:
                pass
            
            # Get timestamp
            timestamp = libtime.get_time() - (self.start_time or 0)
            
            return GazeSample(
                x=x,
                y=y,
                timestamp=timestamp,
                pupil_size=pupil_size,
                valid=True
            )
        except Exception as e:
            self.logger.debug(f"Error getting sample: {e}")
            return None
    
    def wait_for_fixation_start(self) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Wait for a fixation to start
        
        Returns:
            (time, (x, y)) tuple or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            time, gazepos = self.eyetracker.wait_for_fixation_start()
            return (time, gazepos)
        except Exception as e:
            self.logger.error(f"Error waiting for fixation start: {e}")
            return None
    
    def wait_for_fixation_end(self) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Wait for a fixation to end
        
        Returns:
            (time, (x, y)) tuple or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            time, gazepos = self.eyetracker.wait_for_fixation_end()
            return (time, gazepos)
        except Exception as e:
            self.logger.error(f"Error waiting for fixation end: {e}")
            return None
    
    def wait_for_saccade_start(self) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Wait for a saccade to start
        
        Returns:
            (time, (x, y)) tuple or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            time, startpos = self.eyetracker.wait_for_saccade_start()
            return (time, startpos)
        except Exception as e:
            self.logger.error(f"Error waiting for saccade start: {e}")
            return None
    
    def wait_for_saccade_end(self) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """
        Wait for a saccade to end
        
        Returns:
            (endtime, startpos, endpos) tuple or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            endtime, startpos, endpos = self.eyetracker.wait_for_saccade_end()
            return (endtime, startpos, endpos)
        except Exception as e:
            self.logger.error(f"Error waiting for saccade end: {e}")
            return None
    
    def wait_for_blink_start(self) -> Optional[float]:
        """
        Wait for a blink to start
        
        Returns:
            Blink start time or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            time = self.eyetracker.wait_for_blink_start()
            return time
        except Exception as e:
            self.logger.error(f"Error waiting for blink start: {e}")
            return None
    
    def wait_for_blink_end(self) -> Optional[float]:
        """
        Wait for a blink to end
        
        Returns:
            Blink end time or None on error
        """
        if not self.is_connected:
            return None
        
        try:
            time = self.eyetracker.wait_for_blink_end()
            return time
        except Exception as e:
            self.logger.error(f"Error waiting for blink end: {e}")
            return None
    
    def log(self, msg: str):
        """Log a message to the eye tracker log file"""
        if self.is_connected:
            self.eyetracker.log(msg)
    
    def log_var(self, var: str, val: Any):
        """Log a variable to the eye tracker log file"""
        if self.is_connected:
            self.eyetracker.log_var(var, val)
    
    def status_msg(self, msg: str):
        """Send status message to eye tracker (EyeLink only)"""
        if self.is_connected:
            try:
                self.eyetracker.status_msg(msg)
            except:
                pass  # Not all trackers support this
    
    def set_detection_type(self, eventdetection: str):
        """
        Set event detection type
        
        Args:
            eventdetection: 'pygaze' or 'native'
        """
        if self.is_connected:
            try:
                result = self.eyetracker.set_detection_type(eventdetection)
                self.eventdetection = eventdetection
                self.logger.info(f"Event detection set to: {eventdetection} ({result})")
            except Exception as e:
                self.logger.warning(f"Could not set detection type: {e}")
    
    def close(self):
        """Close connection to eye tracker"""
        if self.is_connected:
            try:
                self.eyetracker.close()
                self.is_connected = False
                self.is_recording = False
                self.logger.info("Eye tracker closed")
            except Exception as e:
                self.logger.error(f"Error closing eye tracker: {e}")
        
        if hasattr(self, 'display'):
            try:
                self.display.close()
            except:
                pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
