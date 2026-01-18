"""
Main entry point for Adaptive UI System
Implements the Sense-Analyze-Adapt loop
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
from collections import deque
import time

# Project root (…/adaptiveUi). This makes calibration paths stable even if you run from another cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure we can import from src regardless of where script is run from
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker
from src.data_acquisition.monitor_plane_gaze_tracker import (
    MonitorPlaneGazeTracker,
    MonitorGazeConfig,
)
from src.data_acquisition.emotion_detector import EmotionDetector
from src.data_acquisition.mouse_tracker import MouseTracker, PYNPUT_AVAILABLE
from src.adaptation_engine.cognitive_load_monitor import CognitiveLoadMonitor
from src.adaptation_engine.ui_adapter import UIAdapter
from src.metrics.cognitive_load_metrics import MetricsCollector
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.gaze_calibration import GazeCalibrator
from src import constants as const


class AdaptiveUISystem:
    """
    Main system implementing Sense-Analyze-Adapt loop
    """
    
    def __init__(
        self,
        camera_index: Optional[int] = None,
        use_kalman: Optional[bool] = None,
        emotion_model_path: Optional[str] = None,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize Adaptive UI System

        Args:
            camera_index: Camera device index (overrides config)
            use_kalman: Whether to use Kalman filter for gaze smoothing (overrides config)
            emotion_model_path: Path to TensorFlow Lite emotion model (overrides config)
            config_path: Path to configuration YAML file
        """
        # Load configuration
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            self.config = {}

        # Setup logging
        logging_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name="adaptive_ui",
            log_level=logging_config.get('level', 'INFO'),
            log_dir=logging_config.get('log_directory', 'logs'),
            log_file=logging_config.get('log_file', None),
            console_output=True
        )

        self.logger.info("Initializing Adaptive UI System")

        # Apply configuration with parameter overrides
        camera_index = camera_index if camera_index is not None else self.config.get('camera', {}).get('index', 0)
        use_kalman = use_kalman if use_kalman is not None else self.config.get('gaze_tracking', {}).get('use_kalman', True)
        emotion_model_path = emotion_model_path if emotion_model_path is not None else self.config.get('emotion_detection', {}).get('model_path', None)
        # Initialize components
        gaze_tracking_config = self.config.get('gaze_tracking', {})
        gaze_mode = str(gaze_tracking_config.get('mode', 'direct')).strip().lower()
        self.gaze_mode = gaze_mode

        if gaze_mode == 'monitor_plane':
            mp_cfg = gaze_tracking_config.get('monitor_plane', {}) or {}
            monitor_cfg = MonitorGazeConfig(
                yaw_degrees=float(mp_cfg.get('yaw_degrees', 15.0)),
                pitch_degrees=float(mp_cfg.get('pitch_degrees', 5.0)),
                direction_filter_length=int(mp_cfg.get('direction_filter_length', 15)),
                screen_smoothing_length=int(mp_cfg.get('screen_smoothing_length', 8)),
                monitor_width=int(mp_cfg.get('monitor_width', 1920)),
                monitor_height=int(mp_cfg.get('monitor_height', 1080)),
                auto_lock_eye_spheres=bool(mp_cfg.get('auto_lock_eye_spheres', True)),
                auto_lock_frames_required=int(mp_cfg.get('auto_lock_frames_required', 20)),
                calibration_file=str(PROJECT_ROOT / "config" / "monitor_plane_calibration.json"),
            )
            self.gaze_tracker = MonitorPlaneGazeTracker(config=monitor_cfg)
            self.logger.info("Using Monitor Plane Gaze Tracker (ported monitor tracking)")
        else:
            # Default: DirectGazeTracker for best accuracy (iris-based mapping in 2D)
            self.gaze_tracker = DirectGazeTracker()
            # Sensitivity is already calibrated in DirectGazeTracker initialization
            # Can be adjusted via: tracker.set_sensitivity(scale_x=..., scale_y=...)
            self.logger.info("Using Direct Gaze Tracker (iris-based - most accurate)")
        # Initialize emotion detector with config
        emotion_config = self.config.get('emotion_detection', {})
        self.emotion_detector = EmotionDetector(
            model_path=emotion_model_path,
            use_vit=True,  # Use ViT model by default
            use_fer=False,  # Prefer ViT over FER
            temperature_scaling=emotion_config.get('temperature_scaling', 0.7),
            enable_neutral_bias_reduction=emotion_config.get('enable_neutral_bias_reduction', True),
            neutral_bias_threshold=emotion_config.get('neutral_bias_threshold', 0.35),
            neutral_bias_reduction_amount=emotion_config.get('neutral_bias_reduction_amount', 0.15)
        )

        # Initialize cognitive load monitor with config
        cognitive_load_config = self.config.get('cognitive_load', {})
        self.cognitive_load_monitor = CognitiveLoadMonitor(
            gaze_wander_threshold=cognitive_load_config.get('gaze_wander_threshold', 0.3),
            fixation_duration_threshold=cognitive_load_config.get('fixation_duration_threshold', 3.0),
            negative_affect_threshold=cognitive_load_config.get('negative_affect_threshold', 0.5),
            gaze_history_window=cognitive_load_config.get('gaze_history_window', 30)
        )

        self.ui_adapter = UIAdapter()
        self.metrics_collector = MetricsCollector()
        
        # Mouse tracker for CLI calculation
        self.mouse_tracker: Optional[MouseTracker] = None
        if PYNPUT_AVAILABLE:
            self.mouse_tracker = MouseTracker()
            self.logger.info("Mouse tracker initialized")
        else:
            self.logger.warning("Mouse tracking disabled (pynput not installed)")
        
        # Camera
        self.camera_index = camera_index
        self.camera: Optional[cv2.VideoCapture] = None
        
        # System state
        self.running = False
        self.frame_count = 0
        self._gaze_history = deque(maxlen=const.GAZE_TRAIL_LENGTH)  # For smooth pointer trail (fixed-size deque)

        # Frame rate control
        self.target_fps = self.config.get('camera', {}).get('fps', const.DEFAULT_TARGET_FPS)
        self.last_frame_time = 0

        # Gaze calibration
        self.calibrator = GazeCalibrator(calibration_path=str(PROJECT_ROOT / "config" / "gaze_calibration.json"))
        # Auto-enable saved calibration for direct mode
        self.use_calibration = bool(self.calibrator.is_loaded) and self.gaze_mode == "direct"

    def set_gaze_mode(self, mode: str, monitor_plane_config: Optional[dict] = None) -> bool:
        """
        Switch gaze tracker at runtime (used by WebSocket server / extension).

        Args:
            mode: "direct" or "monitor_plane"
            monitor_plane_config: optional dict of MonitorGazeConfig fields

        Returns:
            True if switched successfully, False otherwise.
        """
        mode = str(mode).strip().lower()
        try:
            if mode == 'monitor_plane':
                cfg = monitor_plane_config or {}
                monitor_cfg = MonitorGazeConfig(
                    yaw_degrees=float(cfg.get('yaw_degrees', 15.0)),
                    pitch_degrees=float(cfg.get('pitch_degrees', 5.0)),
                    direction_filter_length=int(cfg.get('direction_filter_length', 15)),
                    screen_smoothing_length=int(cfg.get('screen_smoothing_length', 8)),
                    monitor_width=int(cfg.get('monitor_width', 1920)),
                    monitor_height=int(cfg.get('monitor_height', 1080)),
                    auto_lock_eye_spheres=bool(cfg.get('auto_lock_eye_spheres', True)),
                    auto_lock_frames_required=int(cfg.get('auto_lock_frames_required', 20)),
                    calibration_file=str(PROJECT_ROOT / "config" / "monitor_plane_calibration.json"),
                )
                self.gaze_tracker = MonitorPlaneGazeTracker(config=monitor_cfg)
                self.gaze_mode = 'monitor_plane'
                self.use_calibration = False  # direct-only calibration
            else:
                self.gaze_tracker = DirectGazeTracker()
                self.gaze_mode = 'direct'
                # If a saved 9-point calibration exists, use it automatically
                self.use_calibration = bool(self.calibrator.is_loaded)

            # Initialize the newly selected tracker if system is already running
            if hasattr(self.gaze_tracker, 'setup_mediapipe'):
                ok = self.gaze_tracker.setup_mediapipe()
                if not ok:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch gaze mode to {mode}: {e}", exc_info=True)
            return False
        
    def initialize(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")

        # Setup MediaPipe
        try:
            self.gaze_tracker.setup_mediapipe()
            self.logger.info("MediaPipe initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

        # Setup emotion detection (deep learning model)
        try:
            self.emotion_detector.initialize()
            self.logger.info("Emotion detector initialized successfully")
        except Exception as e:
            self.logger.warning(f"Emotion detector initialization issue: {e}")

        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_index}")
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Start mouse tracking
        if self.mouse_tracker:
            self.mouse_tracker.start()
            self.logger.info("Mouse tracking started")

        self.logger.info("System initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the Sense-Analyze-Adapt loop

        Args:
            frame: Input video frame

        Returns:
            Dictionary with processing results and adaptation commands
        """
        self.frame_count += 1

        # SENSE: Data Acquisition
        try:
            gaze_coords = self.gaze_tracker.get_gaze(frame)
            # Apply calibration if available
            if gaze_coords is not None and self.use_calibration:
                gaze_coords = self.calibrator.apply_calibration(gaze_coords[0], gaze_coords[1])
        except Exception as e:
            self.logger.error(f"Error in gaze tracking: {e}", exc_info=True)
            gaze_coords = None

        try:
            emotion_result = self.emotion_detector.detect_emotion(frame)
        except Exception as e:
            self.logger.error(f"Error in emotion detection: {e}", exc_info=True)
            emotion_result = {
                'emotions': {},
                'dominant_emotion': 'neutral',
                'negative_affect_score': 0.0,
                'is_stressed': False,
                'face_detected': False
            }

        # SENSE: Mouse Data
        mouse_metrics = None
        if self.mouse_tracker:
            try:
                mouse_metrics = self.mouse_tracker.get_metrics()
            except Exception as e:
                self.logger.debug(f"Error getting mouse metrics: {e}")

        # ANALYZE: Cognitive Load Monitoring
        try:
            if gaze_coords is not None:
                x, y = gaze_coords
                self.cognitive_load_monitor.update_gaze(x, y)

            self.cognitive_load_monitor.update_emotion(emotion_result)
            
            # Update mouse metrics in cognitive load monitor
            if mouse_metrics and self.mouse_tracker:
                mx, my = self.mouse_tracker.current_x, self.mouse_tracker.current_y
                # Check for clicks in the last frame (approximation)
                recent_clicks = self.mouse_tracker.get_recent_clicks(0.1)
                clicked = len(recent_clicks) > 0
                is_error = any(c.is_rapid_click for c in recent_clicks)
                self.cognitive_load_monitor.update_mouse(mx, my, clicked=clicked, is_error_click=is_error)
            
            overload_status = self.cognitive_load_monitor.check_cognitive_overload()
            
            # Add mouse metrics to overload status for display
            if mouse_metrics:
                overload_status['mouse_metrics'] = {
                    'click_rate': mouse_metrics.click_rate,
                    'error_click_rate': mouse_metrics.error_click_rate,
                    'path_efficiency': mouse_metrics.path_efficiency,
                    'hover_dwell': mouse_metrics.avg_hover_dwell,
                    'load_score': mouse_metrics.load_score
                }
        except Exception as e:
            self.logger.error(f"Error in cognitive load monitoring: {e}", exc_info=True)
            overload_status = {
                'overload_detected': False,
                'cognitive_load_score': 0.0,
                'triggers': {}
            }

        # ADAPT: UI Adaptation
        try:
            adaptations = self.ui_adapter.generate_adaptations(overload_status)
            adaptation_commands = self.ui_adapter.apply_adaptations(adaptations)
        except Exception as e:
            self.logger.error(f"Error in UI adaptation: {e}", exc_info=True)
            adaptations = []
            adaptation_commands = {}

        # Record metrics
        try:
            if overload_status.get('overload_detected', False):
                self.metrics_collector.record_adaptation_trigger(overload_status)
        except Exception as e:
            self.logger.warning(f"Error recording metrics: {e}")

        return {
            'frame_count': self.frame_count,
            'gaze_coords': gaze_coords,
            'emotion': emotion_result,
            'overload_status': overload_status,
            'adaptations': [a.value for a in adaptations] if adaptations else [],
            'adaptation_commands': adaptation_commands
        }
    
    def run(self, display: bool = True, max_frames: Optional[int] = None, calibrate: bool = False):
        """
        Run the main processing loop

        Args:
            display: Whether to display video feed (set False for headless operation)
            max_frames: Maximum number of frames to process (None = unlimited)
            calibrate: Whether to run calibration before starting (default: False)
        """
        if self.camera is None:
            self.initialize()
        
        # Verify camera is working
        if self.camera is None:
            self.logger.error("Camera not initialized!")
            self.logger.error("Please check camera connection and permissions.")
            return

        ret, test_frame = self.camera.read()
        if not ret:
            self.logger.error("Cannot read from camera!")
            self.logger.error("Please check:")
            self.logger.error("  1. Camera is connected")
            self.logger.error("  2. Camera permissions are granted")
            self.logger.error("  3. No other application is using the camera")
            return

        # Run calibration if requested
        if calibrate and display:
            # For monitor-plane tracker: center-calibration (yaw/pitch offsets).
            if hasattr(self.gaze_tracker, "calibrate_center"):
                self.logger.info("Starting monitor-center calibration...")
                ret, calib_frame = self.camera.read()
                if ret:
                    calib_frame = cv2.flip(calib_frame, 1)
                    success = bool(self.gaze_tracker.calibrate_center(calib_frame))  # type: ignore[attr-defined]
                else:
                    success = False
                if success:
                    self.logger.info("Monitor-center calibration complete!")
                else:
                    self.logger.warning("Monitor-center calibration failed - continuing uncalibrated")
            else:
                # For direct tracker: 9-point calibration
                self.logger.info("Starting gaze calibration...")
                success = self.calibrator.run_calibration(self.gaze_tracker, self.camera)
                if success:
                    self.use_calibration = True
                    self.logger.info("Calibration complete - tracking will be more accurate!")
                else:
                    self.logger.warning("Calibration cancelled or failed - using uncalibrated tracking")

        self.running = True
        frames_processed = 0

        self.logger.info("Starting Adaptive UI System...")
        if display:
            self.logger.info("Press 'q' to quit, 'c' to calibrate")
            self.logger.info("\nLook for on-screen indicators:")
            self.logger.info("  - 'EYE TRACKING: ACTIVE' (green) = Working!")
            self.logger.info("  - 'EYE TRACKING: INACTIVE' (red) = Need better lighting/positioning")
            self.logger.info("  - Pointer will appear when face is detected")
            if self.use_calibration:
                self.logger.info("  - CALIBRATED mode active for better accuracy")
        else:
            self.logger.info("Running in headless mode (no display)")
            if max_frames:
                self.logger.info(f"Will process {max_frames} frames")

        try:
            while self.running:
                # Check max frames limit
                if max_frames is not None and frames_processed >= max_frames:
                    self.logger.info(f"Reached max frames limit: {max_frames}")
                    break

                # Frame rate control
                current_time = time.time()
                time_since_last_frame = current_time - self.last_frame_time

                # Calculate target frame interval
                target_interval = 1.0 / self.target_fps

                # Skip frame if we're processing too fast
                if time_since_last_frame < target_interval:
                    time.sleep(target_interval - time_since_last_frame)
                    continue

                self.last_frame_time = current_time

                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break

                # Mirror the frame horizontally so right hand appears on right side
                frame = cv2.flip(frame, 1)

                # Process frame
                result = self.process_frame(frame)
                frames_processed += 1

                # Log progress periodically in headless mode
                if not display and frames_processed % 30 == 0:
                    self.logger.info(f"Processed {frames_processed} frames")

                # Log adaptation triggers
                if result['overload_status']['overload_detected']:
                    self.logger.debug(f"Cognitive overload detected: {result['overload_status']}")

                # Display results (only if display is enabled)
                if display:
                    display_frame = self._draw_results(frame, result)
                    cv2.imshow('Adaptive UI System', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        # Run calibration
                        if hasattr(self.gaze_tracker, "calibrate_center"):
                            self.logger.info("Starting monitor-center calibration...")
                            success = bool(self.gaze_tracker.calibrate_center(frame))  # type: ignore[attr-defined]
                            if success:
                                self.logger.info("Monitor-center calibration complete!")
                            else:
                                self.logger.warning("Monitor-center calibration failed")
                        else:
                            self.logger.info("Starting calibration...")
                            success = self.calibrator.run_calibration(self.gaze_tracker, self.camera)
                            if success:
                                self.use_calibration = True
                                self.logger.info("Calibration complete!")
                            else:
                                self.logger.warning("Calibration cancelled")

        except KeyboardInterrupt:
            self.logger.info("\nStopping system...")
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _draw_results(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw processing results on frame (visualization only - no processing)

        Args:
            frame: Input frame
            result: Processing results from process_frame()

        Returns:
            Annotated frame with visualization overlays
        """
        display_frame = frame.copy()
        
        # Draw eye tracking indicator on top-right side
        h, w = frame.shape[:2]
        
        if result['gaze_coords'] is not None:
            # Face detected and gaze calculated
            x, y = result['gaze_coords']
            # Calculate text position (right-aligned)
            text = "EYE TRACKING: ACTIVE"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, const.TEXT_FONT, const.TEXT_FONT_SCALE_MEDIUM, const.TEXT_THICKNESS_BOLD
            )
            text_x = w - text_width - const.TEXT_MARGIN_TOP_RIGHT
            text_y = const.TEXT_Y_EMOTION

            cv2.putText(display_frame, text, (text_x, text_y),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_MEDIUM, const.COLOR_GREEN, const.TEXT_THICKNESS_BOLD)

            # Show gaze coordinates below
            coord_text = f"Gaze: ({x:.2f}, {y:.2f})"
            (coord_width, coord_height), _ = cv2.getTextSize(
                coord_text, const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.TEXT_THICKNESS_NORMAL
            )
            coord_x = w - coord_width - const.TEXT_MARGIN_TOP_RIGHT
            cv2.putText(display_frame, coord_text, (coord_x, text_y + 25),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_WHITE, const.TEXT_THICKNESS_NORMAL)
        else:
            # No face detected or gaze not available
            text = "EYE TRACKING: INACTIVE"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, const.TEXT_FONT, const.TEXT_FONT_SCALE_MEDIUM, const.TEXT_THICKNESS_BOLD
            )
            text_x = w - text_width - const.TEXT_MARGIN_TOP_RIGHT
            text_y = const.TEXT_Y_EMOTION

            cv2.putText(display_frame, text, (text_x, text_y),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_MEDIUM, const.COLOR_RED, const.TEXT_THICKNESS_BOLD)

            # Show tips below
            tips_text = "Look at camera"
            (tips_width, tips_height), _ = cv2.getTextSize(
                tips_text, const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.TEXT_THICKNESS_NORMAL
            )
            tips_x = w - tips_width - const.TEXT_MARGIN_TOP_RIGHT
            cv2.putText(display_frame, tips_text, (tips_x, text_y + 25),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_WHITE, const.TEXT_THICKNESS_NORMAL)

        # Draw calibration status
        if self.use_calibration:
            cal_text = "CALIBRATED"
            (cal_width, cal_height), _ = cv2.getTextSize(
                cal_text, const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.TEXT_THICKNESS_BOLD
            )
            cal_x = w - cal_width - const.TEXT_MARGIN_TOP_RIGHT
            cal_y = text_y + 50
            cv2.putText(display_frame, cal_text, (cal_x, cal_y),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_LIGHT_GREEN, const.TEXT_THICKNESS_BOLD)

        # Draw responsive gaze pointer that follows eye movements accurately
        if result['gaze_coords'] is not None:
            x, y = result['gaze_coords']
            h, w = frame.shape[:2]
            
            # Map normalized coordinates [0, 1] to screen pixels
            # Clamp to ensure pointer stays on screen
            px = int(max(0, min(w - 1, x * w)))
            py = int(max(0, min(h - 1, y * h)))
            
            # Track if pointer position is changing
            if not hasattr(self, '_last_pointer_pos'):
                self._last_pointer_pos = (px, py)
                self._pointer_static_count = 0
            else:
                if abs(px - self._last_pointer_pos[0]) < const.CAMERA_TEST_POSITION_THRESHOLD and \
                   abs(py - self._last_pointer_pos[1]) < const.CAMERA_TEST_POSITION_THRESHOLD:
                    self._pointer_static_count += 1
                else:
                    self._pointer_static_count = 0
                self._last_pointer_pos = (px, py)

            # Warning if pointer isn't moving
            if self._pointer_static_count > const.POINTER_STATIC_THRESHOLD_FRAMES:
                warning_y = h - const.TEXT_Y_WARNING
                cv2.putText(display_frame, "WARNING: Pointer not moving!",
                           (const.TEXT_MARGIN_X, warning_y),
                           const.TEXT_FONT, const.TEXT_FONT_SCALE_MEDIUM, const.COLOR_RED, const.TEXT_THICKNESS_BOLD)
            
            # Add current position to history (deque automatically maintains max size)
            self._gaze_history.append((px, py))
            
            # Draw trail (fading path showing gaze movement)
            for i in range(1, len(self._gaze_history)):
                prev_x, prev_y = self._gaze_history[i-1]
                curr_x, curr_y = self._gaze_history[i]
                # Fade trail based on age (older = more transparent)
                alpha = i / len(self._gaze_history)
                trail_color = (0, int(220 * alpha), int(100 * alpha))  # Green to cyan trail
                thickness = max(1, int(4 * alpha))
                cv2.line(display_frame, (prev_x, prev_y), (curr_x, curr_y),
                        trail_color, thickness)

            # Draw gaze point indicator (HIGHLY VISIBLE)

            # 1. Large semi-transparent background circle
            overlay = display_frame.copy()
            cv2.circle(overlay, (px, py), const.POINTER_INDICATOR_SIZE, (0, 255, 100), -1)
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)

            # 2. Outer glow rings (larger and more prominent)
            cv2.circle(display_frame, (px, py), const.POINTER_OUTER_RADIUS + 10, (0, 255, 255), 3)
            cv2.circle(display_frame, (px, py), const.POINTER_OUTER_RADIUS, (0, 255, 200), 4)
            cv2.circle(display_frame, (px, py), const.POINTER_MIDDLE_RADIUS, (0, 255, 150), 5)

            # 3. Inner target circle (bright yellow/cyan)
            cv2.circle(display_frame, (px, py), const.POINTER_INNER_RADIUS, (0, 255, 255), 6)

            # 4. THICK CROSSHAIR (white with black outline for maximum visibility)
            # Horizontal line
            cv2.line(display_frame, (px - const.POINTER_CROSSHAIR_LENGTH, py),
                    (px + const.POINTER_CROSSHAIR_LENGTH, py), (0, 0, 0), const.POINTER_CROSSHAIR_THICKNESS + 2)
            cv2.line(display_frame, (px - const.POINTER_CROSSHAIR_LENGTH, py),
                    (px + const.POINTER_CROSSHAIR_LENGTH, py), (255, 255, 255), const.POINTER_CROSSHAIR_THICKNESS)
            # Vertical line
            cv2.line(display_frame, (px, py - const.POINTER_CROSSHAIR_LENGTH),
                    (px, py + const.POINTER_CROSSHAIR_LENGTH), (0, 0, 0), const.POINTER_CROSSHAIR_THICKNESS + 2)
            cv2.line(display_frame, (px, py - const.POINTER_CROSSHAIR_LENGTH),
                    (px, py + const.POINTER_CROSSHAIR_LENGTH), (255, 255, 255), const.POINTER_CROSSHAIR_THICKNESS)

            # 5. Center bullseye (bright cyan filled circle)
            cv2.circle(display_frame, (px, py), const.POINTER_CENTER_RADIUS, (255, 255, 0), -1)
            cv2.circle(display_frame, (px, py), const.POINTER_CENTER_RADIUS, (0, 0, 0), 3)

            # 6. LARGE RED CENTER DOT (exact gaze point - VERY VISIBLE)
            cv2.circle(display_frame, (px, py), const.POINTER_DOT_RADIUS, (0, 0, 255), -1)
            cv2.circle(display_frame, (px, py), const.POINTER_DOT_RADIUS, (255, 255, 255), 2)  # White outline

            # 7. Display pixel coordinates with larger text
            coord_label = f"({px}, {py})"
            label_offset_x = 50
            label_offset_y = -30
            # Black outline for readability
            cv2.putText(display_frame, coord_label,
                       (px + label_offset_x, py + label_offset_y),
                       const.TEXT_FONT, 0.7, (0, 0, 0), 4)
            # Bright cyan text
            cv2.putText(display_frame, coord_label,
                       (px + label_offset_x, py + label_offset_y),
                       const.TEXT_FONT, 0.7, (0, 255, 255), 2)
        
        # Draw emotion
        emotion = result['emotion']
        emotion_text = f"Emotion: {emotion['dominant_emotion']} (Neg: {emotion['negative_affect_score']:.2f})"
        cv2.putText(display_frame, emotion_text, (const.TEXT_MARGIN_X, const.TEXT_Y_EMOTION),
                   const.TEXT_FONT, const.TEXT_FONT_SCALE_LARGE, const.COLOR_WHITE, const.TEXT_THICKNESS_BOLD)

        # Draw enhanced cognitive load (CLI)
        overload = result['overload_status']
        cli = overload.get('cli', overload['cognitive_load_score'])
        load_level = overload.get('load_level', 'unknown')
        
        # Color based on load level
        if load_level == 'low':
            color = const.COLOR_GREEN
        elif load_level == 'medium':
            color = (0, 200, 255)  # Orange
        else:
            color = const.COLOR_RED
        
        load_text = f"Cognitive Load: {cli:.2f} ({load_level.upper()})"
        cv2.putText(display_frame, load_text, (const.TEXT_MARGIN_X, const.TEXT_Y_COGNITIVE_LOAD),
                   const.TEXT_FONT, const.TEXT_FONT_SCALE_LARGE, color, const.TEXT_THICKNESS_BOLD)

        # Draw CLI component breakdown
        gaze_score = overload.get('gaze_score', 0.0)
        emotion_score = overload.get('emotion_score', 0.0)
        mouse_score = overload.get('mouse_score', 0.0)
        
        components_text = f"G:{gaze_score:.2f} E:{emotion_score:.2f} M:{mouse_score:.2f}"
        cv2.putText(display_frame, components_text, (const.TEXT_MARGIN_X, const.TEXT_Y_TRIGGERS),
                   const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_GRAY, const.TEXT_THICKNESS_NORMAL)
        
        # Draw detailed metrics if available
        details = overload.get('details', {})
        if details:
            gaze_details = details.get('gaze', {})
            emotion_details = details.get('emotion', {})
            
            # Fixation and saccade info
            fix_dur = gaze_details.get('fixation_duration', 0.0)
            sacc_freq = gaze_details.get('saccade_frequency', 0.0)
            detail_text = f"Fix:{fix_dur:.1f}s Sacc:{sacc_freq:.1f}/s"
            cv2.putText(display_frame, detail_text, (const.TEXT_MARGIN_X + 200, const.TEXT_Y_TRIGGERS),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_GRAY, const.TEXT_THICKNESS_NORMAL)
        
        # Draw mouse metrics if available
        mouse_metrics = overload.get('mouse_metrics', {})
        if mouse_metrics:
            click_rate = mouse_metrics.get('click_rate', 0.0)
            path_eff = mouse_metrics.get('path_efficiency', 1.0)
            mouse_text = f"Clicks:{click_rate:.1f}/s Path:{path_eff:.0%}"
            cv2.putText(display_frame, mouse_text, (const.TEXT_MARGIN_X + 400, const.TEXT_Y_TRIGGERS),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_GRAY, const.TEXT_THICKNESS_NORMAL)

        # Draw adaptations (fixed position to avoid overlap)
        if result['adaptations']:
            adapt_text = f"Adaptations: {', '.join(result['adaptations'])}"
            cv2.putText(display_frame, adapt_text, (const.TEXT_MARGIN_X, const.TEXT_Y_ADAPTATIONS),
                       const.TEXT_FONT, const.TEXT_FONT_SCALE_SMALL, const.COLOR_ADAPTATION, const.TEXT_THICKNESS_BOLD)
        
        return display_frame
    
    def cleanup(self):
        """Cleanup resources and flush logs"""
        # Stop mouse tracker
        if self.mouse_tracker:
            self.mouse_tracker.stop()
        
        # Flush gaze tracker logs if it has logging
        if hasattr(self.gaze_tracker, 'close_logging'):
            self.gaze_tracker.close_logging()
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("System stopped.")


def main():
    """Main entry point"""
    try:
        system = AdaptiveUISystem(
            camera_index=None,  # Will use config value
            use_kalman=None,    # Will use config value
            emotion_model_path=None  # Will use config value
        )
        system.run(display=True)
    except Exception as e:
        # If logger not initialized, fall back to print
        try:
            logger = get_logger("adaptive_ui")
            logger.error(f"Fatal error: {e}", exc_info=True)
        except:
            print(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'system' in locals():
                system.cleanup()


if __name__ == "__main__":
    main()

