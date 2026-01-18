"""
Direct Iris-Based Gaze Tracker (Optimized)
Enhanced accuracy through:
- Head pose compensation
- Temporal filtering with velocity prediction
- Outlier rejection
- Blink detection
- Adaptive smoothing
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import math
import os
from datetime import datetime
from collections import deque
import csv


class DirectGazeTracker:
    """
    Optimized direct gaze tracker using iris position
    Features: Head pose compensation, temporal filtering, blink detection
    """
    
    def __init__(self):
        """Initialize direct gaze tracker"""
        self.face_landmarker = None
        self.initialized = False
        
        # Gaze state
        self._last_gaze_x = 0.5
        self._last_gaze_y = 0.5
        
        # Velocity tracking for prediction
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._last_timestamp = None
        
        # Gaze mapping parameters
        # CRITICAL: After eye-width normalization, iris offset is typically ±0.2 to ±0.5
        # To map this to screen coordinates (±0.5), we need scale of ~1.0 to 2.0
        # Previous values (17-50) were WAY too high, causing immediate saturation
        self.scale_x = 1.8    # Horizontal sensitivity (maps ±0.3 iris to ±0.5 screen)
        self.scale_y = 1.8    # Vertical sensitivity (matched with X)
        self.offset_x = 0.0
        self.offset_y = 0.0
        # Iris is naturally positioned slightly below eye center
        # This compensates for that anatomical offset
        self.normalized_y_offset = 0.05
        
        # Temporal filtering history
        self._gaze_history: deque = deque(maxlen=5)  # Rolling history for smoothing
        self._raw_history: deque = deque(maxlen=10)  # For outlier detection
        
        # Blink detection
        self._blink_threshold = 0.15  # Eye aspect ratio below this = blink
        self._is_blinking = False
        self._blink_frames = 0
        self._max_blink_frames = 5  # Use last good gaze during short blinks
        
        # Head pose tracking
        self._last_head_yaw = 0.0
        self._last_head_pitch = 0.0
        self._head_pose_weight = 0.3  # How much head pose affects gaze
        
        # Outlier rejection
        self._max_jump = 0.15  # Maximum allowed jump per frame (normalized)
        self._outlier_count = 0
        
        # Adaptive smoothing
        self._base_smoothing = 0.25
        self._min_smoothing = 0.1
        self._max_smoothing = 0.5
        
        # Debug/logging (disabled by default for production)
        self.frame_count = 0
        self.log_enabled = False
        self.debug_output = False  # Disabled for production
        self.log_data = []
        self.log_file = None
        
        if self.log_enabled:
            self._init_logging()
    
    def _init_logging(self):
        """Initialize logging for analysis"""
        if self.log_enabled:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"gaze_analysis_{timestamp}.csv")
            
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'iris_offset_x', 'iris_offset_y', 'eye_width_avg',
                    'normalized_x', 'normalized_y_raw', 'normalized_y_calibrated',
                    'gaze_x', 'gaze_y', 'smoothed_x', 'smoothed_y',
                    'head_yaw', 'head_pitch', 'is_blink', 'is_outlier'
                ])
            print(f"✓ Gaze logging initialized: {self.log_file}")
    
    def _log_data(self, **kwargs):
        """Log tracking data for analysis"""
        if self.log_enabled and self.log_file:
            self.log_data.append(kwargs)
            if len(self.log_data) >= 10:
                self._flush_log()
    
    def _flush_log(self):
        """Flush accumulated log data to file"""
        if self.log_enabled and self.log_file and self.log_data:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for entry in self.log_data:
                    writer.writerow([
                        entry.get('frame', 0),
                        entry.get('iris_offset_x', 0),
                        entry.get('iris_offset_y', 0),
                        entry.get('eye_width_avg', 0),
                        entry.get('normalized_x', 0),
                        entry.get('normalized_y_raw', 0),
                        entry.get('normalized_y_calibrated', 0),
                        entry.get('gaze_x', 0),
                        entry.get('gaze_y', 0),
                        entry.get('smoothed_x', 0),
                        entry.get('smoothed_y', 0),
                        entry.get('head_yaw', 0),
                        entry.get('head_pitch', 0),
                        entry.get('is_blink', False),
                        entry.get('is_outlier', False)
                    ])
            self.log_data = []
    
    def close_logging(self):
        """Close logging and flush remaining data"""
        if self.log_enabled:
            self._flush_log()
            if self.log_file:
                print(f"✓ Gaze log saved: {self.log_file}")
    
    def setup_mediapipe(self):
        """Initialize MediaPipe with optimized settings"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request
            
            model_path = 'face_landmarker.task'
            if not os.path.exists(model_path):
                print("Downloading MediaPipe model...")
                model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print("✓ Model downloaded")
                except Exception as e:
                    print(f"Failed to download: {e}")
                    model_path = None
            
            base_options = python.BaseOptions(
                model_asset_path=model_path if model_path and os.path.exists(model_path) else None,
                delegate=python.BaseOptions.Delegate.CPU
            )
            
            # Optimized detection settings
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,  # Enable for blink detection
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,  # Balanced threshold
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True  # Enable for head pose
            )
            
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.initialized = True
            print("✓ Optimized Gaze Tracker initialized")
            return True
            
        except Exception as e:
            print(f"MediaPipe init failed: {e}")
            self.initialized = False
            return False
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: List) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        if len(eye_landmarks) < 6:
            return 1.0
        
        # Vertical distances
        v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x)**2 + 
                       (eye_landmarks[1].y - eye_landmarks[5].y)**2)
        v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x)**2 + 
                       (eye_landmarks[2].y - eye_landmarks[4].y)**2)
        
        # Horizontal distance
        h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x)**2 + 
                      (eye_landmarks[0].y - eye_landmarks[3].y)**2)
        
        if h < 0.001:
            return 1.0
        
        return (v1 + v2) / (2.0 * h)
    
    def _estimate_head_pose(self, face_landmarks) -> Tuple[float, float]:
        """
        Estimate head yaw and pitch from face landmarks
        Returns (yaw, pitch) in normalized units
        """
        try:
            # Use nose tip (1) and face edges for pose estimation
            nose_tip = face_landmarks[1]
            left_face = face_landmarks[234]  # Left face edge
            right_face = face_landmarks[454]  # Right face edge
            forehead = face_landmarks[10]  # Forehead center
            chin = face_landmarks[152]  # Chin
            
            # Estimate yaw (left-right rotation)
            face_center_x = (left_face.x + right_face.x) / 2
            yaw = (nose_tip.x - face_center_x) * 2  # Normalized -1 to 1
            
            # Estimate pitch (up-down rotation)
            face_center_y = (forehead.y + chin.y) / 2
            pitch = (nose_tip.y - face_center_y) * 2  # Normalized -1 to 1
            
            return (yaw, pitch)
        except:
            return (0.0, 0.0)
    
    def _apply_temporal_filter(
        self, 
        gaze_x: float, 
        gaze_y: float,
        timestamp_ms: int
    ) -> Tuple[float, float]:
        """
        Apply temporal filtering with velocity prediction
        Reduces jitter while maintaining responsiveness
        """
        # Calculate time delta
        dt = 0.033  # Default ~30fps
        if self._last_timestamp is not None:
            dt = max(0.001, (timestamp_ms - self._last_timestamp) / 1000.0)
        self._last_timestamp = timestamp_ms
        
        # Predict position based on velocity
        predicted_x = self._last_gaze_x + self._velocity_x * dt
        predicted_y = self._last_gaze_y + self._velocity_y * dt
        
        # Calculate movement from prediction
        movement_x = gaze_x - predicted_x
        movement_y = gaze_y - predicted_y
        movement = math.sqrt(movement_x**2 + movement_y**2)
        
        # Adaptive smoothing: more smoothing for small movements
        if movement < 0.02:
            smoothing = self._max_smoothing
        elif movement > 0.1:
            smoothing = self._min_smoothing
        else:
            # Linear interpolation
            t = (movement - 0.02) / (0.1 - 0.02)
            smoothing = self._max_smoothing - t * (self._max_smoothing - self._min_smoothing)
        
        # Apply smoothing
        smoothed_x = smoothing * self._last_gaze_x + (1 - smoothing) * gaze_x
        smoothed_y = smoothing * self._last_gaze_y + (1 - smoothing) * gaze_y
        
        # Update velocity estimate
        if dt > 0.001:
            new_vel_x = (smoothed_x - self._last_gaze_x) / dt
            new_vel_y = (smoothed_y - self._last_gaze_y) / dt
            # Smooth velocity update
            vel_smooth = 0.5
            self._velocity_x = vel_smooth * self._velocity_x + (1 - vel_smooth) * new_vel_x
            self._velocity_y = vel_smooth * self._velocity_y + (1 - vel_smooth) * new_vel_y
        
        return (smoothed_x, smoothed_y)
    
    def _is_outlier(self, gaze_x: float, gaze_y: float) -> bool:
        """Check if gaze position is an outlier (sudden jump)"""
        jump = math.sqrt(
            (gaze_x - self._last_gaze_x)**2 + 
            (gaze_y - self._last_gaze_y)**2
        )
        return jump > self._max_jump
    
    def get_gaze(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get gaze coordinates with optimized accuracy
        Returns None if no face detected or during blink
        """
        if not self.initialized or self.face_landmarker is None:
            return None
        
        self.frame_count += 1
        
        try:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # No face detected
            if not results.face_landmarks or len(results.face_landmarks) == 0:
                if self.debug_output and self.frame_count % 30 == 0:
                    print(f"[Frame {self.frame_count}] ⚠ No face detected")
                self._last_gaze_x = 0.5
                self._last_gaze_y = 0.5
                return None
            
            face_landmarks = results.face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Landmark indices
            LEFT_IRIS_CENTER = 468
            RIGHT_IRIS_CENTER = 473
            
            # Left eye landmarks for EAR: [33, 160, 158, 133, 153, 144]
            LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
            
            # Eye corner landmarks
            LEFT_EYE_LEFT = 33
            LEFT_EYE_RIGHT = 133
            RIGHT_EYE_LEFT = 362
            RIGHT_EYE_RIGHT = 263
            
            if len(face_landmarks) <= LEFT_IRIS_CENTER:
                return None
            
            # === BLINK DETECTION ===
            # Use MediaPipe blendshapes if available (more accurate than EAR)
            blink_detected = False

            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                blendshapes = results.face_blendshapes[0]
                # Find blink blendshapes
                left_blink = 0.0
                right_blink = 0.0
                for bs in blendshapes:
                    if bs.category_name == 'eyeBlinkLeft':
                        left_blink = bs.score
                    elif bs.category_name == 'eyeBlinkRight':
                        right_blink = bs.score

                avg_blink = (left_blink + right_blink) / 2.0
                blink_detected = avg_blink > 0.5  # MediaPipe blink score threshold
            else:
                # Fallback to EAR-based detection
                left_eye_ear_landmarks = [face_landmarks[i] for i in LEFT_EYE_EAR]
                right_eye_ear_landmarks = [face_landmarks[i] for i in RIGHT_EYE_EAR]

                left_ear = self._calculate_eye_aspect_ratio(left_eye_ear_landmarks)
                right_ear = self._calculate_eye_aspect_ratio(right_eye_ear_landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                blink_detected = avg_ear < self._blink_threshold

            if blink_detected:
                self._is_blinking = True
                self._blink_frames += 1

                # During short blinks, return last good gaze
                if self._blink_frames <= self._max_blink_frames:
                    return (self._last_gaze_x, self._last_gaze_y)
                else:
                    # Long blink - return None to indicate no valid gaze
                    return None
            else:
                self._is_blinking = False
                self._blink_frames = 0
            
            # === HEAD POSE ESTIMATION ===
            head_yaw, head_pitch = self._estimate_head_pose(face_landmarks)
            
            # Smooth head pose
            head_smooth = 0.7
            self._last_head_yaw = head_smooth * self._last_head_yaw + (1 - head_smooth) * head_yaw
            self._last_head_pitch = head_smooth * self._last_head_pitch + (1 - head_smooth) * head_pitch
            
            # === IRIS TRACKING ===
            left_iris = face_landmarks[LEFT_IRIS_CENTER]
            right_iris = face_landmarks[RIGHT_IRIS_CENTER]

            # Eye corner landmarks (horizontal - stable for X reference)
            left_eye_left = face_landmarks[LEFT_EYE_LEFT]    # 33 - inner corner
            left_eye_right = face_landmarks[LEFT_EYE_RIGHT]  # 133 - outer corner
            right_eye_left = face_landmarks[RIGHT_EYE_LEFT]  # 362 - inner corner
            right_eye_right = face_landmarks[RIGHT_EYE_RIGHT] # 263 - outer corner

            # For Y-axis, we need landmarks that DON'T move with eyelid
            # Use the eye corners for X (they're horizontally stable)
            # But for Y, we need a different approach - use the midpoint between
            # upper and lower eyelid at the iris position, or use iris Y directly
            # relative to a stable facial feature

            # Stable Y reference: use the horizontal line through eye corners
            # The corners don't move much vertically with eye opening
            left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2.0
            right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2.0

            # For Y center, use the corner Y positions (more stable than eyelids)
            # These are the canthi - they don't move much with blinking/eye opening
            left_eye_center_y = (left_eye_left.y + left_eye_right.y) / 2.0
            right_eye_center_y = (right_eye_left.y + right_eye_right.y) / 2.0

            # Calculate iris offset from eye center
            # X offset: iris position relative to horizontal center of eye
            left_iris_offset_x = left_iris.x - left_eye_center_x
            right_iris_offset_x = right_iris.x - right_eye_center_x

            # Y offset: For vertical gaze, we compare iris to a STABLE reference
            # Problem: eye corners move slightly with eye opening
            # Solution: Use additional iris landmarks to get gaze direction directly
            # MediaPipe provides iris landmarks 468-472 (left) and 473-477 (right)
            # The iris center (468, 473) relative to other iris points gives direction

            # Get additional iris landmarks for more robust Y tracking
            LEFT_IRIS_TOP = 469
            LEFT_IRIS_BOTTOM = 471
            RIGHT_IRIS_TOP = 474
            RIGHT_IRIS_BOTTOM = 476

            # Calculate iris vertical center from iris landmarks themselves
            # This is independent of eyelid position
            left_iris_top = face_landmarks[LEFT_IRIS_TOP]
            left_iris_bottom = face_landmarks[LEFT_IRIS_BOTTOM]
            right_iris_top = face_landmarks[RIGHT_IRIS_TOP]
            right_iris_bottom = face_landmarks[RIGHT_IRIS_BOTTOM]

            # The iris center Y from iris landmarks (stable, not affected by eyelid)
            left_iris_geometric_center_y = (left_iris_top.y + left_iris_bottom.y) / 2.0
            right_iris_geometric_center_y = (right_iris_top.y + right_iris_bottom.y) / 2.0

            # Y offset: compare iris center to eye corner line
            # When looking up: iris moves up relative to corner line
            # When looking down: iris moves down relative to corner line
            left_iris_offset_y = left_iris_geometric_center_y - left_eye_center_y
            right_iris_offset_y = right_iris_geometric_center_y - right_eye_center_y

            # Average both eyes
            iris_offset_x = (left_iris_offset_x + right_iris_offset_x) / 2.0
            iris_offset_y = (left_iris_offset_y + right_iris_offset_y) / 2.0

            # Eye width for normalization (stable horizontal reference)
            left_eye_width = abs(left_eye_right.x - left_eye_left.x)
            right_eye_width = abs(right_eye_right.x - right_eye_left.x)
            avg_eye_width = (left_eye_width + right_eye_width) / 2.0
            
            # Normalize by eye width (stable reference)
            if avg_eye_width > 0.005:
                normalized_x = iris_offset_x / avg_eye_width
                normalized_y_raw = iris_offset_y / avg_eye_width
            else:
                normalized_x = iris_offset_x * 50
                normalized_y_raw = iris_offset_y * 50
            
            # === HEAD POSE COMPENSATION ===
            # When head turns RIGHT (positive yaw), iris appears to shift LEFT in camera
            # We ADD the head pose to compensate and keep gaze stable
            normalized_x += self._last_head_yaw * self._head_pose_weight
            normalized_y_raw += self._last_head_pitch * self._head_pose_weight
            
            # Apply Y-axis calibration offset
            normalized_y = normalized_y_raw + self.normalized_y_offset
            
            # === MAPPING TO SCREEN ===
            # With correct scale (~1.8), normalized values map directly to screen
            # normalized_x of ±0.3 (typical) -> ±0.54 -> screen edges
            # normalized_x of ±0.5 (max) -> ±0.9 -> beyond screen (clamped)

            mapped_x = normalized_x * self.scale_x
            mapped_y = normalized_y * self.scale_y

            # Center on screen (0.5) and apply calibration offsets
            gaze_x = 0.5 + mapped_x + self.offset_x
            gaze_y = 0.5 + mapped_y + self.offset_y

            # Clamp to valid screen range [0, 1]
            gaze_x = max(0.0, min(1.0, gaze_x))
            gaze_y = max(0.0, min(1.0, gaze_y))
            
            # === OUTLIER REJECTION ===
            is_outlier = self._is_outlier(gaze_x, gaze_y)
            if is_outlier:
                self._outlier_count += 1
                if self._outlier_count <= 2:
                    # Allow 1-2 outlier frames, use last good value
                    gaze_x = self._last_gaze_x
                    gaze_y = self._last_gaze_y
                else:
                    # Accept the jump after multiple outliers (genuine fast movement)
                    self._outlier_count = 0
            else:
                self._outlier_count = 0
            
            # === TEMPORAL FILTERING ===
            smoothed_x, smoothed_y = self._apply_temporal_filter(gaze_x, gaze_y, timestamp_ms)
            
            # Update state
            self._last_gaze_x = smoothed_x
            self._last_gaze_y = smoothed_y
            
            # Add to history for analysis
            self._gaze_history.append((smoothed_x, smoothed_y))
            
            # Logging
            self._log_data(
                frame=self.frame_count,
                iris_offset_x=iris_offset_x,
                iris_offset_y=iris_offset_y,
                eye_width_avg=avg_eye_width,
                normalized_x=normalized_x,
                normalized_y_raw=normalized_y_raw,
                normalized_y_calibrated=normalized_y,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                smoothed_x=smoothed_x,
                smoothed_y=smoothed_y,
                head_yaw=self._last_head_yaw,
                head_pitch=self._last_head_pitch,
                is_blink=self._is_blinking,
                is_outlier=is_outlier
            )
            
            # Debug output
            if self.debug_output and self.frame_count % 10 == 0:
                print(f"[Frame {self.frame_count}] ✓ Face detected")
                print(f"  EAR: {avg_ear:.3f} (blink threshold: {self._blink_threshold})")
                print(f"  HeadPose: yaw={self._last_head_yaw:.3f}, pitch={self._last_head_pitch:.3f}")
                print(f"  Normalized: X={normalized_x:.4f}, Y={normalized_y:.4f}")
                print(f"  Gaze: X={smoothed_x:.3f}, Y={smoothed_y:.3f}")
                print(f"  Screen: ({int(smoothed_x*w)}, {int(smoothed_y*h)})")
            
            return (smoothed_x, smoothed_y)
            
        except Exception as e:
            if self.debug_output:
                print(f"Error in gaze tracking: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def set_sensitivity(self, scale_x: float = None, scale_y: float = None):
        """Adjust gaze sensitivity"""
        if scale_x is not None:
            self.scale_x = scale_x
        if scale_y is not None:
            self.scale_y = scale_y
        print(f"Sensitivity: X={self.scale_x:.1f}, Y={self.scale_y:.1f}")
    
    def set_head_pose_weight(self, weight: float):
        """Set how much head pose affects gaze (0.0 to 1.0)"""
        self._head_pose_weight = max(0.0, min(1.0, weight))
        print(f"Head pose weight: {self._head_pose_weight:.2f}")
    
    def set_smoothing(self, base: float = None, min_val: float = None, max_val: float = None):
        """Adjust smoothing parameters"""
        if base is not None:
            self._base_smoothing = max(0.0, min(1.0, base))
        if min_val is not None:
            self._min_smoothing = max(0.0, min(1.0, min_val))
        if max_val is not None:
            self._max_smoothing = max(0.0, min(1.0, max_val))
        print(f"Smoothing: base={self._base_smoothing:.2f}, min={self._min_smoothing:.2f}, max={self._max_smoothing:.2f}")
    
    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug output"""
        self.debug_output = enabled
        print(f"Debug output: {'enabled' if enabled else 'disabled'}")
    
    def enable_logging(self, enabled: bool = True):
        """Enable or disable logging"""
        self.log_enabled = enabled
        if enabled and self.log_file is None:
            self._init_logging()
        print(f"Logging: {'enabled' if enabled else 'disabled'}")

    def auto_calibrate(self, frame: np.ndarray, target_x: float, target_y: float) -> bool:
        """
        Collect calibration sample when user looks at known target position.
        Call this multiple times with different targets, then call apply_calibration().

        Args:
            frame: Current camera frame
            target_x: Target X position (0.0-1.0, where user should be looking)
            target_y: Target Y position (0.0-1.0)

        Returns:
            True if sample collected successfully
        """
        if not hasattr(self, '_calibration_samples'):
            self._calibration_samples = []

        # Get raw gaze without filtering
        gaze = self.get_gaze(frame)
        if gaze is None:
            return False

        self._calibration_samples.append({
            'target_x': target_x,
            'target_y': target_y,
            'gaze_x': gaze[0],
            'gaze_y': gaze[1]
        })
        return True

    def compute_calibration(self) -> dict:
        """
        Compute calibration parameters from collected samples.
        Uses least squares to find optimal scale and offset.

        Returns:
            Dict with calibration results and accuracy metrics
        """
        if not hasattr(self, '_calibration_samples') or len(self._calibration_samples) < 4:
            return {'success': False, 'error': 'Need at least 4 calibration samples'}

        samples = self._calibration_samples

        # Extract data
        target_x = np.array([s['target_x'] for s in samples])
        target_y = np.array([s['target_y'] for s in samples])
        gaze_x = np.array([s['gaze_x'] for s in samples])
        gaze_y = np.array([s['gaze_y'] for s in samples])

        # Compute linear fit: target = scale * gaze + offset
        # For X axis
        if np.std(gaze_x) > 0.01:
            slope_x = np.cov(target_x, gaze_x)[0, 1] / np.var(gaze_x)
            intercept_x = np.mean(target_x) - slope_x * np.mean(gaze_x)
        else:
            slope_x = 1.0
            intercept_x = 0.0

        # For Y axis
        if np.std(gaze_y) > 0.01:
            slope_y = np.cov(target_y, gaze_y)[0, 1] / np.var(gaze_y)
            intercept_y = np.mean(target_y) - slope_y * np.mean(gaze_y)
        else:
            slope_y = 1.0
            intercept_y = 0.0

        # Compute accuracy metrics
        predicted_x = slope_x * gaze_x + intercept_x
        predicted_y = slope_y * gaze_y + intercept_y
        error_x = np.sqrt(np.mean((target_x - predicted_x) ** 2))
        error_y = np.sqrt(np.mean((target_y - predicted_y) ** 2))

        # Convert to our internal parameters
        # Our mapping: gaze = 0.5 + normalized * scale + offset
        # Inverse: normalized * scale = gaze - 0.5 - offset
        # To correct: we need to adjust scale_x, scale_y, offset_x, offset_y

        result = {
            'success': True,
            'samples': len(samples),
            'slope_x': slope_x,
            'slope_y': slope_y,
            'intercept_x': intercept_x,
            'intercept_y': intercept_y,
            'error_x': error_x,
            'error_y': error_y,
            'error_pixels_approx': np.sqrt(error_x**2 + error_y**2) * 1920,  # Assume 1080p
            'recommended_scale_x': self.scale_x * slope_x,
            'recommended_scale_y': self.scale_y * slope_y,
            'recommended_offset_x': intercept_x - 0.5 * (1 - slope_x),
            'recommended_offset_y': intercept_y - 0.5 * (1 - slope_y)
        }

        return result

    def apply_computed_calibration(self, calibration: dict) -> bool:
        """
        Apply computed calibration parameters.

        Args:
            calibration: Result from compute_calibration()

        Returns:
            True if applied successfully
        """
        if not calibration.get('success', False):
            return False

        self.scale_x = calibration['recommended_scale_x']
        self.scale_y = calibration['recommended_scale_y']
        self.offset_x = calibration['recommended_offset_x']
        self.offset_y = calibration['recommended_offset_y']

        print(f"Calibration applied:")
        print(f"  Scale: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")
        print(f"  Offset: X={self.offset_x:.3f}, Y={self.offset_y:.3f}")
        print(f"  Expected error: ~{calibration['error_pixels_approx']:.0f} pixels")

        return True

    def clear_calibration_samples(self):
        """Clear collected calibration samples"""
        self._calibration_samples = []
        print("Calibration samples cleared")
