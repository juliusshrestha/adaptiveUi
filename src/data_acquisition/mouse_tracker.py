"""
Mouse Movement and Click Tracker

Tracks mouse behavior for Cognitive Load Index (CLI) calculation:
- Click rate and error clicks (frustration indicator)
- Path efficiency / meander ratio (uncertainty indicator)
- Hover dwell time (decisional load indicator)
- Movement speed and variance (confidence indicator)

Uses pynput for cross-platform mouse event capture.
"""

import math
import time
import threading
from typing import Optional, Tuple, List, Dict, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import numpy as np

try:
    from pynput import mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not installed. Mouse tracking disabled.")
    print("Install with: pip install pynput")


@dataclass
class MouseEvent:
    """Represents a single mouse event"""
    x: int
    y: int
    timestamp: float
    event_type: str  # 'move', 'click', 'scroll'
    button: Optional[str] = None  # 'left', 'right', 'middle'
    pressed: bool = False


@dataclass
class ClickEvent:
    """Detailed click event for error detection"""
    x: int
    y: int
    timestamp: float
    button: str
    is_double_click: bool = False
    is_rapid_click: bool = False  # Part of rapid clicking sequence
    target_element: Optional[str] = None  # UI element clicked (if trackable)


@dataclass
class MouseMetricsSnapshot:
    """Current state of mouse metrics for CLI calculation"""
    # Click metrics
    click_rate: float = 0.0  # Clicks per second (last 5 seconds)
    error_click_rate: float = 0.0  # Rapid/repeated clicks per second
    double_click_rate: float = 0.0  # Double clicks per second
    
    # Path metrics
    path_efficiency: float = 1.0  # 0=very meandering, 1=direct path
    meander_ratio: float = 1.0  # actual_distance / optimal_distance
    direction_changes: int = 0  # Number of direction reversals
    
    # Timing metrics
    avg_hover_dwell: float = 0.0  # Average hover time before click (seconds)
    max_hover_dwell: float = 0.0  # Maximum hover time
    time_between_clicks: float = 0.0  # Average time between clicks
    
    # Speed metrics
    avg_speed: float = 0.0  # Average movement speed (pixels/second)
    speed_variance: float = 0.0  # Variance in movement speed
    acceleration_variance: float = 0.0  # Variance in acceleration
    
    # Activity metrics
    idle_time: float = 0.0  # Time since last movement (seconds)
    total_distance: float = 0.0  # Total distance traveled
    movement_time: float = 0.0  # Time spent moving
    
    # Computed load score
    load_score: float = 0.0  # 0-1 cognitive load from mouse behavior


class MouseTracker:
    """
    Comprehensive mouse tracking for cognitive load analysis
    
    Usage:
        tracker = MouseTracker()
        tracker.start()
        
        # Get current metrics
        metrics = tracker.get_metrics()
        print(f"Click rate: {metrics.click_rate}")
        print(f"Path efficiency: {metrics.path_efficiency}")
        
        tracker.stop()
    """
    
    def __init__(
        self,
        history_duration: float = 10.0,  # Seconds of history to keep
        click_window: float = 5.0,  # Window for click rate calculation
        rapid_click_threshold: float = 0.3,  # Clicks faster than this = rapid
        double_click_threshold: float = 0.5,  # Time window for double click
        hover_threshold: float = 0.5,  # Time stationary before it's a "hover"
        movement_threshold: int = 5,  # Pixels - below this is considered stationary
        screen_width: int = 1920,
        screen_height: int = 1080
    ):
        """
        Initialize mouse tracker.
        
        Args:
            history_duration: How many seconds of movement history to keep
            click_window: Time window for calculating click rates
            rapid_click_threshold: Clicks faster than this are "error" clicks
            double_click_threshold: Max time between clicks for double-click
            hover_threshold: Time stationary to count as hovering
            movement_threshold: Pixel distance below which movement is ignored
            screen_width: Screen width for normalization
            screen_height: Screen height for normalization
        """
        self.history_duration = history_duration
        self.click_window = click_window
        self.rapid_click_threshold = rapid_click_threshold
        self.double_click_threshold = double_click_threshold
        self.hover_threshold = hover_threshold
        self.movement_threshold = movement_threshold
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Event storage
        self.movement_history: deque = deque(maxlen=10000)
        self.click_history: deque = deque(maxlen=1000)
        self.hover_events: deque = deque(maxlen=100)
        
        # Current state
        self.current_x = 0
        self.current_y = 0
        self.last_move_time = time.time()
        self.last_click_time = 0.0
        self.last_click_pos: Optional[Tuple[int, int]] = None
        
        # Hover tracking
        self.hover_start_time: Optional[float] = None
        self.hover_position: Optional[Tuple[int, int]] = None
        
        # Path tracking (for efficiency calculation)
        self.path_start: Optional[Tuple[int, int]] = None
        self.path_start_time: Optional[float] = None
        self.path_distance = 0.0
        
        # Listener
        self.listener: Optional[mouse.Listener] = None
        self.running = False
        self._lock = threading.Lock()
        
        # Callbacks for external integration
        self.on_click_callback: Optional[Callable] = None
        self.on_move_callback: Optional[Callable] = None
        
        # Metrics cache
        self._cached_metrics: Optional[MouseMetricsSnapshot] = None
        self._cache_time: float = 0.0
        self._cache_duration: float = 0.1  # Recalculate every 100ms
    
    def start(self) -> bool:
        """Start tracking mouse events"""
        if not PYNPUT_AVAILABLE:
            print("Cannot start mouse tracker: pynput not available")
            return False
        
        if self.running:
            return True
        
        try:
            self.listener = mouse.Listener(
                on_move=self._on_move,
                on_click=self._on_click,
                on_scroll=self._on_scroll
            )
            self.listener.start()
            self.running = True
            self.last_move_time = time.time()
            print("✓ Mouse tracker started")
            return True
        except Exception as e:
            print(f"Failed to start mouse tracker: {e}")
            return False
    
    def stop(self):
        """Stop tracking mouse events"""
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.running = False
        print("✓ Mouse tracker stopped")
    
    def _on_move(self, x: int, y: int):
        """Handle mouse move event"""
        current_time = time.time()
        
        with self._lock:
            # Calculate distance from last position
            if self.movement_history:
                last_event = self.movement_history[-1]
                dx = x - last_event.x
                dy = y - last_event.y
                distance = math.sqrt(dx*dx + dy*dy)
            else:
                distance = 0
                dx, dy = 0, 0
            
            # Only record if movement is significant
            if distance >= self.movement_threshold:
                event = MouseEvent(
                    x=x,
                    y=y,
                    timestamp=current_time,
                    event_type='move'
                )
                self.movement_history.append(event)
                
                # Update path tracking
                self.path_distance += distance
                
                # Update hover tracking
                self._update_hover_state(x, y, current_time, moving=True)
                
                self.last_move_time = current_time
            else:
                # Stationary - update hover
                self._update_hover_state(x, y, current_time, moving=False)
            
            self.current_x = x
            self.current_y = y
        
        # Call external callback if set
        if self.on_move_callback:
            self.on_move_callback(x, y, current_time)
    
    def _on_click(self, x: int, y: int, button, pressed: bool):
        """Handle mouse click event"""
        if not pressed:  # Only track press, not release
            return
        
        current_time = time.time()
        
        with self._lock:
            # Determine button name
            button_name = 'left'
            if hasattr(button, 'name'):
                button_name = button.name
            elif str(button) == 'Button.right':
                button_name = 'right'
            elif str(button) == 'Button.middle':
                button_name = 'middle'
            
            # Check for double click
            is_double = False
            if self.last_click_time > 0:
                time_since_last = current_time - self.last_click_time
                if time_since_last < self.double_click_threshold:
                    if self.last_click_pos:
                        dist = math.sqrt(
                            (x - self.last_click_pos[0])**2 + 
                            (y - self.last_click_pos[1])**2
                        )
                        if dist < 20:  # Within 20 pixels
                            is_double = True
            
            # Check for rapid clicking
            is_rapid = False
            if self.last_click_time > 0:
                time_since_last = current_time - self.last_click_time
                if time_since_last < self.rapid_click_threshold:
                    is_rapid = True
            
            # Create click event
            click = ClickEvent(
                x=x,
                y=y,
                timestamp=current_time,
                button=button_name,
                is_double_click=is_double,
                is_rapid_click=is_rapid
            )
            self.click_history.append(click)
            
            # Record hover dwell if we were hovering before click
            if self.hover_start_time is not None:
                hover_duration = current_time - self.hover_start_time
                if hover_duration >= self.hover_threshold:
                    self.hover_events.append({
                        'duration': hover_duration,
                        'x': self.hover_position[0] if self.hover_position else x,
                        'y': self.hover_position[1] if self.hover_position else y,
                        'timestamp': current_time
                    })
            
            # Reset path tracking (new path starts after click)
            self.path_start = (x, y)
            self.path_start_time = current_time
            self.path_distance = 0.0
            
            # Reset hover
            self.hover_start_time = None
            self.hover_position = None
            
            # Update last click info
            self.last_click_time = current_time
            self.last_click_pos = (x, y)
        
        # Call external callback if set
        if self.on_click_callback:
            self.on_click_callback(x, y, button_name, current_time)
    
    def _on_scroll(self, x: int, y: int, dx: int, dy: int):
        """Handle mouse scroll event"""
        current_time = time.time()
        
        with self._lock:
            event = MouseEvent(
                x=x,
                y=y,
                timestamp=current_time,
                event_type='scroll'
            )
            self.movement_history.append(event)
    
    def _update_hover_state(self, x: int, y: int, current_time: float, moving: bool):
        """Update hover tracking state"""
        if moving:
            # Movement detected - check if we should record hover
            if self.hover_start_time is not None:
                hover_duration = current_time - self.hover_start_time
                if hover_duration >= self.hover_threshold:
                    # Was a significant hover, don't record unless clicked
                    pass
            
            # Reset hover tracking
            self.hover_start_time = None
            self.hover_position = None
        else:
            # Stationary
            if self.hover_start_time is None:
                self.hover_start_time = current_time
                self.hover_position = (x, y)
    
    def _clean_old_events(self, current_time: float):
        """Remove events older than history_duration"""
        cutoff = current_time - self.history_duration
        
        # Clean movement history
        while self.movement_history and self.movement_history[0].timestamp < cutoff:
            self.movement_history.popleft()
        
        # Keep click history longer (for rate calculations)
        click_cutoff = current_time - max(self.click_window * 2, 30.0)
        while self.click_history and self.click_history[0].timestamp < click_cutoff:
            self.click_history.popleft()
    
    def get_metrics(self) -> MouseMetricsSnapshot:
        """
        Calculate and return current mouse metrics.
        
        Returns:
            MouseMetricsSnapshot with all calculated metrics
        """
        current_time = time.time()
        
        # Use cached metrics if recent enough
        if (self._cached_metrics is not None and 
            current_time - self._cache_time < self._cache_duration):
            return self._cached_metrics
        
        with self._lock:
            self._clean_old_events(current_time)
            
            metrics = MouseMetricsSnapshot()
            
            # === CLICK METRICS ===
            click_cutoff = current_time - self.click_window
            recent_clicks = [c for c in self.click_history if c.timestamp > click_cutoff]
            
            if recent_clicks:
                metrics.click_rate = len(recent_clicks) / self.click_window
                
                rapid_clicks = [c for c in recent_clicks if c.is_rapid_click]
                metrics.error_click_rate = len(rapid_clicks) / self.click_window
                
                double_clicks = [c for c in recent_clicks if c.is_double_click]
                metrics.double_click_rate = len(double_clicks) / self.click_window
                
                # Time between clicks
                if len(recent_clicks) >= 2:
                    intervals = []
                    for i in range(1, len(recent_clicks)):
                        interval = recent_clicks[i].timestamp - recent_clicks[i-1].timestamp
                        intervals.append(interval)
                    metrics.time_between_clicks = np.mean(intervals)
            
            # === PATH METRICS ===
            if len(self.movement_history) >= 2:
                positions = [(e.x, e.y) for e in self.movement_history]
                
                # Calculate actual path distance
                actual_distance = 0.0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    actual_distance += math.sqrt(dx*dx + dy*dy)
                
                # Calculate optimal (straight line) distance
                optimal_distance = math.sqrt(
                    (positions[-1][0] - positions[0][0])**2 +
                    (positions[-1][1] - positions[0][1])**2
                )
                
                # Path efficiency
                if actual_distance > 0:
                    metrics.path_efficiency = min(1.0, optimal_distance / actual_distance)
                    metrics.meander_ratio = actual_distance / max(1, optimal_distance)
                
                # Count direction changes
                if len(positions) >= 3:
                    direction_changes = 0
                    for i in range(2, len(positions)):
                        # Vector from i-2 to i-1
                        v1 = (positions[i-1][0] - positions[i-2][0],
                              positions[i-1][1] - positions[i-2][1])
                        # Vector from i-1 to i
                        v2 = (positions[i][0] - positions[i-1][0],
                              positions[i][1] - positions[i-1][1])
                        
                        # Check for direction reversal (dot product < 0)
                        dot = v1[0]*v2[0] + v1[1]*v2[1]
                        if dot < 0:
                            direction_changes += 1
                    
                    metrics.direction_changes = direction_changes
                
                metrics.total_distance = actual_distance
            
            # === HOVER/DWELL METRICS ===
            recent_hovers = [h for h in self.hover_events 
                           if h['timestamp'] > current_time - self.history_duration]
            
            if recent_hovers:
                hover_durations = [h['duration'] for h in recent_hovers]
                metrics.avg_hover_dwell = np.mean(hover_durations)
                metrics.max_hover_dwell = np.max(hover_durations)
            
            # Current hover time (if hovering now)
            if self.hover_start_time is not None:
                current_hover = current_time - self.hover_start_time
                metrics.max_hover_dwell = max(metrics.max_hover_dwell, current_hover)
            
            # === SPEED METRICS ===
            if len(self.movement_history) >= 2:
                speeds = []
                accelerations = []
                prev_speed = None
                
                events = list(self.movement_history)
                for i in range(1, len(events)):
                    dt = events[i].timestamp - events[i-1].timestamp
                    if dt > 0.001:  # Avoid division by zero
                        dx = events[i].x - events[i-1].x
                        dy = events[i].y - events[i-1].y
                        distance = math.sqrt(dx*dx + dy*dy)
                        speed = distance / dt
                        speeds.append(speed)
                        
                        if prev_speed is not None:
                            acceleration = (speed - prev_speed) / dt
                            accelerations.append(abs(acceleration))
                        prev_speed = speed
                
                if speeds:
                    metrics.avg_speed = np.mean(speeds)
                    metrics.speed_variance = np.std(speeds) / (np.mean(speeds) + 1)
                    
                    # Calculate movement time (total time with movement)
                    first_time = events[0].timestamp
                    last_time = events[-1].timestamp
                    metrics.movement_time = last_time - first_time
                
                if accelerations:
                    metrics.acceleration_variance = np.std(accelerations) / (np.mean(accelerations) + 1)
            
            # === IDLE TIME ===
            metrics.idle_time = current_time - self.last_move_time
            
            # === COMPUTE LOAD SCORE ===
            metrics.load_score = self._compute_load_score(metrics)
            
            # Cache the result
            self._cached_metrics = metrics
            self._cache_time = current_time
            
            return metrics
    
    def _compute_load_score(self, metrics: MouseMetricsSnapshot) -> float:
        """
        Compute cognitive load score from mouse metrics.
        
        High load indicators:
        - High click rate (> 1.5/s) = trial and error
        - High error click rate = frustration
        - Low path efficiency (< 0.5) = searching/uncertain
        - High hover dwell (> 1.5s) = decision difficulty
        - High speed variance = hesitation
        
        Returns:
            Load score from 0.0 (low load) to 1.0 (high load)
        """
        scores = []
        
        # Click rate: > 2/s is high load
        if metrics.click_rate > 0:
            click_score = min(1.0, metrics.click_rate / 3.0)
            scores.append(click_score)
        
        # Error click rate: any rapid clicking is concerning
        if metrics.error_click_rate > 0:
            error_score = min(1.0, metrics.error_click_rate / 1.0)
            scores.append(error_score * 1.5)  # Weight errors higher
        
        # Path efficiency: inverted (low efficiency = high load)
        efficiency_score = 1.0 - metrics.path_efficiency
        scores.append(efficiency_score)
        
        # Direction changes: many reversals = searching
        if metrics.direction_changes > 0:
            direction_score = min(1.0, metrics.direction_changes / 20.0)
            scores.append(direction_score)
        
        # Hover dwell: long hovers before click = decision load
        if metrics.avg_hover_dwell > 0.5:
            hover_score = min(1.0, (metrics.avg_hover_dwell - 0.5) / 3.0)
            scores.append(hover_score)
        
        # Speed variance: high variance = hesitation
        if metrics.speed_variance > 0:
            variance_score = min(1.0, metrics.speed_variance)
            scores.append(variance_score)
        
        # Calculate weighted average
        if scores:
            return min(1.0, np.mean(scores))
        return 0.0
    
    def get_normalized_position(self) -> Tuple[float, float]:
        """Get current mouse position normalized to [0, 1]"""
        return (
            self.current_x / self.screen_width,
            self.current_y / self.screen_height
        )
    
    def set_callbacks(
        self,
        on_click: Optional[Callable] = None,
        on_move: Optional[Callable] = None
    ):
        """Set callback functions for mouse events"""
        self.on_click_callback = on_click
        self.on_move_callback = on_move
    
    def get_recent_clicks(self, duration: float = 5.0) -> List[ClickEvent]:
        """Get list of recent click events"""
        cutoff = time.time() - duration
        with self._lock:
            return [c for c in self.click_history if c.timestamp > cutoff]
    
    def get_click_heatmap_data(self, duration: float = 60.0) -> List[Tuple[int, int]]:
        """Get click positions for heatmap visualization"""
        cutoff = time.time() - duration
        with self._lock:
            return [(c.x, c.y) for c in self.click_history if c.timestamp > cutoff]
    
    def reset(self):
        """Reset all tracking data"""
        with self._lock:
            self.movement_history.clear()
            self.click_history.clear()
            self.hover_events.clear()
            self.path_distance = 0.0
            self.hover_start_time = None
            self.hover_position = None
            self.last_click_time = 0.0
            self.last_click_pos = None
            self._cached_metrics = None


# Convenience function for quick testing
def test_mouse_tracker(duration: float = 10.0):
    """Test mouse tracker for specified duration"""
    if not PYNPUT_AVAILABLE:
        print("pynput not available. Install with: pip install pynput")
        return
    
    print(f"Testing mouse tracker for {duration} seconds...")
    print("Move the mouse and click to generate metrics.")
    print()
    
    tracker = MouseTracker()
    tracker.start()
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            time.sleep(1.0)
            metrics = tracker.get_metrics()
            
            print(f"\n[{time.time() - start_time:.1f}s] Mouse Metrics:")
            print(f"  Click rate: {metrics.click_rate:.2f}/s")
            print(f"  Error clicks: {metrics.error_click_rate:.2f}/s")
            print(f"  Path efficiency: {metrics.path_efficiency:.2%}")
            print(f"  Meander ratio: {metrics.meander_ratio:.2f}")
            print(f"  Avg hover dwell: {metrics.avg_hover_dwell:.2f}s")
            print(f"  Speed variance: {metrics.speed_variance:.2f}")
            print(f"  Direction changes: {metrics.direction_changes}")
            print(f"  Load score: {metrics.load_score:.2%}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        tracker.stop()
    
    print("\nTest complete!")


if __name__ == "__main__":
    test_mouse_tracker(30.0)
