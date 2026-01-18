"""
PyGaze Event Detection Example

Demonstrates event detection capabilities:
- Fixation detection
- Saccade detection
- Blink detection
- Event analysis
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pygaze_tracker import PyGazeTracker, EventDetector
import time


def main():
    """Event detection example"""
    
    print("=" * 60)
    print("PyGaze Event Detection Example")
    print("=" * 60)
    
    # Initialize tracker
    print("\n1. Initializing eye tracker...")
    tracker = PyGazeTracker(tracker='dummy')
    
    if not tracker.connect():
        print("ERROR: Failed to connect")
        return
    
    print("   ✓ Connected")
    
    # Calibrate
    print("\n2. Calibrating...")
    if not tracker.calibrate():
        print("ERROR: Calibration failed")
        tracker.close()
        return
    
    print("   ✓ Calibrated")
    
    # Start recording
    print("\n3. Starting recording...")
    tracker.start_recording()
    print("   ✓ Recording")
    
    # Initialize event detector
    print("\n4. Initializing event detector...")
    detector = EventDetector(tracker)
    print("   ✓ Event detector ready")
    
    # Collect fixations
    print("\n5. Collecting fixations (10 seconds)...")
    print("   Keep your mouse still to create fixations")
    print("   Move your mouse to create saccades")
    
    start_time = time.time()
    fixation_count = 0
    saccade_count = 0
    
    while time.time() - start_time < 10.0:
        # Try to collect a fixation (with timeout)
        fixation = detector.collect_fixation(timeout=1.0, min_duration=0.2)
        if fixation:
            fixation_count += 1
            print(f"\n   Fixation #{fixation_count}:")
            print(f"      Duration: {fixation.duration:.3f}s")
            print(f"      Position: ({fixation.x:.1f}, {fixation.y:.1f})")
        
        # Try to collect a saccade (with timeout)
        saccade = detector.collect_saccade(timeout=0.5)
        if saccade:
            saccade_count += 1
            print(f"\n   Saccade #{saccade_count}:")
            print(f"      Duration: {saccade.duration:.3f}s")
            print(f"      Amplitude: {saccade.amplitude:.1f} pixels")
            print(f"      Velocity: {saccade.velocity:.1f} pixels/s")
    
    print(f"\n   ✓ Collected {fixation_count} fixations and {saccade_count} saccades")
    
    # Analyze events
    print("\n6. Analyzing events...")
    
    fixation_stats = detector.analyze_fixations()
    print(f"\n   Fixation Statistics:")
    print(f"      Count: {fixation_stats['count']}")
    print(f"      Mean Duration: {fixation_stats['mean_duration']:.3f}s")
    print(f"      Mean Position: ({fixation_stats['mean_x']:.1f}, {fixation_stats['mean_y']:.1f})")
    print(f"      Dispersion: {fixation_stats['dispersion']:.1f} pixels")
    
    saccade_stats = detector.analyze_saccades()
    print(f"\n   Saccade Statistics:")
    print(f"      Count: {saccade_stats['count']}")
    print(f"      Mean Amplitude: {saccade_stats['mean_amplitude']:.1f} pixels")
    print(f"      Mean Velocity: {saccade_stats['mean_velocity']:.1f} pixels/s")
    
    blink_stats = detector.analyze_blinks()
    print(f"\n   Blink Statistics:")
    print(f"      Count: {blink_stats['count']}")
    print(f"      Mean Duration: {blink_stats['mean_duration']:.3f}s")
    print(f"      Frequency: {blink_stats['frequency']:.1f} blinks/min")
    
    # Stop recording
    print("\n7. Stopping recording...")
    tracker.stop_recording()
    tracker.close()
    print("   ✓ Done")
    
    print("\n" + "=" * 60)
    print("Event detection example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
