"""
Basic PyGaze Example

Demonstrates basic eye tracking operations:
- Connection
- Calibration
- Recording
- Sample collection
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pygaze_tracker import PyGazeTracker
import time


def main():
    """Basic PyGaze usage example"""
    
    print("=" * 60)
    print("PyGaze Basic Example")
    print("=" * 60)
    
    # Initialize tracker (using dummy mode for testing)
    print("\n1. Initializing eye tracker (dummy mode)...")
    tracker = PyGazeTracker(tracker='dummy')
    
    # Connect
    print("2. Connecting to eye tracker...")
    if not tracker.connect():
        print("ERROR: Failed to connect to eye tracker")
        return
    
    print("   ✓ Connected successfully")
    
    # Calibrate
    print("\n3. Starting calibration...")
    print("   (In dummy mode, calibration will use mouse simulation)")
    if not tracker.calibrate():
        print("ERROR: Calibration failed")
        tracker.close()
        return
    
    print("   ✓ Calibration successful")
    
    # Start recording
    print("\n4. Starting recording...")
    if not tracker.start_recording():
        print("ERROR: Failed to start recording")
        tracker.close()
        return
    
    print("   ✓ Recording started")
    
    # Collect samples
    print("\n5. Collecting gaze samples (5 seconds)...")
    print("   Move your mouse to simulate eye movements")
    
    samples = []
    start_time = time.time()
    while time.time() - start_time < 5.0:
        sample = tracker.sample()
        if sample and sample.valid:
            samples.append(sample)
            print(f"   Sample: x={sample.x:.1f}, y={sample.y:.1f}, "
                  f"pupil={sample.pupil_size if sample.pupil_size else 'N/A'}")
        time.sleep(0.1)
    
    print(f"\n   ✓ Collected {len(samples)} samples")
    
    # Stop recording
    print("\n6. Stopping recording...")
    tracker.stop_recording()
    print("   ✓ Recording stopped")
    
    # Close
    print("\n7. Closing eye tracker...")
    tracker.close()
    print("   ✓ Closed")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
