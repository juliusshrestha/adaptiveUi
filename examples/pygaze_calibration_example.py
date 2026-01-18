"""
PyGaze Calibration Example

Demonstrates calibration procedures:
- Full calibration
- Drift correction
- Calibration validation
- Saving/loading calibration data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pygaze_tracker import PyGazeTracker, CalibrationManager
import time


def main():
    """Calibration example"""
    
    print("=" * 60)
    print("PyGaze Calibration Example")
    print("=" * 60)
    
    # Initialize tracker
    print("\n1. Initializing eye tracker...")
    tracker = PyGazeTracker(tracker='dummy')
    
    if not tracker.connect():
        print("ERROR: Failed to connect")
        return
    
    print("   ✓ Connected")
    
    # Initialize calibration manager
    print("\n2. Initializing calibration manager...")
    calibration_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'pygaze_calibration.json'
    )
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
    
    manager = CalibrationManager(tracker, calibration_file=calibration_file)
    print("   ✓ Calibration manager ready")
    
    # Perform calibration
    print("\n3. Performing calibration...")
    print("   (In dummy mode, this will use mouse simulation)")
    if manager.perform_calibration(validate=True, max_attempts=2):
        print("   ✓ Calibration successful and validated")
    else:
        print("   ✗ Calibration failed")
        tracker.close()
        return
    
    # Perform drift correction
    print("\n4. Performing drift correction...")
    if manager.perform_drift_correction(pos=None, fix_triggered=False):
        print("   ✓ Drift correction successful")
    else:
        print("   ✗ Drift correction failed")
    
    # Start recording
    print("\n5. Starting recording...")
    tracker.start_recording()
    print("   ✓ Recording")
    
    # Collect some samples to verify calibration
    print("\n6. Verifying calibration with sample collection...")
    print("   Collecting samples for 3 seconds...")
    
    samples = []
    start_time = time.time()
    while time.time() - start_time < 3.0:
        sample = tracker.sample()
        if sample and sample.valid:
            samples.append(sample)
        time.sleep(0.1)
    
    if samples:
        avg_x = sum(s.x for s in samples) / len(samples)
        avg_y = sum(s.y for s in samples) / len(samples)
        print(f"   ✓ Collected {len(samples)} samples")
        print(f"   Average position: ({avg_x:.1f}, {avg_y:.1f})")
    
    # Get calibration info
    print("\n7. Calibration information:")
    info = manager.get_calibration_info()
    if info:
        print(f"   Tracker Type: {info.get('tracker_type', 'unknown')}")
        print(f"   Success: {info.get('success', False)}")
        print(f"   Timestamp: {info.get('timestamp', 'unknown')}")
    else:
        print("   (No calibration data stored)")
    
    # Save calibration
    print("\n8. Saving calibration data...")
    if manager.save_calibration():
        print(f"   ✓ Calibration saved to {calibration_file}")
    else:
        print("   ✗ Failed to save calibration")
    
    # Stop recording
    print("\n9. Stopping recording...")
    tracker.stop_recording()
    tracker.close()
    print("   ✓ Done")
    
    print("\n" + "=" * 60)
    print("Calibration example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
