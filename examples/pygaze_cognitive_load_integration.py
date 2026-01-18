"""
PyGaze Cognitive Load Integration Example

Demonstrates how to integrate PyGaze eye tracking with the
cognitive load calculator for real-time cognitive load monitoring.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pygaze_tracker import PyGazeTracker, EventDetector
from src.metrics.cognitive_load_calculator import CognitiveLoadCalculator
import time
from datetime import datetime


def main():
    """Cognitive load integration example"""
    
    print("=" * 60)
    print("PyGaze Cognitive Load Integration Example")
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
    
    # Initialize cognitive load calculator
    print("\n4. Initializing cognitive load calculator...")
    calculator = CognitiveLoadCalculator(
        weight_gaze=0.6,      # Higher weight for gaze (since we're using eye tracking)
        weight_emotion=0.3,
        weight_mouse=0.1,
        window_duration=5.0,  # 5-second window
        update_interval=2.0    # Update every 2 seconds
    )
    print("   ✓ Calculator ready")
    
    # Initialize event detector
    print("\n5. Initializing event detector...")
    detector = EventDetector(tracker)
    print("   ✓ Event detector ready")
    
    # Main monitoring loop
    print("\n6. Starting cognitive load monitoring (30 seconds)...")
    print("   Move your mouse to simulate eye movements")
    print("   Keep mouse still to create fixations (higher load)")
    print("   Move mouse quickly to create saccades (searching)")
    print("\n   Monitoring...\n")
    
    start_time = time.time()
    last_update = time.time()
    
    while time.time() - start_time < 30.0:
        # Collect gaze sample
        sample = tracker.sample()
        if sample and sample.valid:
            # Update cognitive load calculator with gaze data
            calculator.update_gaze(
                x=sample.x,
                y=sample.y,
                pupil_size=sample.pupil_size,
                timestamp=datetime.now()
            )
        
        # Try to collect events (non-blocking with timeout)
        try:
            fixation = detector.collect_fixation(timeout=0.1, min_duration=0.2)
            if fixation:
                # Long fixations indicate cognitive load
                if fixation.duration > 1.0:
                    print(f"   ⚠ Long fixation detected: {fixation.duration:.2f}s")
        except:
            pass
        
        try:
            saccade = detector.collect_saccade(timeout=0.1)
            if saccade:
                # High saccade frequency indicates searching/confusion
                if saccade.velocity > 500:
                    print(f"   ⚠ Rapid saccade detected: {saccade.velocity:.0f} px/s")
        except:
            pass
        
        # Calculate and display CLI periodically
        if time.time() - last_update >= 2.0:
            result = calculator.calculate_cli(force_update=True)
            if result:
                cli = result['cli']
                level = result['load_level']
                gaze_score = result['gaze_score']
                
                # Visual indicator
                bar_length = int(cli * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                
                print(f"\n   CLI: {cli:.3f} [{level.upper()}] {bar}")
                print(f"      Gaze Score: {gaze_score:.3f}")
                print(f"      Fixation Duration: {result['details']['gaze']['fixation_duration']:.2f}s")
                print(f"      Saccade Frequency: {result['details']['gaze']['saccade_frequency']:.2f}/s")
                print(f"      Gaze Dispersion: {result['details']['gaze']['gaze_dispersion']:.2f}")
                
                # Interpretation
                if cli > 0.7:
                    print(f"      ⚠ WARNING: High cognitive load detected!")
                elif cli < 0.3:
                    print(f"      ✓ Low cognitive load - user appears comfortable")
                
                last_update = time.time()
        
        time.sleep(0.05)  # Small delay to prevent CPU spinning
    
    # Final analysis
    print("\n7. Final Analysis:")
    print("   " + "=" * 56)
    
    # Get final CLI
    final_result = calculator.calculate_cli(force_update=True)
    if final_result:
        print(f"\n   Final CLI: {final_result['cli']:.3f} ({final_result['load_level']})")
        print(f"\n   Gaze Metrics:")
        gaze_details = final_result['details']['gaze']
        print(f"      Mean Fixation Duration: {gaze_details['fixation_duration']:.2f}s")
        print(f"      Saccade Frequency: {gaze_details['saccade_frequency']:.2f}/s")
        print(f"      Gaze Dispersion: {gaze_details['gaze_dispersion']:.2f}")
        print(f"      Search Pattern Score: {gaze_details['search_pattern']:.2f}")
    
    # Analyze events
    fixation_stats = detector.analyze_fixations()
    saccade_stats = detector.analyze_saccades()
    
    print(f"\n   Event Statistics:")
    print(f"      Total Fixations: {fixation_stats['count']}")
    print(f"      Mean Fixation Duration: {fixation_stats['mean_duration']:.3f}s")
    print(f"      Total Saccades: {saccade_stats['count']}")
    print(f"      Mean Saccade Amplitude: {saccade_stats['mean_amplitude']:.1f} px")
    
    # Interpretation
    interpretation = calculator.get_load_interpretation(final_result)
    print(f"\n   Interpretation:")
    print(f"      {interpretation}")
    
    # Stop recording
    print("\n8. Stopping recording...")
    tracker.stop_recording()
    tracker.close()
    print("   ✓ Done")
    
    print("\n" + "=" * 60)
    print("Cognitive load integration example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
