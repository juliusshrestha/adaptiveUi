"""
PyGaze Gaze-Contingent Display Example

Demonstrates a gaze-contingent display where a dot follows the gaze position.
This is similar to the example in the PyGaze documentation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pygaze_tracker import PyGazeTracker
import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")


def main():
    """Gaze-contingent display example"""
    
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame is required for this example")
        print("Install with: pip install pygame")
        return
    
    print("=" * 60)
    print("PyGaze Gaze-Contingent Display Example")
    print("=" * 60)
    print("\nA red dot will follow your gaze position.")
    print("Press SPACE to exit.\n")
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Gaze-Contingent Display")
    clock = pygame.time.Clock()
    
    # Initialize tracker
    print("1. Initializing eye tracker...")
    tracker = PyGazeTracker(tracker='dummy')
    
    if not tracker.connect():
        print("ERROR: Failed to connect")
        pygame.quit()
        return
    
    print("   ✓ Connected")
    
    # Calibrate
    print("2. Calibrating...")
    if not tracker.calibrate():
        print("ERROR: Calibration failed")
        tracker.close()
        pygame.quit()
        return
    
    print("   ✓ Calibrated")
    
    # Start recording
    print("3. Starting recording...")
    tracker.start_recording()
    print("   ✓ Recording started")
    print("\nDisplay active. Move your mouse to see the dot follow.")
    print("Press SPACE to exit.\n")
    
    # Main loop
    running = True
    dot_pos = (400, 300)  # Center of screen
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False
        
        # Get gaze sample
        sample = tracker.sample()
        if sample and sample.valid:
            # Update dot position (scale to screen size)
            dot_pos = (int(sample.x), int(sample.y))
        
        # Draw
        screen.fill((0, 0, 0))  # Black background
        pygame.draw.circle(screen, (255, 0, 0), dot_pos, 10)  # Red dot
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    # Cleanup
    print("\n4. Stopping recording...")
    tracker.stop_recording()
    tracker.close()
    pygame.quit()
    
    print("   ✓ Done")
    print("\n" + "=" * 60)
    print("Gaze-contingent example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
