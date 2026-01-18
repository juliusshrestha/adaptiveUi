#!/usr/bin/env python3
"""Test to see actual Y-axis values in real-time (development utility)."""

import cv2
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker


def main():
    tracker = DirectGazeTracker()
    tracker.debug_output = True
    tracker.log_enabled = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    print("=" * 80)
    print("REAL-TIME Y-AXIS DEBUGGING")
    print("=" * 80)
    print("Instructions:")
    print("1. Look at the CENTER of your screen (marker should be in middle)")
    print("2. Look at the TOP of your screen (marker should move up)")
    print("3. Look at the BOTTOM of your screen (marker should move down)")
    print("4. Press 'q' to quit and see log analysis")
    print("=" * 80)
    print()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Get gaze
        gaze = tracker.get_gaze(frame)

        if gaze:
            x, y = gaze
            px, py = int(x * w), int(y * h)

            # Draw marker
            cv2.circle(frame, (px, py), 20, (0, 255, 0), 3)
            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

            # Draw reference lines
            cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)  # Horizontal center
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)  # Vertical center

            # Status text with color coding
            if y < 0.3:
                status = "TOP"
                color = (0, 0, 255)
            elif y > 0.7:
                status = "BOTTOM"
                color = (255, 0, 0)
            else:
                status = "CENTER"
                color = (0, 255, 0)

            cv2.putText(frame, f"Gaze Y: {y:.3f} ({status})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Gaze X: {x:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Y-Axis Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Close logging and show analysis
    tracker.close_logging()
    print("\nTest complete. Check the log file for detailed analysis.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test to see actual Y-axis values in real-time"""
import cv2
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.direct_gaze_tracker import DirectGazeTracker

tracker = DirectGazeTracker()
tracker.debug_output = True
tracker.log_enabled = True

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)

print("="*80)
print("REAL-TIME Y-AXIS DEBUGGING")
print("="*80)
print("Instructions:")
print("1. Look at the CENTER of your screen (marker should be in middle)")
print("2. Look at the TOP of your screen (marker should move up)")
print("3. Look at the BOTTOM of your screen (marker should move down)")
print("4. Press 'q' to quit and see log analysis")
print("="*80)
print()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Get gaze
    gaze = tracker.get_gaze(frame)
    
    if gaze:
        x, y = gaze
        px, py = int(x * w), int(y * h)
        
        # Draw marker
        cv2.circle(frame, (px, py), 20, (0, 255, 0), 3)
        cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
        
        # Draw reference lines
        cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)  # Horizontal center
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)  # Vertical center
        
        # Status text with color coding
        if y < 0.3:
            status = "TOP"
            color = (0, 0, 255)
        elif y > 0.7:
            status = "BOTTOM"
            color = (255, 0, 0)
        else:
            status = "CENTER"
            color = (0, 255, 0)
        
        cv2.putText(frame, f"Gaze Y: {y:.3f} ({status})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Gaze X: {x:.3f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Y-Axis Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Close logging and show analysis
tracker.close_logging()
print("\nTest complete. Check the log file for detailed analysis.")
