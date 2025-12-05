"""Test script to verify camera connectivity."""

import cv2
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import CAMERA_INDEX


def main():
    """Display camera feed for testing."""
    print(f"Testing camera at index {CAMERA_INDEX}...")
    print("Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera at index {CAMERA_INDEX}")
        sys.exit(1)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Display frame info
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Resolution: {width}x{height}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera Test', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCamera test complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()

